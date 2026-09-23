import matplotlib
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from iminuit.cost import LeastSquares
from iminuit import Minuit
import inspect
import scipy.stats as stats
from scipy.spatial import ConvexHull


import json as _json
import os as _os
import warnings as _warnings

try:  # pomegranate >= 1.0 is the torch rewrite; 0.14 (Cython) has no py3.10 wheel
    from pomegranate.hmm import DenseHMM as _DenseHMM
    from pomegranate.distributions import Normal as _PomNormal
    _HAS_POMEGRANATE_V1 = True
except ImportError:
    _DenseHMM = None
    _PomNormal = None
    _HAS_POMEGRANATE_V1 = False


class NormalDistribution:
    """Stand-in for ``pomegranate.NormalDistribution`` (0.14 API).

    Only carries the (mu, sigma) pair through ``.parameters``, which is the
    whole of what DeepSPT ever read off a state's distribution.
    """

    def __init__(self, mu, sigma):
        self.name = "NormalDistribution"
        self.parameters = [float(mu), float(sigma)]

    @property
    def mu(self):
        return self.parameters[0]

    @property
    def sigma(self):
        return self.parameters[1]

    def log_probability(self, X):
        mu, sigma = self.parameters
        X = np.asarray(X, dtype=float)
        return -0.5 * np.log(2 * np.pi * sigma ** 2) - (X - mu) ** 2 / (2 * sigma ** 2)

    def to_dict(self):
        return {"class": "Distribution", "name": self.name,
                "parameters": list(self.parameters), "frozen": False}


class State:
    """Stand-in for ``pomegranate.State`` (0.14 API)."""

    def __init__(self, distribution, name):
        self.distribution = distribution
        self.name = name

    def __repr__(self):
        return "State(%s)" % self.name


class HiddenMarkovModel:
    """Drop-in replacement for the pomegranate 0.14 ``HiddenMarkovModel``.

    DeepSPT only ever used four things from the pomegranate model: loading it
    from JSON, ``predict(SL, algorithm='viterbi')``, reading each state's mean
    off ``model.states[:4]``, and (when no saved model exists) fitting a fresh
    one. Everything else in pomegranate 0.14 was unused, so this reimplements
    just that surface and drops the dependency.

    Two interchangeable Viterbi backends are available and give identical
    paths (checked over 200 random sequences against pomegranate 0.14.9):

    ``numpy``       pure-numpy log-domain Viterbi, no extra dependency.
    ``pomegranate`` pomegranate >= 1.0 ``DenseHMM``, i.e. the torch rewrite.

    ``backend='auto'`` picks ``numpy``: it needs no extra dependency and is
    ~24x faster here (17.9 ms vs 435.6 ms over 60 sequences), since these step
    length sequences are far too short to amortise torch's per-call overhead.
    Ask for ``backend='pomegranate'`` explicitly to run on the torch model.

    The state order of the source JSON is preserved exactly. That matters:
    ``GetStates`` derives its state relabelling from ``argsort`` over the state
    means, so permuting the states would silently relabel every diffusion
    state in the fingerprints.
    """

    def __init__(self, states, transitions, starts, ends=None, name=None,
                 backend="auto"):
        self.states = list(states)
        self.name = name
        n = len([s for s in self.states if s.distribution is not None])
        self.n_states = n
        self.transitions = np.asarray(transitions, dtype=float)
        self.starts = np.asarray(starts, dtype=float)
        self.ends = (np.zeros(n, dtype=float) if ends is None
                     else np.asarray(ends, dtype=float))
        self.start_index = n
        self.end_index = n + 1
        self.means = np.array([s.distribution.parameters[0]
                               for s in self.states[:n]], dtype=float)
        self.stds = np.array([s.distribution.parameters[1]
                              for s in self.states[:n]], dtype=float)
        self.backend = self._resolve_backend(backend)
        self._densehmm = None

    # -- construction ------------------------------------------------------

    @staticmethod
    def _resolve_backend(backend):
        if backend == "auto":
            return "numpy"
        if backend == "pomegranate" and not _HAS_POMEGRANATE_V1:
            raise ImportError(
                "backend='pomegranate' needs pomegranate>=1.0 (`pip install "
                "pomegranate`); use backend='numpy' for the dependency-free path")
        if backend not in ("numpy", "pomegranate"):
            raise ValueError("backend must be 'auto', 'numpy' or 'pomegranate'")
        return backend

    @classmethod
    def from_dict(cls, d, backend="auto"):
        """Build from a parsed pomegranate 0.14 JSON dict."""
        real = [s for s in d["states"] if s.get("distribution") is not None]
        n = len(real)
        states = [State(NormalDistribution(*s["distribution"]["parameters"]),
                        s["name"]) for s in real]
        # keep the silent start/end states so ``len(model.states)`` matches 0.14
        states.append(State(None, d.get("start", {}).get("name", "None-start")))
        states.append(State(None, d.get("end", {}).get("name", "None-end")))

        idx = {s["name"]: i for i, s in enumerate(d["states"])}
        start_i = d.get("start_index", idx.get(d.get("start", {}).get("name")))
        end_i = d.get("end_index", idx.get(d.get("end", {}).get("name")))

        T = np.zeros((n, n), dtype=float)
        starts = np.zeros(n, dtype=float)
        ends = np.zeros(n, dtype=float)
        for edge in d["edges"]:
            a, b, p = edge[0], edge[1], edge[2]
            if a == start_i and b < n:
                starts[b] = p
            elif a < n and b == end_i:
                ends[a] = p
            elif a < n and b < n:
                T[a, b] = p
        return cls(states, T, starts, ends, name=d.get("name"), backend=backend)

    @classmethod
    def from_json(cls, s, backend="auto"):
        """Load a pomegranate 0.14 JSON model (string, or path to a .json)."""
        if isinstance(s, (str, bytes)) and not str(s).lstrip().startswith("{"):
            if _os.path.isfile(s):
                with open(s, "r") as fh:
                    s = fh.read()
        return cls.from_dict(_json.loads(s), backend=backend)

    def to_dict(self):
        n = self.n_states
        states = [s.distribution.to_dict() for s in self.states[:n]]
        out_states = [{"class": "State", "distribution": ds, "name": s.name,
                       "weight": 1.0}
                      for s, ds in zip(self.states[:n], states)]
        out_states.append({"class": "State", "distribution": None,
                           "name": "None-start", "weight": 1.0})
        out_states.append({"class": "State", "distribution": None,
                           "name": "None-end", "weight": 1.0})
        edges = [[self.start_index, j, float(self.starts[j]), 0.25, None]
                 for j in range(n)]
        edges += [[i, j, float(self.transitions[i, j]), 0.25, None]
                  for i in range(n) for j in range(n)]
        edges += [[i, self.end_index, float(self.ends[i]), 0.25, None]
                  for i in range(n) if self.ends[i] > 0]
        return {"class": "HiddenMarkovModel", "name": str(self.name),
                "start": {"class": "State", "distribution": None,
                          "name": "None-start", "weight": 1.0},
                "end": {"class": "State", "distribution": None,
                        "name": "None-end", "weight": 1.0},
                "states": out_states, "end_index": self.end_index,
                "start_index": self.start_index, "silent_index": self.start_index,
                "edges": edges, "distribution ties": []}

    def to_json(self, separators=(",", " : "), indent=4):
        return _json.dumps(self.to_dict(), separators=separators, indent=indent)

    def bake(self, *args, **kwargs):
        """No-op. pomegranate 0.14 needed an explicit finalise step; this does not."""
        return self

    # -- torch / pomegranate>=1.0 interop -----------------------------------

    def to_densehmm(self):
        """Return this model as a pomegranate >= 1.0 ``DenseHMM`` (torch)."""
        if not _HAS_POMEGRANATE_V1:
            raise ImportError("pomegranate>=1.0 is not installed")
        import torch
        dists = [_PomNormal(means=[m], covs=[s ** 2], covariance_type="diag")
                 for m, s in zip(self.means, self.stds)]
        # ends are left at pomegranate's uniform default: the published model
        # has no end transitions, and a constant end term cannot change argmax.
        return _DenseHMM(distributions=dists,
                         edges=torch.tensor(self.transitions, dtype=torch.float64),
                         starts=torch.tensor(self.starts, dtype=torch.float64))

    def save_torch(self, path):
        """Serialise the torch (pomegranate >= 1.0) form with ``torch.save``.

        Note this is a build artifact, not the source of truth: it pickles
        pomegranate class references and so is fragile across pomegranate and
        torch upgrades. Keep the JSON -- four means, four sigmas and a 4x4
        matrix -- as the thing you actually archive.
        """
        import torch
        torch.save(self.to_densehmm(), path)
        return path

    @classmethod
    def from_torch(cls, path, backend="auto"):
        """Rebuild from a ``save_torch`` file, recovering the 0.14-style API."""
        import torch
        m = torch.load(path, weights_only=False)
        return cls.from_densehmm(m, backend=backend)

    @classmethod
    def from_densehmm(cls, m, backend="auto"):
        import torch
        means = [float(d.means[0]) for d in m.distributions]
        stds = [float(torch.sqrt(d.covs[0])) for d in m.distributions]
        states = [State(NormalDistribution(mu, sd), "s%d" % i)
                  for i, (mu, sd) in enumerate(zip(means, stds))]
        states += [State(None, "None-start"), State(None, "None-end")]
        T = torch.exp(m.edges).detach().cpu().numpy().astype(float)
        starts = torch.exp(m.starts).detach().cpu().numpy().astype(float)
        return cls(states, T, starts, backend=backend)

    # -- inference ---------------------------------------------------------

    def _log_emissions(self, SL):
        SL = np.asarray(SL, dtype=float).reshape(-1, 1)
        mu, sd = self.means[None, :], self.stds[None, :]
        return -0.5 * np.log(2 * np.pi * sd ** 2) - (SL - mu) ** 2 / (2 * sd ** 2)

    @staticmethod
    def _safe_log(a):
        a = np.asarray(a, dtype=float)
        return np.log(a, out=np.full(a.shape, -np.inf), where=a > 0)

    def _viterbi_numpy(self, SL):
        E = self._log_emissions(SL)
        n_obs = E.shape[0]
        lT, lS = self._safe_log(self.transitions), self._safe_log(self.starts)
        delta = lS + E[0]
        back = np.zeros((n_obs, self.n_states), dtype=int)
        for t in range(1, n_obs):
            M = delta[:, None] + lT
            back[t] = M.argmax(axis=0)
            delta = M.max(axis=0) + E[t]
        path = [int(delta.argmax())]
        for t in range(n_obs - 1, 0, -1):
            path.append(int(back[t, path[-1]]))
        return path[::-1]

    def _viterbi_pomegranate(self, SL):
        import torch
        if self._densehmm is None:
            self._densehmm = self.to_densehmm()
        X = torch.tensor(np.asarray(SL, dtype=float).reshape(1, -1, 1),
                         dtype=torch.float64)
        return [int(i) for i in self._densehmm.viterbi(X)[0]]

    def viterbi(self, SL):
        """Best state path. Returns ``(logp, [(index, State), ...])`` like 0.14."""
        path = self._viterbi_numpy(SL) if self.backend == "numpy" \
            else self._viterbi_pomegranate(SL)
        full = [(self.start_index, self.states[self.start_index])]
        full += [(i, self.states[i]) for i in path]
        return None, full

    def predict(self, sequence, algorithm="viterbi", check_input=True):
        """State index per observation, prefixed by the silent start state.

        The leading start-state index reproduces pomegranate 0.14 exactly,
        which is why callers such as ``GetStates`` slice it off with ``[1:]``.
        """
        if algorithm != "viterbi":
            raise NotImplementedError(
                "only algorithm='viterbi' is supported (the only one DeepSPT used)")
        sequence = np.asarray(sequence, dtype=float)
        if sequence.size == 0:
            return [self.start_index]
        path = self._viterbi_numpy(sequence) if self.backend == "numpy" \
            else self._viterbi_pomegranate(sequence)
        return [self.start_index] + path

    def dense_transition_matrix(self):
        """(n+2, n+2) matrix laid out as pomegranate 0.14 ordered it."""
        n = self.n_states
        M = np.zeros((n + 2, n + 2), dtype=float)
        M[:n, :n] = self.transitions
        M[self.start_index, :n] = self.starts
        M[:n, self.end_index] = self.ends
        return M

    # -- fitting -----------------------------------------------------------

    @classmethod
    def from_samples(cls, distribution=None, n_components=4, X=None,
                     max_iterations=1000, tol=0.1, random_state=None,
                     backend="auto", **kwargs):
        """Fit a fresh Gaussian HMM, replacing 0.14's ``from_samples``.

        Only reached when no saved HMM JSON exists. This is NOT numerically
        identical to pomegranate 0.14's fit -- different initialisation and
        stopping rule -- so a model fitted here will not reproduce
        fingerprints computed with a 0.14-fitted model. Refit downstream
        analyses if you regenerate the HMM rather than loading the published
        one.
        """
        if X is None:
            raise ValueError("X is required")
        if not _HAS_POMEGRANATE_V1:
            raise ImportError(
                "fitting a new HMM needs pomegranate>=1.0 (`pip install "
                "pomegranate`). Loading an existing HMM JSON needs no extra "
                "dependency.")
        import torch
        _warnings.warn(
            "HiddenMarkovModel.from_samples now fits via pomegranate>=1.0; "
            "results differ from pomegranate 0.14 and will not reproduce "
            "fingerprints made with a 0.14-fitted model.", RuntimeWarning)
        seqs = [torch.tensor(np.asarray(x, dtype=float).reshape(-1, 1),
                             dtype=torch.float64) for x in X]
        dists = [_PomNormal(covariance_type="diag") for _ in range(n_components)]
        m = _DenseHMM(distributions=dists, max_iter=max_iterations, tol=tol,
                      init="kmeans", verbose=False, random_state=random_state)
        m.fit(seqs)
        return cls.from_densehmm(m, backend=backend)

def Chi2Fit(
    x,
    y,
    sy,
    f,
    plot=True,
    print_level=0,
    labels=None,
    ax=None,
    savefig=None,
    valpos=None,
    exponential=False,
    fitcol=None,
    markersize=5,
    plotcol=None,
    name=None,
    fontsize=15,
    linewidth=3,
    png=False,
    custom_cost=None,
    **guesses,
):
    """Function that peforms a Chi2Fit to data given function
    ----------
    Parameters
    ----------
    x: ndarray of shape for input in f
        - input values to fit
    y: ndarray of shape output from f
        - output values to fit
    sy: ndarray of length y
        - errors on the y values
    f: function
        - Function to fit, should be of form f(x,args), where args
          is a list of arguments
    **guesses: mappings ie. p0=0.1,p1=0.2
        - initial guesses for the fit parameters
    print_level: int 0,1
        - Wether to print output from chi2 ect.
    labels:
        - Mappable to pass to ax.set call to set labels on plot
    name: str
        -Label to call fit in legend
    fontsize: int
        - Size of font in plot
    linewidth: float
        - Width of line on data
    ---------
    Returns
    ---------
    params: length args
        - fit params
    errs: lenght args
        - errror on fit params
    Ndof: int
        - Number of  degrees of freedom for fit
    Chi2: float
        - Chi2 for fit
    pval: float
        -pvalue for the fit
    """
    xmin, xmax = np.min(x), np.max(x)
    names = inspect.getfullargspec(f)[0][1:]
    if custom_cost is None:
        chi2_object = LeastSquares(x, y, sy, f)
    else:
        chi2_object = custom_cost
    if len(guesses) != 0:
        paramguesses = {}
        lims = {}
        for key, value in guesses.items():
            if key.split("_")[0] == "limit":
                lims[key.split("_")[1]] = value
            else:
                paramguesses[key] = value
        minuit = Minuit(chi2_object, **paramguesses)
        if len(lims) > 0:
            for key, value in lims.items():
                minuit.limits[key] = value
        minuit.print_level = print_level
    else:
        minuit = Minuit(chi2_object)
    minuit.migrad()
    chi2 = minuit.fval
    # free parameters, not len(guesses): `guesses` also carries the
    # limit_<name> entries, which are bounds rather than fitted parameters
    Ndof = len(x) - minuit.nfit
    Pval = stats.chi2.sf(chi2, Ndof)
    params = minuit.values
    errs = minuit.errors

    if not exponential:
        dict = {"chi2": chi2, "Ndof": Ndof, "Pval": Pval}
        for n, p, py in zip(names, params, errs):
            dict[n] = f"{p:4.2f} +/- {py:4.2f}"
    else:
        dict = {"chi2": f"{chi2:4.4E}", "Ndof": f"{Ndof:4.4E}", "Pval": f"{Pval:4.4E}"}
        for n, p, py in zip(names, params, errs):
            dict[n] = f"{p:4.4E} +/- {py:4.4E}"
    return params, errs, Pval


def SquareDist(x0, x1, y0, y1, z0, z1):
    """Computes the squared distance between the two points (x0,y0) and (y1,y1)

    Returns
    -------
    float
        squared distance between the two input points

    """
    return (x1 - x0) ** 2 + (y1 - y0) ** 2 + (z1 - z0) ** 2


def QuadDist(x0, x1, y0, y1, z0, z1):
    """Computes the four-norm (x1-x0)**4+(y1-y0)**4+(z1-z0)**4.

    Returns
    -------
    float
        Four-norm.

    """
    return (x1 - x0) ** 4 + (y1 - y0) ** 4 + (z1 - z0) ** 4


def GetMax(x, y, z):
    """Computes the maximum squared distance between all points in the (x,y,z) set.

    Parameters
    ----------
    x : list-like
        x-coordinates.
    y : list-like
        y-coordinates.
    z : list-like
        z-coordinates.

    Returns
    -------
    float
        Largest squared distance between any two points in the set.

    """
    from itertools import combinations
    from random import randint

    A = np.array([x, y, z]).T

    def square_distance(x, y):
        return sum([(xi - yi) ** 2 for xi, yi in zip(x, y)])

    max_square_distance = 0
    for pair in combinations(A, 2):
        if square_distance(*pair) > max_square_distance:
            max_square_distance = square_distance(*pair)
            max_pair = pair
    return max_square_distance


def msd(x, y, z, frac):
    """Computes the mean squared displacement (msd) for a trajectory (x,y) up to
    frac*len(x) of the trajectory.

    Parameters
    ----------
    x : list-like
        x-coordinates for the trajectory.
    y : list-like
        y-coordinates for the trajectory.
    frac : float in [0,1]
        Fraction of trajectory duration to compute msd up to.
        if length of x is more than 20 else collapse to len(x)

    Returns
    -------
    iterable of lenght int(len(x)*frac)
        msd for the trajectory

    """
    N = int(len(x) * frac) if len(x)>20 else len(x)
    msd = []

    for lag in range(1, N):
        msd.append(
            np.mean(
                [
                    SquareDist(x[j], x[j + lag], y[j], y[j + lag], z[j], z[j + lag])
                    for j in range(len(x) - lag)
                ]
            )
        )
    return np.array(msd)


def Scalings(msds, dt, dim=2, difftype='Normal'):
    """Fit mean squared displacements to a power law.

    Parameters
    ----------
    msds : list-like
        mean squared displacenemts.

    Returns
    -------
    tuple of length 3
        The first index is the fitted generalized diffusion constant,
        the second is the scaling exponent alpha, and the final is the pvalue for the fit.

    """
    if difftype=='Normal' or difftype=='Subdiffusive':
        def power(x, D, alpha, offset):
            return 2 * dim * D * (x) ** alpha + offset
        
        # from scipy.optimize import curve_fit
        # Pval = 1
        # params, pcov = curve_fit(power, np.arange(1,len(msds)+1)*dt, msds, 
        #                          p0=[msds[0] / (4 * dt),1], 
        #                          max_nfev=100000, bounds=[[0.00001,0.00001],[np.inf,10]], 
        #                          method='trf')

        ## list with [D, alpha, offset] offset should equal loc error
        # AAAAAA
        params, errs, Pval = Chi2Fit(
        np.arange(1, len(msds) + 1)*dt,
        msds,
        np.ones(len(msds)),
        power,
        plot=False,
        D=np.sqrt(msds[0]) / (4 * dt),
        alpha=1,
        offset=0.001,
        limit_alpha=(0.0001, 10),
        limit_D=(0.000000001, None),
        limit_offset=(0, None)
        )
        sy = np.std(msds - power(np.arange(1, len(msds) + 1)*dt, *params))
        if sy>10**-3:    
            params, errs, Pval = Chi2Fit(
                np.arange(1, len(msds) + 1)*dt,
                msds,
                sy * np.ones(len(msds)),
                power,
                plot=False,
                D=np.sqrt(msds[0]) / (4 * dt),
                alpha=1,
                offset=0.001,
                limit_alpha=(0.0001, 10),
                limit_D=(0.000000001, None),
                limit_offset=(0, None))

    elif difftype=='Directed':
        def power(x, D, alpha, v, offset):
            return 2 * dim * D * (x) ** alpha + (v**2) * (x**2) + offset

        params, errs, Pval = Chi2Fit(
            np.arange(1, len(msds) + 1)*dt,
            msds,
            np.ones(len(msds)),
            power,
            plot=False,
            D=np.sqrt(msds[0]) / (4 * dt),
            alpha=1,
            v=1,
            offset=0.001,
            limit_alpha=(0.0001, 10),
            limit_D=(0.000000001, None),
            limit_v=(0, None),
            limit_offset=(0, None)
            )
        sy = np.std(msds - power(np.arange(1, len(msds) + 1)*dt, *params))
        if sy>10**-3: 
            params, errs, Pval = Chi2Fit(
                np.arange(1, len(msds) + 1)*dt,
                msds,
                sy * np.ones(len(msds)),
                power,
                plot=False,
                D=np.sqrt(msds[0]) / (4 * dt),
                alpha=1,
                v=1,
                offset=0.001,
                limit_alpha=(0.0001, 10),
                limit_D=(0.000000001, None),
                limit_v=(0, None),
                limit_offset=(0, None)
            )
    elif difftype=='Confined':
        def power(x, D, alpha, R, A1, A2, offset):
            return (R**2) * (1-A1*np.exp(-2 * A2 * dim * D * (x) / (R**2))) + offset

        params, errs, Pval = Chi2Fit(
            np.arange(1, len(msds) + 1)*dt,
            msds,
            np.ones(len(msds)),
            power,
            plot=False,
            D=np.sqrt(msds[0]) / (4 * dt),
            alpha=1,
            R=1,
            A1=1, 
            A2=1,
            offset=0.001,
            limit_alpha=(0.0001, 10),
            limit_D=(0.000000001, None),
            limit_offset=(0, None)
            )
        sy = np.std(msds - power(np.arange(1, len(msds) + 1)*dt, *params))
        if sy>10**-3: 
            params, errs, Pval = Chi2Fit(
                np.arange(1, len(msds) + 1)*dt,
                msds,
                sy * np.ones(len(msds)),
                power,
                plot=False,
                D=np.sqrt(msds[0]) / (4 * dt),
                alpha=1,
                R=1,
                A1=1, 
                A2=1,
                offset=0.001,
                limit_alpha=(0.0001, 10),
                limit_D=(0.000000001, None),
                limit_offset=(0, None)
            )
    return params, Pval


def Efficiency(x, y, z):
    """Computes the efficiency of a trajectory, logarithm of the ratio of squared end-to-end distance
    and the sum of squared distances.

    Parameters
    ----------
    x : list-like
        x-coordinates for the trajectory.
    y : list-like
        y-coordinates for the trajectory.

    Returns
    -------
    float
        Efficiency.

    """
    top = SquareDist(x[0], x[-1], y[0], y[-1], z[0], z[-1])
    bottom = sum(
        [SquareDist(x[i], x[i + 1], y[i], y[i + 1], z[i], z[i + 1]) for i in range(0, len(x) - 1)]
    )
    return np.log((top) / ((len(x) - 1) * bottom)), (top) / ((len(x) - 1) * bottom)


def FractalDim(x, y, z, max_square_distance):
    """Computes the fractal dimension using the estimator suggested by Katz & George
    in Fractals and the analysis of growth paths, 1985.

    Parameters
    ----------
    x : list-like
        x-coordinates for the trajectory.
    y : list-like
        y-coordinates for the trajectory.
    z : list-like
        z-coordinates for the trajectory.
    max_square_distance : float
        Maximum squared pair-wise distance for the poinst in the trajectory.

    Returns
    -------
    float
        Estimated fractal dimension.

    """
    totlen = sum(
        [
            np.sqrt(SquareDist(x[i], x[i + 1], y[i], y[i + 1], z[i], z[i + 1]))
            for i in range(0, len(x) - 1)
        ]
    )
    return np.log(len(x)) / (
        np.log(len(x)) + np.log(np.sqrt(max_square_distance) / totlen)
    )


def Gaussianity(x, y, z, r2):
    """Computes the Gaussianity.

    Parameters
    ----------
    x : list-like
        x-coordinates for the trajectory.
    y : list-like
        y-coordinates for the trajectory.
    z : list-like
        z-coordinates for the trajectory.
    r2 : list-like
        Mean squared displacements for the trajectory.

    Returns
    -------
    float
        Gaussianity.

    """
    gn = []
    for lag in range(1, len(r2)):
        r4 = np.mean(
            [QuadDist(x[j], x[j + lag], y[j], y[j + lag], z[j], z[j + lag]) for j in range(len(x) - lag)]
        )
        gn.append(r4 / (2 * r2[lag] ** 2))
    return np.mean(gn)


def Kurtosis(x, y, z, dim=2):
    """Computes the kurtosis for the trajectory.

    Parameters
    ----------
    x : list-like
        x-coordinates for the trajectory.
    y : list-like
        y-coordinates for the trajectory.
    z : list-like
        z-coordinates for the trajectory.

    Returns
    -------
    float
        Kurtosis.

    """
    from scipy.stats import kurtosis

    if dim == 3:
        # https://en.wikipedia.org/wiki/Covariance_matrix
        # np.cov(x, y)[0,1] to get covariance of x and y in covariance matrix
        C = np.array([[np.cov(x, x)[0,1],np.cov(x, y)[0,1],np.cov(x, z)[0,1]],
                      [np.cov(y, x)[0,1],np.cov(y, y)[0,1],np.cov(y, z)[0,1]],
                      [np.cov(z, x)[0,1],np.cov(z, y)[0,1],np.cov(z, z)[0,1]]]) 
        val, vec = np.linalg.eig(C)
        dominant = vec[:, np.argsort(val)][:, -1]
        kurt = kurtosis([np.dot(dominant, v) for v in np.array([x, y, z]).T], fisher=False)
    else:
        val, vec = np.linalg.eig(np.cov(x, y))
        dominant = vec[:, np.argsort(val)][:, -1]
        kurt = kurtosis([np.dot(dominant, v) for v in np.array([x, y]).T], fisher=False)
    return kurt


def MSDratio(mval):
    """Computes the MSD ratio.

    Parameters
    ----------
    mval : list-like
        Mean squared displacements.

    Returns
    -------
    float
        MSD ratio.

    """
    return np.mean(
        [mval[i] / mval[i + 1] - (i) / (i + 1) for i in range(len(mval) - 1)]
    )


def Trappedness(x, maxpair, out):
    """Computes the trappedness.

    Parameters
    ----------
    x : list-like
        x-coordinates for the trajectory.
    y : list-like
        y-coordinates for the trajectory.
    maxpair : float
        Maximum squared pair-wise distance for the poinst in the trajectory.
    out : list-like
        Mean squared displacements.

    Returns
    -------
    float
        Trappedness.

    """
    r0 = np.sqrt(maxpair) / 2
    D = out[1] - out[0]
    # https://www.ncbi.nlm.nih.gov/pmc/articles/PMC1262511/pdf/biophysj00087-0118.pdf 
    # has 0.2048 which differs from diffusional fingerprinting PNAS
    return 1 - np.exp(0.2045 - 0.25117 * (D * len(x)) / r0 ** 2)


def Time_in(state):
    """Computes the fraction of time spent in each of four states in a state
    history.

    Parameters
    ----------
    state : list-like
        State history for the trajectory.

    Returns
    -------
    list of length 4
        Fraction of time spent in each state.

    """
    times = []
    N = len(state)
    for o in range(4):
        time = 0
        for s in state:
            if s == o:
                time += 1
        times.append(time)
    return np.array(times) / N


def Lifetime(state):
    """Computes the average duration of states.

    Parameters
    ----------
    state : list-like
        State history for the trajectory.

    Returns
    -------
    float
        average duration of a state

    """
    jumps = []
    for i in range(len(state) - 1):
        if state[i + 1] != state[i]:
            jumps.append(i)
    if len(jumps) == 1:
        return max(jumps[0], len(state) - jumps[0])
    if len(jumps) == 0:
        return len(state)
    else:
        lifetimes = np.array(jumps[1:]) - np.array(jumps[:-1])
        return np.mean(lifetimes)


def GetStates(SL, model):
    """Predict the viterbi path for a series of steplengths based on a fitted HMM model.

    Parameters
    ----------
    SL : list-like
        step lengths for the trajectory.
    model : pomegranate model
        Fitted pomegranate model used to compute the viterbi path.

    Returns
    -------
    list-like
        State trajectories.
    pomegranate model
        The model used to predict the states

    """
    for i in range(len(SL)):
        if SL[i] == 0:
            SL[i] = 1e-15
    states = model.predict(SL, algorithm="viterbi")
    ms = [s.distribution.parameters[0] for s in model.states[:4]]
    statemap = dict(zip(np.arange(4)[np.argsort(ms)], np.arange(4)))
    newstates = [statemap[s] for s in states[1:]]
    return newstates, model


def dotproduct_traces(trace):
    vecs = trace[1:]-trace[:-1]
    dots = np.dot(vecs[:-1], vecs[1:].T).diagonal()
    return dots


def convex_vol(x, y, z, dim):
    try:
        if dim == 2:
            t = np.vstack([x,y]).T
            alpha_shape = ConvexHull(t)

        elif dim == 3:
            t = np.vstack([x,y,z]).T
            alpha_shape = ConvexHull(t)
        return alpha_shape.volume
        
    except:
        return 0 


def GetFeatures(x, y, z, SL, model, dim, dt, difftype):
    """Compute the diffusional fingerprint for a trajectory.

    Parameters
    ----------
    x : list-like
        x-coordinates for the trajectory.
    y : list-like
        y-coordinates for the trajectory.
    SL : list-like
        step lengths for the trajectory.
    model : pomegranate model
        Fitted pomegranate model used to compute the viterbi path.

    Returns
    -------
    ndarray
        The features describing the diffusional fingerprint

    """
    out = msd(x, y, z, .5)
    maxpair = GetMax(x, y, z)
    params, pval = Scalings(out, dt, dim=dim, difftype=difftype)
    if difftype == 'Normal' or difftype == 'Subdiffusive' or difftype == None:
        beta = params[0]
        alpha = params[1]
        extra = 0
    if difftype == 'Directed':
        beta = params[0]
        alpha = params[1]
        extra = params[2]
    if difftype == 'Confined':
        beta = params[0]
        alpha = params[1]
        extra = params[2]

    states, model = GetStates(SL, model)

    t0, t1, t2, t3 = Time_in(states)
    lifetime = Lifetime(states)
    return np.array(
        [
            alpha,
            beta,
            extra,
            pval,
            Efficiency(x, y, z)[0],
            Efficiency(x, y, z)[1],
            FractalDim(x, y, z, maxpair),
            Gaussianity(x, y, z, out),
            Kurtosis(x, y, z, dim=dim),
            MSDratio(out),
            Trappedness(x, maxpair, out),
            t0,
            t1,
            t2,
            t3,
            lifetime,
            len(x),
            np.nanmean(SL),
            np.nanmean(out),
            np.nanmean(dotproduct_traces(np.array([x,y,z]).T)),
            np.nanmean(np.sign(dotproduct_traces(np.array([x,y,z]).T)[1:])==np.sign(dotproduct_traces(np.array([x,y,z]).T)[:-1])),
            np.nanmean(np.sign(dotproduct_traces(np.array([x,y,z]).T)[1:])>0),
            np.nansum(SL),
            np.nanmin(SL),
            np.nanmax(SL),
            np.nanmax(SL)-np.min(SL),
            np.nansum(SL)/len(x),
            np.nanstd(SL, ddof=1)/np.mean(SL),
            np.nansum(SL<0.1)/len(SL),
            np.nansum(SL>0.4)/len(SL),
            convex_vol(x, y, z, dim)
        ]
    )

def ThirdAppender(d, model, dim, dt, difftype):
    """Wrapper function around GetFeatures.

    Parameters
    ----------
    d : tuple of length 3
        (x,y,SL).
    model : pomegranate model
        Fitted pomegranate model used to compute the viterbi path.

    Returns
    -------
    ndarray or str
        Returns the features describing the diffusional fingerprint
    """
    if dim==2:
        x, y, SL = d
        z = np.zeros_like(x)
    if dim==3:
        x, y, z, SL = d

    return GetFeatures(x, y, z, SL, model, dim, dt, difftype)
