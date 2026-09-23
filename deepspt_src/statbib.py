# @Author: Henrik Pinholt <henrikpinholt>
# @Date:   20-Apr-2020
# @Email:  henrikpinholt@hotmail.com
# @Last modified by:   henrikpinholt
# @Last modified time: 25-Sep-2020


import numpy as np
import matplotlib.pyplot as plt
from iminuit.cost import LeastSquares


def _probfit_removed(name, replacement):
    def _raise(*args, **kwargs):
        raise NotImplementedError(
            "probfit.%s is unavailable: probfit ships no cp310 wheel and pins "
            "iminuit<2, so it cannot be installed on python 3.10. Every call "
            "site of %s in this module was already dead (it passed the "
            "`pedantic=` kwarg that iminuit dropped in 2.0), so nothing was "
            "ported. Rewrite against iminuit.cost.%s if you need it."
            % (name, name, replacement))
    return _raise


BinnedLH = _probfit_removed("BinnedLH", "BinnedNLL")
BinnedChi2 = _probfit_removed("BinnedChi2", "BinnedNLL")
UnbinnedLH = _probfit_removed("UnbinnedLH", "UnbinnedNLL")
from iminuit import Minuit
import inspect
import scipy.stats as stats
import math as m
import matplotlib
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import pandas as pd

SMALL_SIZE = 15
MEDIUM_SIZE = 18
BIGGER_SIZE = 22

plt.rc("font", size=SMALL_SIZE)  # controls default text sizes
plt.rc("axes", titlesize=SMALL_SIZE)  # fontsize of the axes title
plt.rc("axes", labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
plt.rc("xtick", labelsize=SMALL_SIZE)  # fontsize of the tick labels
plt.rc("ytick", labelsize=SMALL_SIZE)  # fontsize of the tick labels
plt.rc("legend", fontsize=SMALL_SIZE)  # legend fontsize
plt.rc("figure", titlesize=BIGGER_SIZE)  # fontsize of the figure title
"""From External functions"""


def Add_labels_to_ax(ax, fontsize=44):
    for n, a in enumerate(ax):
        a.text(
            0.1,
            0.98,
            ["A", "B", "C", "D", "E", "F", "G", "H"][n],
            transform=a.transAxes,
            fontsize=fontsize,
            va="top",
            ha="right",
        )


def format_value(value, decimals):
    """
    Checks the type of a variable and formats it accordingly.
    Floats has 'decimals' number of decimals.
    """

    if isinstance(value, (float, np.floating)):
        return f"{value:.{decimals}f}"
    elif isinstance(value, (int, np.integer)):
        return f"{value:d}"
    else:
        return f"{value}"


def values_to_string(values, decimals):
    """
    Loops over all elements of 'values' and returns list of strings
    with proper formating according to the function 'format_value'.
    """

    res = []
    for value in values:
        if isinstance(value, list):
            tmp = [format_value(val, decimals) for val in value]
            res.append(f"{tmp[0]} +/- {tmp[1]}")
        else:
            res.append(format_value(value, decimals))
    return res


def len_of_longest_string(s):
    """ Returns the length of the longest string in a list of strings """
    return len(max(s, key=len))


def nice_string_output(d, extra_spacing=5, decimals=3):
    """
    Takes a dictionary d consisting of names and values to be properly formatted.
    Makes sure that the distance between the names and the values in the printed
    output has a minimum distance of 'extra_spacing'. One can change the number
    of decimals using the 'decimals' keyword.
    """

    names = d.keys()
    max_names = len_of_longest_string(names)

    values = values_to_string(d.values(), decimals=decimals)
    max_values = len_of_longest_string(values)

    string = ""
    for name, value in zip(names, values):
        spacing = extra_spacing + max_values + max_names - len(name) - 1
        string += "{name:s} {value:>{spacing}} \n".format(
            name=name, value=value, spacing=spacing
        )
    return string[:-2]


def add_text_to_ax(x_coord, y_coord, string, ax, fontsize=12, color="k"):
    """ Shortcut to add text to an ax with proper font. Relative coords."""
    ax.text(
        x_coord,
        y_coord,
        string,
        family="monospace",
        fontsize=fontsize,
        transform=ax.transAxes,
        verticalalignment="top",
        color=color,
    )
    return None


"""Distributions"""


def Plotdist(
    x,
    y,
    ax=None,
    color="b",
    name=None,
    label={"xlabel": "x", "ylabel": "P(x)"},
    savefig="",
):
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    ax.vlines(x, 0, y, colors=color, lw=3, alpha=0.5)
    if name is None:
        ax.plot(x, y, color, marker="o", linestyle="", ms=4, alpha=0.5)
    else:
        ax.plot(x, y, color, marker="o", linestyle="", ms=4, alpha=0.5, label=name)
    ax.set(**label)
    if savefig != "":
        plt.savefig(savefig + ".pdf", dpi=500)


def Two_Gauss(sigma):
    return np.abs(1 - 2 * stats.norm(0, 1).cdf(sigma))


def One_Gauss(sigma):
    return np.abs(1 - stats.norm(0, 1).cdf(sigma))


def Binomial(k, p, N):
    return (
        m.factorial(N)
        / (m.factorial(k) * m.factorial(N - k))
        * p ** (k)
        * (1 - p) ** (N - k)
    )


def LogNormal(x, mu, sigma):
    return (
        1
        / (x * sigma * np.sqrt(2 * np.pi))
        * np.exp(-np.power(np.log(x) - mu, 2) / (2 * np.power(sigma, 2)))
    )


def Lognormal_extended(x, mu, sigma, N):
    return N * LogNormal(x, mu, sigma)


def Poisson(k, mu):
    return (np.exp(-mu) * mu ** k) / m.factorial(k)


def Poisson_continuous(k, mu, loc=0):
    return stats.poisson.pmf(k, mu, loc=loc)


def BinomialSum(p, N, start, end):
    if end > N:
        raise ValueError("to larger than N!")
    if start < 0:
        raise ValueError("from less than zero N!")
    if end < start:
        raise ValueError("end less than start")
    s = np.sum([Binomial(k, p, N) for k in range(start, end + 1)])
    return s


def PoissonSum(mu, start, end):
    if end < start:
        raise ValueError("end less than start")
    s = np.sum([Poisson(k, mu) for k in range(start, end + 1)])
    return s


def gauss_pdf(x, mu, sigma):
    """Normalized Gaussian"""
    return stats.norm.pdf(x, loc=mu, scale=sigma)


def gauss_extended(x, N, mu, sigma):
    """Non-normalized (extended) Gaussian"""
    return N * gauss_pdf(x, mu, sigma)


def poisson_extended(k, mu, N):
    """Extended Poisson"""
    return N * Poisson(k, mu)


def exponential(x, l):
    return l * np.exp(-l * x)


def exponential_extended(x, l, N):
    return l * np.exp(-l * x) * N


"""Descriptive statistics"""


def Corr(
    samp1,
    samp2,
    ax=None,
    plot=True,
    s=1.5,
    savefig="",
    labels=None,
    bins=None,
    fig=None,
    fontsize=12,
    tpos=(0.1, 0.1),
    png=False,
    color="darkblue",
):
    """Computes Pearsons correlation coefficient
    ----------
    Params
    ----------
    samp1: ndarray or list
        - random variables drawn from a distribution
    samp2: ndarray or list
        - random variables drawn from a distribution
    --------
    Returns
    --------
    float:
        - Pearsons correlation coefficient
    float:
        - pvalue
    The Pearson correlation coefficient measures the linear relationship between two
    datasets. Strictly speaking, Pearson’s correlation requires that each dataset be
    normally distributed. Like other correlation coefficients, this one varies between
    -1 and +1 with 0 implying no correlation. Correlations of -1 or +1 imply an exact
    linear relationship. Positive correlations imply that as x increases, so does y.
    Negative correlations imply that as x increases, y decreases.

    The p-value roughly indicates the probability of an uncorrelated system producing
    datasets that have a Pearson correlation at least as extreme as the one computed
    from these datasets. The p-values are not entirely reliable but are probably
    reasonable for datasets larger than 500 or so.

    Thus the smaller, the more significant
    """
    coff, pval = stats.pearsonr(samp1, samp2)
    if plot:
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        else:
            if fig is None and ax is None:
                raise ValueError("Pass both axis and fig object")
        if bins is None:
            ax.scatter(samp1, samp2, s=s, c=color)
        else:
            counts, xedges, yedges, im = ax.hist2d(samp1, samp2, bins=bins, cmin=1)
            fig.colorbar(im)
        if labels is None:
            ax.set(xlabel="x", ylabel="y")
        else:
            ax.set(**labels)
        dict = {"Corr": f"{coff:4.2f}", "Pval": f"{pval:4.2E}"}
        text = nice_string_output(dict)
        if bins is None:
            add_text_to_ax(tpos[0], tpos[1], text, ax, fontsize=fontsize)
        else:
            add_text_to_ax(tpos[0], tpos[1], text, ax, fontsize=fontsize, color="white")
        if savefig != "":
            if not png:
                plt.savefig(savefig + ".pdf", dpi=500)
            else:
                plt.savefig(savefig, dpi=500)
    return coff, pval


def Malinger(values, sigmas, verbose=True, savetable="", exponential=False):
    """script that takes values, variances and spits out mean, variance and pval
    ---------
    Params:
    ---------
    values:
        - array of measurments
    sigmas:
        - errors on the measurments
    savetable: string
        - name to save a latex table of the calculation
    verbose:
        - if to print out calculations,ndof and chi2(pval)
    ---------
    returns:
    ---------
    mean:
        - weighted mean
    sigma:
        - weighted error on the mean
    pval:
        - pvalue from a chi2 of the weighted mean
    """
    # Get list over inverse variances used in weight calculation
    sigmas = np.array(sigmas)
    inversevar, inversevart = 0, []
    for i in sigmas:
        inversevar += 1 / (i ** 2)
        inversevart.append(1 / (i ** 2))
    # Calculate weighted mean
    mean, weight, meanweight = 0, [], []
    for i in range(len(sigmas)):
        weight.append(((1 / (sigmas[i] ** 2)) / inversevar))
        meanweight.append(((1 / (sigmas[i] ** 2)) / inversevar) * values[i])
        mean += ((1 / (sigmas[i] ** 2)) / inversevar) * values[i]
    # Calculate weighted variance
    sigma = np.sqrt((1 / (inversevar)))
    # Calculate chi2
    chi2 = sum([((v - mean) ** 2 / d ** 2) for v, d in zip(values, sigmas)])
    # Calculate pval from ndof
    ndof = len(values) - 1
    Pval = stats.chi2.sf(chi2, ndof)
    # Print out answer if wanted
    if verbose:
        print(
            "chi2 = "
            + str(round(chi2, 5))
            + "      ndof = "
            + str(round(ndof, 5))
            + "        Pval = "
            + str(round(Pval, 5))
        )
        print(
            "mean = "
            + str(round(mean, 5))
            + "      sigma = "
            + str(round(sigma, 5))
            + "        sum 1/sigma^2 = "
            + str(round(inversevar, 5))
        )
    if savetable != "":
        if not exponential:
            latex = [
                "\\begin{center}\n",
                "\\begin{tabular}{| l | l | l | l | l |}\n",
                "\\hline\n",
                f"Mean & $\sigma$ & Chi2 & Ndof & Pval\\\\ \\hline\n",
                f"{round(mean,3)} & {round(sigma,3)} & {round(chi2,3)} & "
                + f"{round(ndof,3)} & {round(Pval,5)} \\\\ \\hline\n"
                "\\end{tabular}\n",
                "\\end{center}\n",
            ]
            f = open(savetable + ".txt", "w")
            [f.write(l) for l in latex]
        else:
            latex = [
                "\\begin{center}\n",
                "\\begin{tabular}{| l | l | l | l | l |}\n",
                "\\hline\n",
                f"Mean & $\sigma$ & Chi2 & Ndof & Pval\\\\ \\hline\n",
                f"{mean:4.2E} & {sigma:4.2E} & {chi2:2E} & "
                + f"{round(ndof,3)} & {Pval:4.2E} \\\\ \\hline\n"
                "\\end{tabular}\n",
                "\\end{center}\n",
            ]
            f = open(savetable + ".txt", "w")
            [f.write(l) for l in latex]
    return mean, sigma, Pval


"""Monte Carlo things"""


def IntHM(f, args, Nexp, Npoints, box, plot=True, savefig=""):
    """Integrates a function based on the hit & miss method
    ----------
    Parameters
    ----------
    f: function
        - Function to generate from, should be of form f(x,args), where args
          is a list of arguments
    args: list of length args in function call
        - List of parameters to pass to the function
    Nexp: integer
        - Number of integration experiments to average
    Npoints: integer
        - Number of points used in each experiment to get average
    box: list or ndarray of length 4
        - boundaries for hit & miss (xmin,xmax,ymin,ymax)
    firstplot: Boolean
        - Wether to plot first experiment scatterplot
    intplot: Boolean
        - Wether to plot histogram of the experiment results
    ---------
    Returns
    ---------
    I: float
        - Integral of the function in the interval
    Ierr: float
        - Standard deviation on the integral value
    """
    C = []
    xmin, xmax, ymin, ymax = box  # boundaries taken from the plot above
    if plot:
        fig, ax = plt.subplots(1, 2, figsize=(12, 8))
    for iexp in range(Nexp):
        hit = np.zeros(Npoints)
        xs, ys = np.zeros(Npoints), np.zeros(Npoints)

        # Generate Npoints and compute integral
        for i in range(Npoints):
            # Generate random numbers in the interval
            xs[i] = np.random.uniform(low=xmin, high=xmax)
            ys[i] = np.random.uniform(low=ymin, high=ymax)

            # Check if they are below or above the function
            if f(xs[i], *args) > ys[i]:
                hit[i] = 1
        # Append integral to C
        # fraction of function * area of box = C
        Integral = len(hit[hit == 1]) / float(Npoints) * (xmax - xmin) * (ymax - ymin)
        C.append(Integral)

        # If first run, plot points to see it works
        if iexp == 0 and plot:
            ax[0].scatter(xs[hit == 1], ys[hit == 1], color="tab:red", s=2)
            ax[0].plot(
                np.linspace(xmin, xmax),
                f(np.linspace(xmin, xmax), *args),
                label="f(x)",
                color="k",
            )
            ax[0].set_xlabel("x")
            ax[0].set_ylabel("f(x)")
            ax[0].set_xlim(xmin, xmax)
            ax[0].set_ylim(ymin, ymax)
            ax[0].set_title(f"C = {C[0]:6.4}")
            ax[0].grid()
    C = np.array(C)
    if plot:
        xmax, xmin = max(C), min(C)
        Nbins = max(10, len(C) / 20)
        ax[1].hist(C, range=(xmin, xmax), bins=Nbins)
        ax[1].set_ylabel(f"Frequency/{(xmax-xmin)/Nbins:4.2f}")
        ax[1].set_xlabel("Integral of f(x)")
        mean, std = np.mean(C), np.std(C)
        ax[1].axvline(
            x=np.mean(C),
            linestyle="--",
            color="k",
            label=f"Mean={mean:4.5f}+/-{std:4.5f}",
        )
        ax[1].grid()
        plt.tight_layout()
        ax[1].legend(prop={"size": 15})
        if savefig != "":
            plt.savefig(savefig + ".pdf", dpi=500)
        plt.show()
    return mean, std


def GenHM(f, args, N, box):
    """Generates N random numbers according to f based on hit and miss method
    ----------
    Parameters
    ----------
    f: function
        - Function to integrate, should be of form f(x,args), where args
          is a list of arguments
    args: list of length args in function call
        - List of parameters to pass to the function
    N: integer
        - Number of random points to generate
    box: list or ndarray of length 4
        - boundaries for hit & miss (xmin,xmax,ymin,ymax)
    ----------
    Returns
    ----------
    np.array of random numbers from f
    """
    Random_Points = np.zeros(N)
    xmin, xmax, ymin, ymax = box  # boundaries taken from the plot above
    for i in range(N):
        # Generate random number in the interval
        x = np.random.uniform(low=xmin, high=xmax)
        y = np.random.uniform(low=ymin, high=ymax)

        # Check if they are below or above the function
        while f(x, *args) < y:
            x = np.random.uniform(low=xmin, high=xmax)
            y = np.random.uniform(low=ymin, high=ymax)
        Random_Points[i] = x
    return Random_Points


def GenTrans(Npoints, f, C=1):
    """Generates Npoints random numbers according to f
    ---------
    Parameters
    ---------
    Npoints: integer
        - number of points to generate
    f: callable
        - cdf to generate from
    C: float
        - normalization of f to deal with nonnormalized functions
    ---------
    Return
    ---------
    Random_Points: ndarray of length Npoinst:
        - Random points generated from f
    """
    Random_Points = np.zeros(Npoints)
    for i in range(Npoints):
        # Generate random number in the interval
        Random_Points[i] = f(np.random.uniform(low=0, high=C))
    return Random_Points


def GenTransHM(Npoints, f, F, G, args=None, C=1):
    """Generates Npoints random numbers according to f
    Npoints: integer
        - number of points to generate
    f: callable
        - function bounding the interval (pdf to F)
    F: callable
        - invertion of f to generate from
    G: callable of type G(x,*args)
        - function to integrate the
    args: list of length *args
        - list of arguments to pass to G
    C: float
        - normalization of f to deal with nonnormalized functions
    """
    Random_Points = np.zeros(Npoints)
    if args is None:
        for i in range(Npoints):
            # Generate random number in the interval
            x = F(np.random.uniform(low=0, high=C))
            # use the generated x to find ymax
            y = np.inf
            while y > G(x):
                x = F(np.random.uniform(low=0, high=C))
                y = np.random.uniform(low=0, high=f(x))
            Random_Points[i] = x
    else:
        for i in range(Npoints):
            # Generate random number in the interval
            x = F(np.random.uniform(low=0, high=C))
            # use the generated x to find ymax
            y = np.inf
            while y > G(x, *args):
                x = F(np.random.uniform(low=0, high=C))
                y = np.random.uniform(low=0, high=f(x))
            Random_Points[i] = x
    return Random_Points


def IntTransHM(
    Nexp,
    Npoints,
    f,
    F,
    G,
    args=None,
    C=1,
    savefig="",
    firstplot=True,
    yscale="linear",
    intplot=True,
):
    """Integrates G(x) from 0 to infinity using numbers from F and bound from f
    Nexp: integer
        - number of experiments to do
    Npoints: integer
        - number of points to generate per experiment
    f: callable
        - function bounding the interval (pdf to F)
    F: callable
        - invertion of f to generate from
    G: callable of type G(x,*args)
        - function to integrate the
    args: list of length *args
        - list of arguments to pass to G
    C: float
        - normalization of f to deal with nonnormalized functions
    firstplot: Boolean
        - Wether to plot first experiment scatterplot
    intplot: Boolean
        - Wether to plot histogram of the experiment results
    ---------
    Returns
    ---------
    I: float
        - Integral of the function in the interval
    Ierr: float
        - Standard deviation on the integral value
    """
    integrals = []
    if args is None:
        for iexp in range(Nexp):
            xs = GenTrans(Npoints, f, C=C)
            Hit = np.zeros(Npoints)
            ys = np.zeros(Npoints)
            # use the generated x to integrate function
            for i in range(Npoints):
                ys[i] = np.random.uniform(low=0, high=f(xs[i]))
                if ys[i] < G(xs[i]):
                    Hit[i] = 1
            # Plot result if first
            if iexp == 0:
                Hitmask, notHitmask = Hit == 1, Hit == 0
                plt.scatter(
                    xs[Hitmask], ys[Hitmask], color="tab:blue", label="hit", s=0.5
                )
                plt.scatter(
                    xs[notHitmask],
                    ys[notHitmask],
                    color="tab:red",
                    label="outside",
                    s=0.1,
                )
                plt.xlabel("x")
                plt.ylabel("y")
                plt.yscale(yscale)
                plt.legend()
                plt.title(f"Integral = {(C*np.sum(Hit)/float(Npoints)):4.2f}")
                if savefig != "":
                    plt.savefig(savefig + "_first" + ".pdf", dpi=500)
                plt.show()

            # Append integral
            # 3 is the integral of the exponential
            integrals.append((np.sum(Hit) / float(Npoints)) * C)
    else:
        for iexp in range(Nexp):
            xs = GenTransHM(Npoints, f, F, G, args=args, C=C)
            Hit = np.zeros(Npoints)
            ys = np.zeros(Npoints)

            # use the generated x to integrate function
            for i in range(Npoints):
                ys[i] = np.random.uniform(low=0, high=f(xs[i]))
                if ys[i] < G(xs[i], *args):
                    Hit[i] = 1
            # Plot result if first
            if iexp == 0:
                Hitmask, notHitmask = Hit == 1, Hit == 0
                plt.scatter(
                    xs[Hitmask], ys[Hitmask], color="tab:blue", label="hit", s=0.5
                )
                plt.scatter(
                    xs[notHitmask],
                    ys[notHitmask],
                    color="tab:red",
                    label="outside",
                    s=0.1,
                )
                plt.xlabel("x")
                plt.ylabel("y")
                plt.legend()
                plt.title(f"Integral = {(np.sum(Hit)/float(Npoints))*3:4.2f}")
                if savefig != "":
                    plt.savefig(savefig + "_first" + ".pdf", dpi=500)
                plt.show()

            # Append integral
            # 3 is the integral of the exponential
            integrals.append((np.sum(Hit) / float(Npoints)) * C)

    xmax, xmin = max(integrals), min(integrals)
    Nbins = np.max([int(len(integrals) / 15), 5])

    plt.hist(integrals, bins=Nbins, range=(xmin, xmax))
    plt.xlabel("Integral")
    plt.ylabel(f"Frequency/{(xmax-xmin)/Nbins:4.2f}")
    plt.axvline(x=np.mean(integrals), linestyle="--", color="k")
    if savefig != "":
        plt.savefig(savefig + "_second" + ".pdf", dpi=500)
    plt.show()
    return np.mean(integrals), np.std(integrals)


def ErrPropMC(
    f,
    vals=None,
    errs=None,
    savefig="",
    distributions=None,
    numpoints=100,
    plot=True,
    valpos=None,
    labels=None,
    correlation=None,
):
    """Numerically errorpropagates f around vals with errors (gaussian) or
    with an arbitrary input distribution (distributions)
    --------
    Params
    --------
    f: callable
        - function to be errorpropagated of type f(*vals)
    vals: list of length *vals
        - values to errorpropagate through f (assumed gaussian)
    errs: list of length *vals
        - errors on the values (assumed gaussian)
    distributions: ndarray of shape (*vals,N)
        - distribution of input parameters to f, overvrites vals and
        errs if given
    numpoints: integer
        - number of points to use for Errprop if no distribution is given
    plot: Boolean
        - Wether to plot the distribution of f
    """
    from multiprocessing import Pool
    from itertools import product

    # Generate gaussian inputs if none specified
    if distributions is None:
        N_bins = int((numpoints ** 1.1))
        if correlation is None:
            cov = [[errs[0] ** 2, 0], [0, errs[1] ** 2]]
            distributions = np.random.multivariate_normal(
                [vals[0], vals[1]], cov, size=numpoints
            )
        else:
            cov = [
                [errs[0] ** 2, correlation * errs[0] * errs[1]],
                [correlation * errs[0] * errs[1], errs[1] ** 2],
            ]
            distributions = np.random.multivariate_normal(
                [vals[0], vals[1]], cov, size=numpoints
            )
    # Sample all combinations using threading
    out = []
    for d in distributions:
        out.append(f(d[0], d[1]))
    if plot:
        fig, ax = plt.subplots(2, 1, figsize=(12, 10))
        counts, xedges, yedges, im = ax[0].hist2d(
            distributions[:, 0], distributions[:, 1], bins=[50, 50], cmin=1
        )
        ax[0].set(xlabel=r"$v_1$", ylabel=r"$v_2$", aspect="auto")
        fig.colorbar(im, ax=ax[0])
        bins = np.max([int(len(out) / 15), 5])
        xmin, xmax = min(out), max(out)
        BinWidth = (xmax - xmin) / bins

        def fitfunc(x, mu, N, sigma):
            return gauss_extended(x, N, mu, sigma) * BinWidth

        if valpos is None:
            x, y, sy = histogram(
                out, remove0=True, plot=False, range=(xmin, xmax), bins=bins
            )

            Chi2Fit(
                x,
                y,
                sy,
                fitfunc,
                N=len(out),
                mu=np.mean(out),
                sigma=np.std(out),
                ax=ax[1],
                valpos=[0.65, 0.4],
                exponential=True,
                fitcol="g",
            )
        else:
            x, y, sy = histogram(
                out, remove0=True, plot=False, range=(xmin, xmax), bins=bins
            )
            Chi2Fit(
                x,
                y,
                sy,
                fitfunc,
                N=len(out),
                mu=np.mean(out),
                sigma=np.std(out),
                ax=ax[1],
                valpos=[valpos[0], valpos[1] - 0.2],
                exponential=True,
                fitcol="g",
            )
        ax[1].set(xlabel=r"x", ylabel=f"Frequency/{BinWidth:4.2E}")
        ax[1].set_title(r"Error propagated function")

        ax[1].axvline(x=np.mean(out), linestyle="--", color="k", label="mean")

        x, y, sy = histogram(
            out, bins=int(np.sqrt(numpoints) * 10), remove0=True, plot=False
        )
        ax[1].axvline(x=x[np.argmax(y)], linestyle="--", color="b", label="Mode")
        std = np.sqrt(np.sum((out - np.mean(out)) ** 2) / (len(out) - 1))
        errstd = std / (np.sqrt(2 * len(out) - 1))
        dict = {
            "Mean": f"{np.mean(out):4.2E}",
            f"Mode": f"{x[np.argmax(y)]:4.2E}",
            r"Std": f"{np.std(out):4.2E}+/-{errstd:4.2E}",
        }

        text = nice_string_output(dict)
        if valpos is None:
            add_text_to_ax(0.65, 0.6, text, ax[1], fontsize=15)
        else:
            add_text_to_ax(valpos[0], valpos[1], text, ax[1], fontsize=15)
        ax[1].legend(prop={"size": 13})
        if labels is not None:
            ax[1].set(**labels)
        ax[1].grid()
        plt.tight_layout()
        if savefig != "":
            plt.savefig(savefig + ".pdf", dpi=500)
        plt.show()
    return np.mean(out), np.std(out)


def Binary_search(
    f, args, minstart, maxstart, accuracy=1e-8, verbose=False, maxcount=10000, counts=0
):
    """Solves numerically the equation f(x,args) = 0
    ----------
    Params
    ----------
    f: callable of type f(x,*args)
        - function to solve
    args: list
        - arguments to pass to f
    minstart: float
        - minimum bound for search window
    maxstart: float
        - maximum bound for search window
    accuracy: float, optional
        - numerical accuracy of algorithm
    verbose: boolean
        - print progress
    maxcount: int
        - maximum number of runs before finding solution
    counts: int
        - internal variable for keeping track of updates, not to be changed!!
    ----------
    return
    ----------
    middle: float
        - The input x for which f is within the accuracy of 0
    """
    counts = 0
    # Calculate Gdiff for min and max and find middle
    if counts > maxcount:
        if verbose:
            print("Didn't Converge")
        return np.nan
    minres, maxres = f(minstart, *args), f(maxstart, *args)
    if np.abs(maxstart - minstart) < accuracy:  # minstart is close enough, return
        middle = (maxstart + minstart) * 0.5
        out = f(middle, *args)
        if verbose:
            print(f"Converged at theta={middle:4.2E} and equality={out:4.2E}")
        return middle
    else:  # Otherwise not close enough, keep goings
        smallarg = np.argmin(
            np.array([np.abs(minres), np.abs(maxres)])
        )  # abs so that minimum is closest to 0
        middle = (maxstart + minstart) * 0.5
        minout = min([[minstart, maxstart][smallarg], middle])
        maxout = max([[minstart, maxstart][smallarg], middle])
        closest = [minres, maxres][smallarg]
        if verbose:
            print(
                f"Continuing, itteration: {counts}, min:{minout:4.2E}, max:{maxout:4.2E}, besttheta:{closest:4.2E}"
            )
        return Binary_search(
            f,
            args,
            minout,
            maxout,
            accuracy=accuracy,
            counts=counts + 1,
            verbose=verbose,
            maxcount=maxcount,
        )


"""Fitting and binning"""


def histogram(
    data,
    range=(None, None),
    bins=None,
    remove0=False,
    plot=True,
    savefig="",
    labels=None,
    bars=False,
    ax=None,
    color="r",
    alpha=1,
    remove_num=None,
    calibration=None,
    legend=None,
    logbin=False,
    normalize=False,
    elinewidth=5,
    capsize=4,
):
    """Get data for histogram
    --------
    Params
    --------
    data: 1D ndarray
        - Raw Data
    xmin: float
        - Minimum value for data
    xmax: float
        - Maximum value for data
    nbins: integer
        - Number of bins in histogram
    plot: Boolean
        - Wether to plot histogram
    savefig: string
        - Wether to save the plotted histogram
    labels: dictionary
        - Values to pass to ax.set()
    legend: optional, str
        - What to call the plot in a legend
    logbin: optional, Boolean
        - wether to bin the histogram logarithmically
    """
    xmin, xmax = range
    if xmin is None:
        xmin = np.min(data)
    if xmax is None:
        xmax = np.max(data)
    if bins is None:
        bins = np.max([int(len(data) / 15), 5])
    if logbin:
        N = bins
        inputs = np.logspace(np.log10(xmin), np.log10(xmax), N)
        if normalize:
            hist = np.histogram(data, bins=inputs, density=True)
            hist2 = np.histogram(data, bins=inputs)
        else:
            hist = np.histogram(data, bins=inputs)
    else:
        if normalize:
            hist = np.histogram(data, bins=bins, range=(xmin, xmax), density=True)
            hist2 = np.histogram(data, bins=bins, range=(xmin, xmax))
        else:
            hist = np.histogram(data, bins=bins, range=(xmin, xmax))
    counts, bin_edges = hist
    bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])

    if calibration is None:
        mask1 = (xmin < bin_centers) & (bin_centers <= xmax)
        if remove0:
            mask2 = counts > 0
            if remove_num is not None:
                mask3 = counts > remove_num
                mask_final = mask1 & mask2 & mask3
            else:
                mask_final = mask1 & mask2
        else:
            mask_final = mask1
        x, y, sy = (
            bin_centers[mask_final],
            counts[mask_final],
            np.sqrt(counts[mask_final]),
        )
        if normalize:
            counts1, bin_edges1 = hist2
            bin_centers1 = 0.5 * (bin_edges1[1:] + bin_edges1[:-1])
            mask1 = (xmin < bin_centers1) & (bin_centers1 <= xmax)
            if remove0:
                mask2 = counts1 > 0
                if remove_num is not None:
                    mask3 = counts > remove_num
                    mask_final = mask1 & mask2 & mask3
                else:
                    mask_final = mask1 & mask2
            else:
                mask_final = mask1
            x1, y1, sy1 = (
                bin_centers1[mask_final],
                counts1[mask_final],
                np.sqrt(counts1[mask_final]),
            )
            relerr = y1 / sy1
            sy = y / relerr
    else:
        counts = counts - calibration(bin_centers)
        mask1 = (xmin < bin_centers) & (bin_centers <= xmax)
        if remove0:
            mask2 = counts > 0
            if remove_num is not None:
                mask3 = counts > remove_num
                mask_final = mask1 & mask2 & mask3
            else:
                mask_final = mask1 & mask2
        else:
            mask_final = mask1
        x, y, sy = (
            bin_centers[mask_final],
            counts[mask_final],
            np.sqrt(counts[mask_final]),
        )
    if plot:
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        if logbin is None:
            Binwidth = (xmax - xmin) / bins
        else:
            Binwidth = (bin_edges[1:] - bin_edges[:-1])[mask_final]
        if not bars:
            if legend is None:
                ax.errorbar(
                    x,
                    y,
                    yerr=sy,
                    xerr=Binwidth / 2,
                    linestyle="",
                    ecolor=color,
                    fmt=".",
                    mfc=color,
                    mec=color,
                    capsize=capsize,
                    elinewidth=elinewidth,
                )
            else:
                ax.errorbar(
                    x,
                    y,
                    yerr=sy,
                    xerr=Binwidth / 2,
                    linestyle="",
                    ecolor=color,
                    fmt=".",
                    mfc=color,
                    mec=color,
                    capsize=capsize,
                    label=legend,
                    elinewidth=elinewidth,
                )
        else:
            if legend is None:
                ax.bar(
                    x,
                    y,
                    width=Binwidth,
                    color=color,
                    yerr=sy,
                    capsize=capsize,
                    alpha=alpha,
                    error_kw={"elinewidth": elinewidth},
                )
            else:
                ax.bar(
                    x,
                    y,
                    width=Binwidth,
                    color=color,
                    yerr=sy,
                    capsize=capsize,
                    alpha=alpha,
                    label=legend,
                    error_kw={"elinewidth": elinewidth},
                )
        if np.abs(np.mean(Binwidth)) < 100:
            ax.set(xlabel="x", ylabel=f"Frequency / {np.abs(np.mean(Binwidth)):4.2e}")
        else:
            ax.set(xlabel="x", ylabel=f"Frequency / {np.abs(np.mean(Binwidth)):4.2e}")
        if labels is not None:
            ax.set(**labels)
        if logbin:
            ax.set_xscale("log")
        if legend is not None:
            ax.legend()
        if savefig != "":
            plt.savefig(savefig + ".pdf")
        if ax is None:
            plt.show()
    return x, y, sy


def Chi2Fit(
    x,
    y,
    sy,
    f,
    plot=True,
    print_level=0,
    labels=None,
    ax=None,
    savefig="",
    valpos=None,
    exponential=False,
    fitcol=None,
    markersize=5,
    plotcol=None,
    name=None,
    fontsize=15,
    linewidth=3,
    png=False,
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
    chi2_object = LeastSquares(x, y, sy, f)
    if len(guesses) != 0:
        minuit = Minuit(chi2_object, **guesses)
    else:
        minuit = Minuit(chi2_object)
    minuit.print_level = print_level
    minuit.migrad()
    chi2 = minuit.fval
    # free parameters, not len(guesses): `guesses` also carries the
    # limit_<name> entries, which are bounds rather than fitted parameters
    Ndof = len(x) - minuit.nfit
    Pval = stats.chi2.sf(chi2, Ndof)
    params = list(minuit.values)
    errs = list(minuit.errors)

    if not exponential:
        dict = {"chi2": chi2, "Ndof": Ndof, "Pval": Pval}
        for n, p, py in zip(names, params, errs):
            dict[n] = f"{p:4.2f} +/- {py:4.2f}"
    else:
        dict = {"chi2": f"{chi2:4.4E}", "Ndof": f"{Ndof:4.4E}", "Pval": f"{Pval:4.4E}"}
        for n, p, py in zip(names, params, errs):
            dict[n] = f"{p:4.4E} +/- {py:4.4E}"
    if plot:
        # Plot the fit
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        text = nice_string_output(dict)
        x_fit = np.linspace(xmin, xmax, 200)
        y_fit = [f(i, *params) for i in x_fit]
        if labels is None:
            ax.set(xlabel="x", ylabel="f(x)")
        else:
            ax.set(**labels)
        if fitcol is None:
            if name is None:
                ax.plot(x_fit, y_fit, color="r", label="Fit", linewidth=linewidth)
            else:
                ax.plot(x_fit, y_fit, color="r", label=name, linewidth=linewidth)
        else:
            if name is None:
                ax.plot(x_fit, y_fit, color=fitcol, label="Fit", linewidth=linewidth)
            else:
                ax.plot(x_fit, y_fit, color=fitcol, label=name, linewidth=linewidth)
        if valpos is None:
            if fitcol is None:
                add_text_to_ax(0.05, 0.9, text, ax, fontsize=fontsize)
            else:
                add_text_to_ax(0.05, 0.9, text, ax, color=fitcol, fontsize=fontsize)
        else:
            if fitcol is None:
                add_text_to_ax(
                    valpos[0], valpos[1], text, ax, color="k", fontsize=fontsize
                )
            else:
                add_text_to_ax(
                    valpos[0], valpos[1], text, ax, color=fitcol, fontsize=fontsize
                )
        if name is None:
            if plotcol is None:
                ax.errorbar(
                    x,
                    y,
                    yerr=sy,
                    linestyle="",
                    ecolor="k",
                    fmt=".r",
                    label="Data",
                    capsize=2,
                    markersize=markersize,
                )
            else:
                ax.errorbar(
                    x,
                    y,
                    yerr=sy,
                    linestyle="",
                    ecolor="k",
                    marker=".",
                    color=plotcol,
                    label="Data",
                    capsize=2,
                    markersize=markersize,
                )
        else:
            if plotcol is None:
                ax.errorbar(
                    x,
                    y,
                    yerr=sy,
                    linestyle="",
                    ecolor="k",
                    fmt=".r",
                    capsize=2,
                    markersize=markersize,
                )
            else:
                ax.errorbar(
                    x,
                    y,
                    yerr=sy,
                    linestyle="",
                    ecolor="k",
                    marker=".",
                    color=plotcol,
                    capsize=2,
                    markersize=markersize,
                )
        if savefig != "":
            if not png:
                plt.savefig(savefig + ".pdf", dpi=500)
            else:
                plt.savefig(savefig, dpi=500)
        if ax is None:
            plt.legend()
            plt.show()
    return params, errs, Pval
    # return params, errs, Pval


def BChi2Fit(
    data,
    bins,
    f,
    bound=None,
    ax=None,
    plot=True,
    print_level=0,
    labels=None,
    savefig="",
    exponential=False,
    valpos=None,
    fitcol=None,
    **guesses,
):
    """Function that peforms a binned Chi2Fit to data given function
    ----------
    Parameters
    ----------
    data: ndarray
        - raw data from random variable
    bins: int
        - number of bins in fig
    f: function
        - Function to fit, should be of form f(x,args), where args
          is a list of arguments
    bound: list of length 2
        -[xmin,xmax] for the fit region
    **guesses: mappings ie. p0=0.1,p1=0.2
        - initial guesses for the fit parameters
    print_level: int 0,1
        - Wether to print output from chi2 ect.
    labels:
        - Mappable to pass to ax.set call to set labels on plot
    ---------
    Returns
    ---------
    params: length args
        - fit params
    errs: lenght args
        - errror on fit params
    LLH: float
        -Log Likelihood for the fit
    """
    names = inspect.getfullargspec(f)[0][1:]
    chi2_object = BinnedChi2(f, data, bound=bound)
    if len(guesses) != 0:
        minuit = Minuit(chi2_object, **guesses)
    else:
        minuit = Minuit(chi2_object)
    minuit.print_level = print_level
    minuit.migrad()
    chi2 = minuit.fval
    Ndof = bins - len(guesses)
    Pval = stats.chi2.sf(chi2, Ndof)
    params = list(minuit.values)
    errs = list(minuit.errors)
    if not exponential:
        dict = {"chi2": chi2, "Ndof": Ndof, "Pval": Pval}
        for n, p, py in zip(names, params, errs):
            dict[n] = f"{p:4.2f} +/- {py:4.2f}"
    else:
        dict = {"chi2": f"{chi2:4.2E}", "Ndof": f"{Ndof:4.2E}", "Pval": f"{Pval:4.2E}"}
        for n, p, py in zip(names, params, errs):
            dict[n] = f"{p:4.2E} +/- {py:4.2E}"
    if plot:
        # Plot the fit
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(12, 6))
        text = nice_string_output(dict)
        if valpos is None:
            if fitcol is None:
                add_text_to_ax(0.05, 0.9, text, ax, fontsize=15)
            else:
                add_text_to_ax(0.05, 0.9, text, ax, color=fitcol, fontsize=15)
        else:
            if fitcol is None:
                add_text_to_ax(valpos[0], valpos[1], text, ax, fontsize=15)
            else:
                add_text_to_ax(
                    valpos[0], valpos[1], text, ax, fontsize=15, color=fitcol
                )
        if bound is None:
            xmin, xmax = np.min(data), np.max(data)
        else:
            xmin, xmax = bound[0], bound[1]
        x, y, sy = histogram(data, range=(xmin, xmax), bins=bins, plot=False)
        BinWidth = (xmax - xmin) / bins

        ax.errorbar(
            x,
            y,
            yerr=sy,
            xerr=BinWidth / 2,
            linestyle="",
            fmt=".r",
            ecolor="k",
            label="Data",
            capsize=2,
        )
        x_fit = np.linspace(xmin, xmax)
        y_fit = [f(i, *params) * (xmax - xmin) / bins for i in x_fit]
        if labels is None:
            if not exponential:
                ax.set(
                    xlabel="x",
                    ylabel=f"u(x) / {BinWidth:4.2f}",
                    ylim=(np.min(y) - np.min(y) * 0.1, np.max(y) * 2),
                )
            else:
                ax.set(
                    xlabel="x",
                    ylabel=f"u(x) / {BinWidth:4.2E}",
                    ylim=(np.min(y) - np.min(y) * 0.1, np.max(y) * 2),
                )
        else:
            ax.set(**labels)
        if fitcol is None:
            ax.plot(x_fit, y_fit, color="r", label="Fit")
        else:
            ax.plot(x_fit, y_fit, color=fitcol, label="Fit")
        ax.grid()
        ax.legend()
        if savefig != "":
            fig.savefig(savefig + ".pdf", dpi=500)
        if ax is None:
            plt.show()
    return params, errs, Pval


def BLLHFit(
    data,
    bins,
    f,
    bound=None,
    plot=True,
    print_level=0,
    extended=False,
    ax=None,
    labels=None,
    savefig="",
    valpos=None,
    fitcol=None,
    exponential=False,
    **guesses,
):
    """Function that peforms a Chi2Fit to data given function
    ----------
    Parameters
    ----------
    data: ndarray
        - raw data from random variable
    bins: int
        - number of bins in fig
    f: function
        - Function to fit, should be of form f(x,args), where args
          is a list of arguments
    bound: list of length 2
        -[xmin,xmax] for the fit region
    **guesses: mappings ie. p0=0.1,p1=0.2
        - initial guesses for the fit parameters
    print_level: int 0,1
        - Wether to print output from chi2 ect.
    extended: Boolean
        - Wether the fit should be extended or not (N is a free parameter)
    labels:
        - Mappable to pass to ax.set call to set labels on plot
    ---------
    Returns
    ---------
    params: length args
        - fit params
    errs: lenght args
        - errror on fit params
    LLH: float
        -Log Likelihood for the fit
    """
    names = inspect.getfullargspec(f)[0][1:]
    if extended:
        LLH_object = BinnedLH(f, data, bins=bins, bound=bound, extended=True)
    else:
        LLH_object = BinnedLH(f, data, bins=bins, bound=bound)
    if len(guesses) != 0:
        minuit = Minuit(LLH_object, **guesses)
    else:
        minuit = Minuit(LLH_object)
    minuit.print_level = print_level
    minuit.migrad()
    LLH = -minuit.fval
    params = list(minuit.values)
    errs = list(minuit.errors)
    if not exponential:
        dict = {"LLH": LLH}
        for n, p, py in zip(names, params, errs):
            dict[n] = f"{p:4.2f} +/- {py:4.2f}"
    else:
        dict = {"LLH": f"{LLH:4.2E}"}
        for n, p, py in zip(names, params, errs):
            dict[n] = f"{p:4.2E} +/- {py:4.2E}"
    if plot:
        # Plot the fit
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(12, 6))
        text = nice_string_output(dict)
        if valpos is None:
            if fitcol is None:
                add_text_to_ax(0.55, 0.9, text, ax, fontsize=15)
            else:
                add_text_to_ax(0.55, 0.9, text, ax, color=fitcol, fontsize=15)
        else:
            add_text_to_ax(valpos[0], valpos[1], text, ax, color=fitcol, fontsize=15)
        if bound is None:
            xmin, xmax = np.min(data), np.max(data)
        else:
            xmin, xmax = bound[0], bound[1]
        x, y, sy = histogram(data, range=(xmin, xmax), bins=bins, plot=False)
        BinWidth = (xmax - xmin) / bins
        ax.errorbar(
            x,
            y,
            yerr=sy,
            xerr=BinWidth / 2,
            linestyle="",
            fmt=".r",
            ecolor="k",
            label="Data",
            capsize=2,
        )
        x_fit = np.linspace(xmin, xmax)
        y_fit = [f(i, *params) * (xmax - xmin) / bins for i in x_fit]
        if labels is None:
            ax.set(
                xlabel="x",
                ylabel=f"f(x) / {BinWidth:4.2f}",
                ylim=(np.min(y) - np.min(y) * 0.1, np.max(y) * 2),
            )
        else:
            ax.set(**labels)
        if fitcol is None:
            ax.plot(x_fit, y_fit, color="r", label="Fit")
        else:
            ax.plot(x_fit, y_fit, color=fitcol, label="Fit")
        ax.grid()
        plt.legend()
        if savefig != "":
            plt.savefig(savefig + ".pdf", dpi=500)
        if ax is None:
            plt.show()
    return params, errs, LLH


def LLHFit(
    x,
    f,
    plot=True,
    print_level=0,
    bound=None,
    labels=None,
    savefig="",
    extended=False,
    fitcol=None,
    valpos=None,
    plotvals=[],
    ax=None,
    exponential=False,
    lines=True,
    **guesses,
):
    """Function that peforms an unbinned likelihood to data given function
    ----------
    Parameters
    ----------
    x: ndarray of shape for input in f
        - input values to fit
    f: function
        - Function to fit, should be of form f(x,args), where args
          is a list of arguments
    **guesses: mappings ie. p0=0.1,p1=0.2
        - initial guesses for the fit parameters
    print_level: int 0,1
        - Wether to print output from chi2 ect.
    labels:
        - Mappable to pass to ax.set call to set labels on plot
    extended:
        - Wether the number of observations is a fit variable
    ---------
    Returns
    ---------
    params: length args
        - fit params
    errs: lenght args
        - errror on fit params
    pval: float
        -pvalue for the fit
    """
    names = inspect.getfullargspec(f)[0][1:]
    LLH_object = UnbinnedLH(f, x, extended=extended)
    if len(guesses) != 0:
        minuit = Minuit(LLH_object, **guesses)
    else:
        minuit = Minuit(LLH_object)
    minuit.print_level = print_level
    minuit.migrad()
    LLH = -minuit.fval
    params = list(minuit.values)
    errs = list(minuit.errors)

    if not exponential:
        dict = {"LLH": LLH}
        for n, p, py in zip(names, params, errs):
            dict[n] = f"{p:4.4f} +/- {py:4.4f}"
    else:
        dict = {"LLH": f"{LLH:4.2E}"}
        for n, p, py in zip(names, params, errs):
            dict[n] = f"{p:4.2E} +/- {py:4.2E}"
    if plot:
        # Plot the fit
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(12, 6))
        text = nice_string_output(dict)
        if valpos is None:
            if fitcol is None:
                add_text_to_ax(0.55, 0.9, text, ax, fontsize=15)
            else:
                add_text_to_ax(0.55, 0.9, text, ax, color=fitcol, fontsize=15)
        else:
            if fitcol is None:
                add_text_to_ax(valpos[0], valpos[1], text, ax, fontsize=15)
            else:
                add_text_to_ax(
                    valpos[0], valpos[1], text, ax, fontsize=15, color=fitcol
                )
        if bound is None:
            xmin, xmax = np.min(x), np.max(x)
        else:
            xmin, xmax = bound[0], bound[1]
        x_fit = np.linspace(xmin, xmax)
        y_fit = [f(i, *params) * (xmax - xmin) / len(x_fit) for i in x_fit]
        diff = np.max(y_fit) * np.ones(len(x)) - np.min(y_fit) * np.ones(len(x))
        if fitcol is None:
            ax.plot(x_fit, y_fit, color="r", label="Fit", linewidth=3)
        else:
            ax.plot(x_fit, y_fit, color=fitcol, label="Fit")
        if lines:
            ax.vlines(
                x,
                np.min(y_fit) * np.ones(len(x)),
                diff / 8,
                colors="k",
                lw=2,
                label="data",
            )
        if len(plotvals) == 2:
            ax.errorbar(
                x,
                plotvals[0],
                yerr=plotvals[1],
                linestyle="",
                ecolor="k",
                fmt=".r",
                label="Data",
                capsize=2,
            )
        if labels is None:
            ax.set(xlabel="x", ylabel="f(x)", ylim=(0, np.max(y_fit) * 1.4))
        else:
            ax.set(**labels, ylim=(np.min(y_fit), np.max(y_fit) * 1.4))
        ax.set(xlim=(xmin, xmax))

        ax.grid()
        plt.legend()
        if savefig != "":
            plt.savefig(savefig + ".pdf", dpi=500)
        if ax is None:
            plt.show()
    return params, errs, LLH


"""Statistical tests"""
# General tests
def TwoSided(mean1, mean2, std1, std2):
    """Computes the TwoSided test statistic and returns z-val and pval
    smaller pvalue the better"""
    zval = (mean1 - mean2) / np.sqrt(std1 ** 2 + std2 ** 2)
    pval = 2 * stats.norm(0, 1).sf(zval)
    return zval, pval


def OneSided(mean1, mean2, std2):
    """Computes the TwoSided test statistic and returns z-val and pval
    higher pvalue the better"""
    zval = (mean1 - mean2) / std2
    pval = 2 * stats.norm(0, 1).sf(-zval)
    return zval, pval


def Fisher(matrix):
    """Peforms Fishers exact test on a matrix
    ----------
    Params
    ----------
    matrix: ndarray of shape (2,2)
        - Contingency table to test for randomness
    ----------
    Returns
    ----------
    pval: float
        - Pvalue for the fisher test
    Example oceans vs. whales and sharks pval:3 .5%
    The probability that we would observe this or an even more
    imbalanced ratio by chance is about 3.5%.
    A commonly used significance level is 5%,
    if we adopt that we can therefore conclude
    that our observed imbalance is statistically
    significant; whales prefer the Atlantic while
    sharks prefer the Indian ocean.
    """
    return stats.fisher_exact(matrix)[-1]


def Chi2(observed, expected, error, fitparams=0):
    """Computes a chi2 between expected and observed and returns pval as well
    ---------
    Params
    ---------
    observed: 1D ndarray or list
        - observed values
    expected: 1D ndarray or list of len(observed)
        - expected values
    error: 1D ndarray or list of len(observed)
        - errors on the expected values
    ---------
    Returns
    ---------
    chi2: float
        - calculated chi2
    ndof: integer
        - number of degrees of freedom for fit (no fit params are assumed)
    pval: float
        -pvalue for the fit the smaller the better ;)
    """
    observed, expectd, error = np.array(observed), np.array(expected), np.array(error)
    chi2 = np.sum((observed - expected) ** 2 / error ** 2)
    ndof = len(observed) - fitparams
    pval = stats.chi2.sf(chi2, ndof)
    return chi2, ndof, pval


# Two similarity tests
def KS(samp1, samp2):
    """Calculates Kolmogorov Smirnoff test on two samples
    --------
    Params
    --------
    samp1: ndarray or list
        - random variables drawn from a distribution
    samp2: ndarray or list
        - random variables drawn from a distribution
    --------
    Returns
    --------
    val: KS test statistic
    pval: pvalue for that statistic

    If the K-S statistic is small or the p-value is high,
    then we cannot reject the hypothesis that the distributions of
    the two samples are the same.
    """
    return stats.ks_2samp(samp1, samp2)


def Mann(samp1, samp2):
    """Mann-Whitney rank test for samp1 and samp2 being the same
    --------
    Params
    --------
    samp1: ndarray or list
        - random variables drawn from a distribution
    samp2: ndarray or list
        - random variables drawn from a distribution
    --------
    Returns
    --------
    statistic: float, Mann-Whitney U test statistic
    pval: float, pvalue, the lower the more alike less than 1 usually the same
    --------
    Use only when the number of observation in each sample is > 20
    and you have 2 independent samples of ranks.
    Mann-Whitney U is significant if the u-obtained is LESS THAN or
    equal to the critical value of U.
    This test corrects for ties and by default uses a continuity correction.
    """
    return stats.mannwhitneyu(samp1, samp2)


# Sample against distribution tests
def Anderson(data, dist="norm"):
    """Anderson Darling test for data belonging to distribution
    ----------
    Params:
    ----------
    data: ndarray or list
        - raw data from random variable
    dist : string
        - {‘norm’,’expon’,’logistic’,’gumbel’,’gumbel_l’, gumbel_r’,
          ‘extreme1’}, optional the type of distribution to test against.
          The default is ‘norm’ and ‘extreme1’, ‘gumbel_l’ and ‘gumbel’
          are synonyms.
    ----------
    returns:
    ----------
    statistic: float
    critical_values: list
        - critical values for the distribution
    significance_level: list
        - The significance levels for the corresponding critical
         values in percents. The function returns critical values for
         a differing set of significance levels depending on
         the distribution that is being tested against.
    ----------
    from scipy.stats.anderson regarding significance_level:
    Critical values provided are for the following significance levels:
    normal/exponenential
    15%, 10%, 5%, 2.5%, 1%
    logistic
    25%, 10%, 5%, 2.5%, 1%, 0.5%
    Gumbel
    25%, 10%, 5%, 2.5%, 1%
    If the returned statistic is larger than these critical values then for
    the corresponding significance level, the null hypothesis that the
    data come from the chosen distribution can be rejected.
    The returned statistic is referred to as ‘A2’ in the references.
    """
    return stats.anderson(data, dist=dist)


def kstest(data, cdf, *args, N=20):
    """Two-sided KS test to see if data comes from given distribution
    ----------
    Params
    ----------
    data: 1D array
        - raw data observed from random variable
    cdf:
        - {‘norm’,’expon’,’logistic’,’gumbel’,’gumbel_l’, gumbel_r’,
          ‘extreme1’} distribution to test against
    ----------
    Returns
    ----------
    statistic: float
        - KS test statistic
    pvalue: float
        - two sided p-value for the statistic the lower the more different
    """
    # Generate 1000000 normal numbers:
    return stats.kstest(data, cdf, args=args, N=N)


# Normality tests
def Shapiro(data):
    """Shapiro-Wilk test for normality
    The Shapiro-Wilk test tests the null hypothesis that the data was
    drawn from a normal distribution.
    ----------
    returns:
    W: float
        -test statistics
    pval: float
        -pvalue, the smaller the less normal
    ----------
    Note from scipy.stats:
    The algorithm used is described in [4] but censoring parameters as described are not implemented. For N > 5000 the W test statistic is accurate but the p-value may not be.

    The chance of rejecting the null hypothesis when it is true is close to 5% regardless of sample size.
    """
    return stats.shapiro(data)


def Jarque(data):
    """Jarque-Bera test for normality
    ----------
    Params
    ----------
    data: 1d array of random variable observations
    ----------
    Return
    ----------
    jb_values: float, test statistic
    p: float, pvalue for hypothesis the smaller the less normal
    ---------
    The Jarque-Bera test tests whether the sample data has the skewness and
    kurtosis matching a normal distribution.
    Note that this test only works for a large enough number of data samples
    (>2000) as the test statistic asymptotically has a Chi-squared distribution
    with 2 degrees of freedom.
    """
    return stats.jarque_bera(data)


# Test for normality using all test!!
def NormalityFunc(f, n_samples, n_exp, savefig=""):
    """Test normality of a sample generator
    f: callable
        -
    """
    chi, KSs, Shap, Jar = [], [], [], []
    for i in range(n_exp):
        if i % 10 == 0:
            print(i)
        exp = [f() for i in range(n_samples)]
        KSs.append(KS(exp, np.random.normal(np.mean(exp), np.std(exp), len(exp)))[-1])
        Shap.append(Shapiro(exp)[-1])
        Jar.append(Jarque(exp)[-1])
        x, y, sy = histogram(exp, remove0=True, bins=40, range=(3, 8.5), plot=False)
        params, errs, pval = Chi2Fit(
            x,
            y,
            sy,
            gauss_extended,
            N=1000,
            mu=5 * 1.17,
            sigma=1.64 - 1.17 ** 2,
            plot=False,
        )
        chi.append(pval)
    fig, ax = plt.subplots(2, 2, figsize=(12, 8))
    histogram(
        KSs,
        ax=ax[0, 0],
        bins=10,
        plot=True,
        labels={"xlabel": "pval", "title": f"Kulmogorov Smirnof test"},
    )
    histogram(
        Shap,
        ax=ax[0, 1],
        bins=10,
        plot=True,
        labels={"xlabel": "pval", "title": f"Shapiro-Wilk test"},
    )
    histogram(
        Jar,
        ax=ax[1, 0],
        bins=10,
        plot=True,
        labels={"xlabel": "pval", "title": f"Jarque-Bera test"},
    )
    histogram(
        chi,
        ax=ax[1, 1],
        bins=10,
        plot=True,
        labels={"xlabel": "pval", "title": f"chi2 test from fit"},
    )
    fig.tight_layout()

    if savefig != "":
        fig.savefig(savefig + ".pdf", dpi=500)
    plt.show()


def NormalityDat(data, latex=True):
    """Test normality of a sample
    ---------
    params
    ---------
    data: 1D array
        - sample to test for normality without fit
    latex: bool
        -wether to gen latex table
    --------
    Returns
    --------
    - list of pvalues for statistics KS,Shap,Jarq
    - if latex, a string latex table
    """
    KS = kstest(data, "norm", np.mean(data), np.std(data), N=len(data))[-1]
    Shap = Shapiro(data)[-1]
    Jarq = Jarque(data)[-1]
    latex = (
        "\\begin{center}\n"
        + "\\begin{tabular}{| l | l | l |}\n"
        + "\\hline\n"
        + f"KS & Shapiro & Jarque\\\\ \\hline\n"
        + f"{round(KS,3)} & {round(Shap,3)} & {round(Jarq,3)}"
        + f"\\\\ \\hline\n"
        + "\\end{tabular}\n"
        + "\\end{center}\n"
    )
    return [KS, Shap, Jarq], latex


"""Hypothesis testing"""


def calc_ROC_data(data1, data2, plot=True, savefig=""):
    """Computes ROC data from two 1D histogram x,y data arrays
    ----------
    Params
    ----------
    data1: ndarray of shape (N,2)
        - x,y from signal
    data2: ndarray of shape (N,2)
        - x,y from background
    plot: Boolean
        - wether to plot the ROC curve
    --------
    Returns
    --------
    FPR: ndarray
        - Computed false positive rate
    TPR: ndarray
        - Computed True positive rate
    """
    # hist1 is signal, hist2 is background
    # first we extract the entries (y values) and the edges of the histograms
    x_sig_centers, y_sig = data1
    x_bkg_centers, y_bkg = data2
    y_sig = y_sig.astype(float)
    y_bkg = y_bkg.astype(float)
    # Check that the two histograms have the same x edges:
    if np.array_equal(x_sig_centers, x_bkg_centers):
        x_centers = x_sig_centers
        # calculate the integral (sum) of the signal and background
        integral_sig = y_sig.sum()
        integral_bkg = y_bkg.sum()
        # initialize empty arrays for the True Positive Rate (TPR) and the False Positive Rate (FPR).
        TPR = np.zeros_like(y_sig)  # True positive rate (sensitivity)
        FPR = np.zeros_like(y_sig)  # False positive rate ()

        # loop over all bins (x_centers) of the histograms and calculate TN, FP, FN, TP, FPR, and TPR for each bin
        for i, x in enumerate(x_centers):

            # the cut mask
            cut = x_centers < x
            # true positive
            TP = np.sum(y_sig[~cut]) / integral_sig  # True positives
            FN = np.sum(y_sig[cut]) / integral_sig  # False negatives
            TPR[i] = TP / (TP + FN)
            # True positive rate

            # true negative
            TN = np.sum(y_bkg[cut]) / integral_bkg  # True negatives (background)
            FP = np.sum(y_bkg[~cut]) / integral_bkg  # False positives
            FPR[i] = FP / (FP + TN)  # False positive rate
        if plot:
            fig, ax = plt.subplots(1, 1, figsize=(12, 8))
            ax.plot(FPR, TPR)
            ax.set(
                xlabel="False positive rate (FPR)",
                ylabel="True positive rate (TPR)",
                title="ROC curve",
            )
            if savefig != "":
                plt.savefig(savefig + ".pdf", dpi=500)
            plt.show()
        return FPR, TPR
    else:
        AssertionError(
            "Signal and Background histograms have different bins and ranges"
        )


def calc_ROC_hist(hist1, hist2, plot=True, savefig=""):
    """Computes ROC data from two histogram objects
    ----------
    Params
    ----------
    hist1: list of len 3
        - output from plt.hist call on signal
    hist2: list of len 3
        - output from plt.hist call on background
    plot: Boolean
        - wether to plot the ROC curve
    --------
    Returns
    --------
    FPR: ndarray
        - Computed false positive rate
    TPR: ndarray
        - Computed True positive rate
    """
    # hist1 is signal, hist2 is background

    # first we extract the entries (y values) and the edges of the histograms
    y_sig, x_sig_edges, _ = hist1
    y_bkg, x_bkg_edges, _ = hist2

    # Check that the two histograms have the same x edges:
    if np.array_equal(x_sig_edges, x_bkg_edges):

        # extract the center positions (x values) of the bins (doesn't matter if we use signal or background because they are equal)
        x_centers = 0.5 * (x_sig_edges[1:] + x_sig_edges[:-1])
        # calculate the integral (sum) of the signal and background
        integral_sig = y_sig.sum()
        integral_bkg = y_bkg.sum()

        # initialize empty arrays for the True Positive Rate (TPR) and the False Positive Rate (FPR).
        TPR = np.zeros_like(y_sig)  # True positive rate (sensitivity)
        FPR = np.zeros_like(y_sig)  # False positive rate ()

        # loop over all bins (x_centers) of the histograms and calculate TN, FP, FN, TP, FPR, and TPR for each bin
        for i, x in enumerate(x_centers):

            # the cut mask
            cut = x_centers < x

            # true positive
            TP = np.sum(y_sig[~cut]) / integral_sig  # True positives
            FN = np.sum(y_sig[cut]) / integral_sig  # False negatives
            TPR[i] = TP / (TP + FN)  # True positive rate

            # true negative
            TN = np.sum(y_bkg[cut]) / integral_bkg  # True negatives (background)
            FP = np.sum(y_bkg[~cut]) / integral_bkg  # False positives
            FPR[i] = FP / (FP + TN)  # False positive rate
        if plot:
            fig, ax = plt.subplots(1, 1, figsize=(12, 8))
            ax.plot(FPR, TPR)
            ax.set(
                xlabel="False positive rate (FPR)",
                ylabel="True positive rate (TPR)",
                title="ROC curve",
            )
            if savefig != "":
                plt.savefig(savefig + ".pdf", dpi=500)
            plt.show()
        return FPR, TPR
    else:
        AssertionError(
            "Signal and Background histograms have different bins and ranges"
        )


def RF_ROC(
    X, y, savefig="", classnames=[], plot=True, bins=500, labelsL=None, labelsR=None
):
    """Does Random Forest training on training data of two classes
    ----------
    Params
    ----------
    X: ndarray of (n_samples,n_features)
        - input data
    y: ndarray of (n_samples)
        - labels for the data
    savefig: optional, string
        - name for the savefile, if nothing specified it will not save plot
    classnames: optional, list of length two
        - names for classes, if nothing speciefied defaults to 0,1
    bins: int
        - number of bins to use in plot of seperation
    ----------
    Returns
    ----------
    fpr: 1D ndarray
        - False positive rate for the ith treshold
    tpr: 1D ndarray
        - True positive rate for ith threshold
    thresholds: 1D ndarray
        - Thresholds used to calc fpr and tpr
    val1: list [bincen0,count0,err0]
        - Histogram data for 0 class
    val2: list [bincen1,count1,err1]
        - Histogram data for 1 class
    """
    if len(classnames) == 0:
        classnames = ["0", "1"]
    clf = RandomForestClassifier(oob_score=True)
    clf = clf.fit(X, y)
    scores = clf.oob_decision_function_[~np.isnan(clf.oob_decision_function_[:, 0])][
        :, 0
    ]
    fpr, tpr, thresholds = metrics.roc_curve(
        y[~np.isnan(clf.oob_decision_function_[:, 0])], scores
    )
    scoremask = y[~np.isnan(clf.oob_decision_function_[:, 0])] == 0
    x0, y0, sy0 = histogram(scores[scoremask], bins=bins, range=(0, 1))
    x1, y1, sy1 = histogram(scores[~scoremask], bins=bins, range=(0, 1))
    if plot:
        fig2, ax = plt.subplots(1, 2, figsize=(12, 8))
        ax[0].errorbar(x0, y0, yerr=sy0, color="b", label=classnames[0])
        ax[0].errorbar(x1, y1, yerr=sy1, color="r", label=classnames[1])
        if labelsL is None:
            ax[0].set(
                xlabel="Normalized fisher discriminant",
                ylabel=f"frequency / {1/500:4.2E}",
                title="Efficiency of the Fisher discriminant",
            )
        else:
            ax[0].set(**labelsL)
        ax[0].legend()

        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        ax[1].plot(
            tpr,
            fpr,
            color="darkorange",
            lw=3,
            label="ROC curve (area = %0.2f)" % metrics.auc(tpr, fpr),
        )
        ax[1].plot([0, 1], [0, 1], color="navy", linewidth=3, linestyle="--")
        if labelsR is None:
            ax[1].set(
                xlabel="True Positive Rate",
                ylabel="False Positve Rate",
                title=f"ROC for Fisher discriminant classifier (area = {metrics.auc(tpr,fpr):4.4f})",
            )
        else:
            ax[1].set(**labels)
        ax[1].legend()
        if savefig != "":
            plt.savefig(savefig + ".pdf", dpi=500)
    return fpr, tpr, thresholds, [x0, y0, sy0], [x1, y1, sy1]


def Get_ROC_erros(data1, data2, threshold, table=True):
    """Gives confusion matrix for a given threshold from a binary decison distibution
    ----------
    Params
    ----------
    data1: [bin_centers,counts]
        - histogram data for background
    data2: [bin_centers,counts]
        - histogram data for signal
    threshold: float
        - Threshold on which to base decision
    table: optional, boolean
        - Wether to print a latex table of the confusion matrix
    """

    x_sig_centers, y_sig = data1
    x_bkg_centers, y_bkg = data2
    if np.array_equal(x_sig_centers, x_bkg_centers):
        x_centers = x_sig_centers
        # calculate the integral (sum) of the signal and background
        integral_sig = y_sig.sum()
        integral_bkg = y_bkg.sum()

        # the cut mask
        cut = x_centers < threshold
        # true positive
        TP = np.sum(y_sig[~cut]) / integral_sig  # True positives
        FN = np.sum(y_sig[cut]) / integral_sig  # False negatives
        TPR = TP / (TP + FN)
        # True positive rate

        # true negative
        TN = np.sum(y_bkg[cut]) / integral_bkg  # True negatives (background)
        FP = np.sum(y_bkg[~cut]) / integral_bkg  # False positives
        FPR = FP / (FP + TN)
    if table:
        latex = (
            "\\begin{center}\n"
            + "\\begin{tabular}{| l | l | l |}\n"
            + "\\hline\n"
            + f" & True & False\\\\ \\hline\n"
            + f"Fail to reject & {round(TN,3)} & {round(FN,3)}"
            + f"\\\\ \\hline\n"
            + f"Reject & {round(FP,3)} & {round(TP,3)}\\\\ \\hline\n"
            + "\\end{tabular}\n"
            + "\\end{center}\n"
        )
    return TN, TP, FP, FN, latex


def Fisher_ROC(X, y, savefig="", classnames=[], bins=500, labelsL=None, labelsR=None):
    """Does fisher linear discriminant analysis on training data of two classes
    ----------
    Params
    ----------
    X: ndarray of (n_samples,n_features)
        - input data
    y: ndarray of (n_samples)
        - labels for the data
    savefig: optional, string
        - name for the savefile, if nothing specified it will not save plot
    classnames: optional, list of length two
        - names for classes, if nothing speciefied defaults to 0,1
    bins: int
        - number of bins to use in plot of seperation
    ----------
    Returns
    ----------
    fpr: 1D ndarray
        - False positive rate for the ith treshold
    tpr: 1D ndarray
        - True positive rate for ith threshold
    thresholds: 1D ndarray
        - Thresholds used to calc fpr and tpr
    val1: list [bincen0,count0,err0]
        - Histogram data for 0 class
    val2: list [bincen1,count1,err1]
        - Histogram data for 1 class
    """
    lda = LinearDiscriminantAnalysis(solver="svd")
    scores = lda.predict_proba(X)
    fpr, tpr, thresholds = metrics.roc_curve(y, scores)
    discr = lda.fit(X, y).transform(X)
    bacgr, sig = discr[y == 0], discr[y == 1]
    x0, y0, sy0 = histogram(bacgr, bins=bins)
    x1, y1, sy1 = histogram(sig, bins=bins)
    if plot:
        fig2, ax = plt.subplots(1, 2, figsize=(12, 8))
        ax[0].errorbar(x0, y0, yerr=sy0, color="b", label=classnames[0])
        ax[0].errorbar(x1, y1, yerr=sy1, color="r", label=classnames[1])
        ax[0].set(
            xlabel="Out-of-bag error",
            ylabel=f"frequency / {1/500:4.2E}",
            title="Efficiency of the Out-of-bag error",
        )
        ax[0].legend()

        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        ax[1].plot(
            tpr,
            fpr,
            color="darkorange",
            lw=3,
            label="ROC curve (area = %0.2f)" % metrics.auc(tpr, fpr),
        )
        ax[1].plot([0, 1], [0, 1], color="navy", linewidth=3, linestyle="--")
        ax[1].set(
            xlabel="True Positive Rate",
            ylabel="False Positve Rate",
            title=f"ROC for Random forest classifier (area = {metrics.auc(tpr,fpr):4.4f})",
        )
        ax[1].legend()
        if savefig != "":
            plt.savefig(savefig + ".pdf", dpi=500)
    return fpr, tpr, thresholds, [x0, y0, sy0], [x1, y1, sy1]


# """
# Places to look for python inspiration
# - Week1/ProgrammingIntro/NiceFigure.ipynb (nice plot with inset)
# - Week2/LikelihoodFit/LikelihoodFit.ipynb (plot with shaded box)
# - Week3/ExampleLikelihoodFit/TrackMin... (3D plot of Minuit progress)
# - Week5/Calibration/Calibration.ipynb (Calibration and nice 2D histograms)
# - Week5/FittingAndTestingDistributions (coloring text insets)
# """
