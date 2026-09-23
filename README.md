# DeepSPT
## Deep Learning Assisted Analysis of Single Particle Tracking for Automated Correlation Between Diffusion and Function
DeepSPT, a deep learning framework to interpret the diffusional 2D or 3D temporal behavior of objects in a rapid and efficient manner, agnostically. DeepSPT is a deep learning framework, encompassing three sequentially connected modules: A temporal behavior segmentation module; a diffusional fingerprinting module; and a task-specific downstream classifier module (Fig. 1a). The first two modules are universal, applicable directly to any trajectory dataset characterized by x, y, (z) and t coordinates across diverse biological systems. The final module capitalizes on experimental data to learn a task that is specific to the system under investigation.

![image](_Images/figure1.png)
### Citing
https://www.nature.com/articles/s41592-025-02665-8

If you use DeepSPT, please cite:

```bibtex
@article{kaestelhansen2025deepspt,
  title   = {Deep learning-assisted analysis of single-particle tracking for automated correlation between diffusion and function},
  author  = {K{\ae}stel-Hansen, Jacob and de Sautu, Marilina and Saminathan, Anand and Scanavachi, Gustavo and Bango Da Cunha Correia, Ricardo F. and Nielsen, Annette Juma and Bleshøy, Sara Vogt and Tsolakidis, Konstantinos and Boomsma, Wouter and Kirchhausen, Tomas and Hatzakis, Nikos S.},
  journal = {Nature Methods},
  year    = {2025},
  volume  = {22},
  number  = {5},
  pages   = {1091--1100},
  doi     = {10.1038/s41592-025-02665-8}
}
```

### Usage
#### Installation
DeepSPT targets **Python 3.10** and installs with plain `pip` — no conda, no manual `probfit`/`iminuit` pinning. `probfit` and `pomegranate` 0.14 (used by the original Python 3.8 environment) ship no Python 3.10 wheel, so `deepspt_src` was ported to `iminuit.cost.LeastSquares` and a self-contained `HiddenMarkovModel` implementation instead; nothing extra needs to be installed by hand.

```bash
git clone git@github.com:hatzakislab/DeepSPT.git
cd DeepSPT
python3.10 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

That installs everything `deepspt_src` needs. If you also want to run the paper-reproduction material in `_For_publication/`, `Unet_mlflow_utils/`, or `deepspt_mlflow_utils/`, install the extra dependencies those scripts need on top:

```bash
pip install -r requirements_publication.txt
```

If you prefer conda, `environment_droplet_minimal.yml` and `DeepSPT_simple.yml` are Python 3.10 equivalents of `requirements.txt` — same story, no manual `probfit`/`iminuit` install needed:

```bash
conda env create -f DeepSPT_simple.yml   # or environment_droplet_minimal.yml
conda activate DeepSPT                   # or simpleDeepSPT, matching the file used
```

`environment_droplet.yml` is the **original, unported Python 3.8 environment** (kept for anyone who needs to reproduce results exactly as the paper's environment). It still requires the manual steps from the original install, since it predates the Python 3.10 port:

```bash
conda env create -f environment_droplet.yml
conda activate DeepSPT
pip install probfit==1.2.0
pip install iminuit==2.11.0
```

DeepSPT modules and additional/helpful functions are contained in the `deepspt_src` folder.
When running/building scripts in the DeepSPT directory modules are imported as:
```python
from deepspt_src import *

```
Three test python scripts are provided:
  - `simulate_diffusion.py` - Data generation of 2D or 3D diffusion of heterogeneous/homogeneous motion.
  - `usage_example0.py` - Usage example for loading numpy array saved as pickle or csv file.
  - `usage_example1.py` - Usage example for the three DeepSPT modules: Temporal segmentation, diffusional fingerprinting and task-specific classifier module on simulated data. This code transforms trajectories by temporal segmentation of diffusion and provide diffusional fingerprints to generate feature representation of trajectories both in the form of NumPy arrays. Runtime depends on dataset size but runs in less than 10 minutes for typical data volumes.
  - `usage_example2.py` - Usage example for the three DeepSPT modules: Temporal segmentation, diffusional fingerprinting and task-specific classifier module for time-resolved classification on simulated data. This code transforms trajectories by temporal segmentation of diffusion and provide diffusional fingerprints both in a temporal manner and returns the representations in the form of NumPy arrays. Runtime depends on dataset size but runs in less than 10 minutes for typical data volumes.

### For demostration
For demostration regarding presented data and analysis contained in the manuscript, please refer to the `_For_publication` folder where you will find the required information and scripts. To run on the same data download the data as outlined below.

### Data
  - Your own: DeepSPT accepts csv files or numpy arrays of shape (number of tracks, x,y,(z)). The csv files should contain columns named ['x', 'y', 'z', 'particle', 'frame']. Where 'x','y','z' represent coordinates, 'particle' represents particle id (number) and frame represents timepoint (number). The order is not of major importance. For 2D usecases 'z' column should not be included. The numpy arrays inputed as pickle files should have the shape (number of tracks, length of track, 5 or 4) 5 (particle id, frame, x, y, z) for 3D cases and 4 (particle id, frame, x, y) for 2D cases
  - Simulated data: simulate_diffusion.py, usage_example.py, and usage_example2.py (WIP) contains functions to simulate trajectories.
  - To access data of the publication "Deep learning assisted Single Particle Tracking for automated correlation between diffusion and function" please download from: TBA. Please extract .zip files in place of folders with the same names.

For just code:
https://erda.ku.dk/archives/752e4b0695c0dd16ec3c1a130f6ac70b/published-archive.html

for Code and models:
https://erda.ku.dk/archives/4c5adaaacc5c867f6450bcf89ec55a45/published-archive.html

For models and data:
https://erda.ku.dk/archives/804ea1ea88f340b79ada3e57141a6d6e/published-archive.html

Trained model weights (`mlruns/`, ~1.6 GB) are **not** stored in this git repository — download them from one of the "models" links above and place the `mlruns/` folder at the repo root.


### Files
  - For_publication: Scripts as used in "Deep Learning Assisted Analysis of Single Particle Tracking for Automated Correlation Between Diffusion and Function". Folders with data and precomputed files are available, see Data availability.
  - _Images: Contains figure seen in Readme. Copyrighted as detailed in journal carrying "Deep learning assisted Single Particle Tracking for automated correlation between diffusion and function".
  - deepspt_mlflow_utils: MLflow helper functions
  - deepspt_src: Source code for DeepSPT
  - environment_droplet.yml: original Python 3.8 conda environment (unported; needs the manual probfit/iminuit install above).
  - environment_droplet_minimal.yml / DeepSPT_simple.yml: Python 3.10 conda environments, equivalent to requirements.txt.
  - requirements.txt / requirements_publication.txt: Python 3.10 pip install, see Installation above.
  - mlruns: pretrained model weights, not tracked in git — see Data section below for download links.

  - Pickle files found in this repo contain arrays of tracks as described in Data section in the Readme. These include both pickle files of trajectories from experimental data and simulated data.
  - Test pickle files for DeepSPT can be found under the _Data folder. The pickle (.pkl) files shape following the above mentioned (see line 81) shape and order.
  

### Runnng on data from publication
- Firstly, download all data as stated in the "Data availability" section in "Deep Learning Assisted Analysis of Single Particle Tracking for Automated Correlation Between Diffusion and Function". Roughly 117 GB in zipped version.
-  Secondly, follow the install instructions described above.
-  Thirdly, the folder "_For_publication" contains all scripts used for the publication and running these will completely reproduce all results. Specifically, temporalsegm_eval.py contains most code for figure 2, timeresolved_uncoating_prediction.py contains most code for figure 3, and benchmark_for_fig4.py contains most code for figure 4.


## Instructions for DeepSPT application - GUI
In this repository you will find the "DeepSPT_GUI_manual.pdf", providing detailed insturcitons on how the applciation of DeepSPT - GUI should be used.
  
### Contact

Jacob Kæstel-hansen, PhD fellow\
Department of Chemistry\
jkh@chem.ku.dk

Konstantinos Tsolakidis,Software Engineer
Department of Chemistry, Hatzakis Lab 
kt@chem.ku.dk

Nikos Hatzakis, Professor\
Department of Chemistry\
hatzakis@chem.ku.dk

or commit an issue to this github. 
