<div align="center">

# SPoT: Subpixel Placement of Tokens in Vision Transformers

**[Martine Hjelkrem-Tan](https://www.mn.uio.no/ifi/english/people/aca/matan/), [Marius Aasan](https://www.mn.uio.no/ifi/english/people/aca/mariuaas/), [Gabriel Y. Arteaga](https://www.mn.uio.no/ifi/english/people/aca/gabrieya/), [Adín Ramírez Rivera](https://www.mn.uio.no/ifi/english/people/aca/adinr/)** <br>


**[DSB @ IFI @ UiO](https://www.mn.uio.no/ifi/english/research/groups/dsb/)** <br>

[![Website](https://img.shields.io/badge/Website-green)](https://dsb-ifi.github.io/SPoT/)
[![PaperArxiv](https://img.shields.io/badge/Paper-arXiv-red)](https://arxiv.org/abs/2507.01654)
[![PaperICCVW](https://img.shields.io/badge/Paper-ICCVW_2025-blue)](https://eclr-workshop.github.io/)
[![NotebookExample](https://img.shields.io/badge/Notebook-Example-orange)](https://nbviewer.jupyter.org) <br>

![SPoT Figure 1](/assets/placements.png#gh-light-mode-only "Examples of feature trajectoreis with SPoT-ON")
![SPoT Figure 1](/assets/placements.png#gh-dark-mode-only "Examples of feature trajectoreis with SPoT-ON")

</div>

## SPoT: Subpixel Placement of Tokens

This repo contains code and weights for **SPoT: Subpixel Placement of Tokens**, accepted for ECLR, ICCVW 2025.

For an introduction to our work, visit the [project webpage](https://dsb-ifi.github.io/SPoT/). 

## Installation

The package can currently be installed via:

```bash
# HTTPS
pip install git+https://github.com/dsb-ifi/SPoT.git

# SSH
pip install git+ssh://git@github.com/dsb-ifi/SPoT.git
```

## Loading models

To load the model, first download the checkpoints from [Google Drive](https://drive.google.com/drive/folders/1ZABszElqoD3U83KXaLCtamb8DyDJPBYB?usp=sharing).
Then extract the checkpoints into a folder named `checkpoints/` in the repo.

The model can be loaded easily by

```
from spot.load_models import *

model_name = 'spot_mae_b16'
assert model_name in valid_models
model = load_trained_model(
    model_name=model_name,
    sampler='grid_center',      # Spatial prior
    ksize=16,                   # Window size
    n_features=25,              # Number of tokens
)
```


## More Examples

We provide a [Jupyter notebook](./get_started.ipynb) that illustrates loading, evaluating, and extracting token placements for the models. 

## Citation

If you find our work useful, please consider citing our paper.

```
@inproceedings{hjelkremtan2025spot,
  title={{SPoT}: Subpixel Placement of Tokens in Vision Transformers},
  author={Hjelkrem-Tan, Martine and Aasan, Marius and Arteaga, Gabriel Y. and Ram\'irez Rivera, Ad\'in},
  journal={{CVF/ICCV} Efficient Computing under Limited Resources: Visual Computing ({ECLR} {ICCVW})},
  year={2025}
}
```
