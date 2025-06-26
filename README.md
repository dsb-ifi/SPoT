<div align="center">

# SPoT: Subpixel Placement of Tokens in Vision Transformers

**[Martine Hjelkrem-Tan](https://www.mn.uio.no/ifi/english/people/aca/matan/), [Marius Aasan](https://www.mn.uio.no/ifi/english/people/aca/mariuaas/), [Gabriel Yanci Arteaga](https://www.mn.uio.no/ifi/english/people/aca/gabrieya/), [Adín Ramírez Rivera](https://www.mn.uio.no/ifi/english/people/aca/adinr/)** <br>


**[DSB @ IFI @ UiO](https://www.mn.uio.no/ifi/english/research/groups/dsb/)** <br>

[![Website](https://img.shields.io/badge/Website-green)](https://dsb-ifi.github.io/SPoT/)
[![PaperArxiv](https://img.shields.io/badge/Paper-arXiv-red)](https://arxiv.org)
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

You can load the Superpixel Transformer model easily via `torch.hub`:

```python
model = torch.hub.load(
    'dsb-ifi/spot', 
    'spot_vit_base_16_in21k',
    pretrained=True,
    source='github',
)
```

This will load the model and downloaded the pretrained weights, stored in your local `torch.hub` directory. 

## More Examples

We provide a [Jupyter notebook](https://nbviewer.jupyter.org/) as a sandbox for loading, evaluating, and extracting token placements for the models. 

## Citation

If you find our work useful, please consider citing our paper.

```
@inproceedings{hjelkremtan2025spot,
  title={SPoT: Subpixel Placement of Tokens in Vision Transformers},
  author={Hjelkrem-Tan, Martine and Aasan, Marius and Yanci Arteaga, Gabriel, and Ram\'irez Rivera, Ad\'in},
  journal={{CVF/ICCV} Efficient Computing under Limited Resources: Visual Computing ({ECLR} {ICCVW})},
  year={2025}
}
```
