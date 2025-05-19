# ManifoldEntropicMetrics
Code repository for the paper "Analyzing Generative Models by Manifold Entropic Metrics" https://arxiv.org/abs/2410.19426

See conference [poster](Poster_v3.pdf).

## Installation
Create new conda environment with the following commands:
    
    conda create python=3.12 -c pytorch -n manentmet
    conda activate manentmet
    pip install -r requirements.txt

## Files
minimal_example.ipynb includes the Proof of concept experiments on 1. "Two Moons" and 2. "10-D Torus"

beta_vae_tests.ipynb includes the experiments on beta-VAEs

## Citation
beta-VAE Code adapted from https://github.com/ludovicobuizza/gan_vae

PF_objective adapted from paper "Principal Manifold Flows" https://arxiv.org/abs/2202.07037
