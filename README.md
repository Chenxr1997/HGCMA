# HGCMA

This repo is for source code of paper "Heterogeneous Graph Contrastive Learning with Metapath-Based Augmentations".

## Environment Settings
> python==3.7.3 \
> scipy==1.2.1 \
> torch==1.7.1 \
> numpy==1.21.5 \
> scikit_learn==1.0.1

## Usage
Fisrt, go into ./src, and then you can use the following commend to run our model: 
> python main.py dataset --gpu=0

Here, "dataset" can be "acm", "dblp" or "imdb".
