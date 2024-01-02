# BD-SKUNet: Selective-Kernel UNets for Building Damage Assessment in High-Resolution Satellite Images
Authors: S. Ali Ahmadi, Ali Mohammadzadeh, Naoto Yokoya, Arsalan Ghorbanian

Official repository for "Selective-Kernel UNets for Building Damage Assessment in High-Resolution Satellite Images", which is published in [Remote Sensing - MDPI](https://www.mdpi.com/journal/remotesensing) journal. [Link](https://doi.org/10.3390/rs16010182) to get the open-access article.

![Graphical Abstract](https://github.com/Seyed-Ali-Ahmadi/BDA_SKUNet/assets/53389122/9fe1d5bb-4cf8-4d5e-9a32-1d9857d3f5b6)

## Requirements (keras)
* python 3.9.16
* albumentations
* scikit-image
* segmentation-models
* keras


The codes are not very clean, since they were used to make many experiments for the paper. If you have any questions in implementing the codes or getting your results, feel free to ask me on an Issue, or by emailing me on cpt.ahmadisnipiol@yahoo.com .

## Citation
```
https://doi.org/10.3390/rs16010182

@Article{rs16010182,
AUTHOR = {Ahmadi, Seyed Ali and Mohammadzadeh, Ali and Yokoya, Naoto and Ghorbanian, Arsalan},
TITLE = {BD-SKUNet: Selective-Kernel UNets for Building Damage Assessment in High-Resolution Satellite Images},
JOURNAL = {Remote Sensing},
VOLUME = {16},
YEAR = {2024},
NUMBER = {1},
ARTICLE-NUMBER = {182},
URL = {https://www.mdpi.com/2072-4292/16/1/182},
ISSN = {2072-4292},
ABSTRACT = {When natural disasters occur, timely and accurate building damage assessment maps are vital for disaster management responders to organize their resources efficiently. Pairs of pre- and post-disaster remote sensing imagery have been recognized as invaluable data sources that provide useful information for building damage identification. Recently, deep learning-based semantic segmentation models have been widely and successfully applied to remote sensing imagery for building damage assessment tasks. In this study, a two-stage, dual-branch, UNet architecture, with shared weights between two branches, is proposed to address the inaccuracies in building footprint localization and per-building damage level classification. A newly introduced selective kernel module improves the performance of the model by enhancing the extracted features and applying adaptive receptive field variations. The xBD dataset is used to train, validate, and test the proposed model based on widely used evaluation metrics such as F1-score and Intersection over Union (IoU). Overall, the experiments and comparisons demonstrate the superior performance of the proposed model. In addition, the results are further confirmed by evaluating the geographical transferability of the proposed model on a completely unseen dataset from a new region (Bam city earthquake in 2003).},
DOI = {10.3390/rs16010182}
}
```
