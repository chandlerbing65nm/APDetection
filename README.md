# Improving Localization in Oriented Aerial Detection with Attention-Points Network

## Abstract

Localization in computer vision is the process of finding and identifying objects in an image. In aerial images, objects are usually very-small and arbitrary-oriented. Current state-of-the-art oriented aerial detectors are focused only on finding objects and fewer efforts have been made on advancing both classification (identify) and regression (find) methods. To tackle both the classification and regression problems, we used the architectures of existing oriented aerial detectors and improved their performance by inserting our designed Attention-Points Network consisting of two losses: Guided-Attention Loss (GALoss) and Box-Points Loss (BPLoss). GALoss uses a coarse-level segmentation mask as ground-truth to learn the rich attention features essential to improve the classification of small objects. Meanwhile, BPLoss complements the existing regression loss by predicting box points and determining if they are inside or outside the oriented bounding box.

## Install

Please refer to [install.md](docs/install.md) for installation.


## Get Started

For dataset preparation and training/testing of our model, please refer to [get-started.md](docs/get-started.md).

## Citation
```bibtex
@article{Doloriel2022,
	title="Improving Localization in Oriented Aerial Detection with Attention-Points Network",
	author="Doloriel, C.T. and Cajote, R.D.",
	year="2022"
}
```
## Acknowledgements

Below are some great resources we used. We would like to thank the owners of:

[MMDetection](https://github.com/open-mmlab/mmdetection) \
[OBBdetection](https://github.com/jbwang1997/OBBDetection) \
[BBoxToolKit](https://github.com/jbwang1997/BboxToolkit) \
[DOTA Dataset](https://captain-whu.github.io/DOTA/) 

