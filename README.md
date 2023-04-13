# <p align="center"> [Improving the Detection of Small Oriented Objects in Aerial Images](https://openaccess.thecvf.com/content/WACV2023W/MaCVi/papers/Doloriel_Improving_the_Detection_of_Small_Oriented_Objects_in_Aerial_Images_WACVW_2023_paper.pdf) </p>

<p align="center">
  <img 
    src="https://github.com/chandlerbing65nm/APDetection/blob/main/docs/attentionpoints-arch.jpg?raw=true"
  >
</p>

## Abstract

<p align="justify"> Small oriented objects that represent tiny pixel-area in large-scale aerial images are difficult to detect due to their size and orientation. Existing oriented aerial detectors have shown promising results but are mainly focused on orientation modeling with less regard to the size of the objects. In this work, we proposed a method to accurately detect small oriented objects in aerial images by enhancing the classification and regression tasks of the oriented object detection model. We designed the Attention-Points Network consisting of two losses: Guided-Attention Loss (GALoss) and Box-Points Loss (BPLoss). GALoss uses an instance segmentation mask as ground-truth to learn the attention features needed to improve the detection of small objects. These attention features are then used to predict box points for BPLoss, which determines the points' position relative to the target oriented bounding box. Experimental results show the effectiveness of our Attention-Points Network on a standard oriented aerial dataset with small object instances (DOTA-v1.5) and on a maritime-related dataset (HRSC2016). </p>

## Install

Please refer to [install.md](docs/install.md) for installation.


## Get Started

For dataset preparation and training/testing of our model, please refer to [get-started.md](docs/get-started.md).

## Cite
```bibtex
@InProceedings{Doloriel_2023_WACV,
    author    = {Doloriel, Chandler Timm C. and Cajote, Rhandley D.},
    title     = {Improving the Detection of Small Oriented Objects in Aerial Images},
    booktitle = {Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision (WACV) Workshops},
    month     = {January},
    year      = {2023},
    pages     = {176-185}
}
```
## Acknowledgements

Below are some great resources we used. We would like to thank the owners of:

[MMDetection](https://github.com/open-mmlab/mmdetection) \
[OBBdetection](https://github.com/jbwang1997/OBBDetection) \
[BBoxToolKit](https://github.com/jbwang1997/BboxToolkit) \
[DOTA Dataset](https://captain-whu.github.io/DOTA/) 

