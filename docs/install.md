## Installation

### Requirements (tested)

- Linux
- Python 3.8+
- PyTorch 1.9.0
- CUDA 11.1
- GCC 5+
- GPU: RTX [30-series, A-series]
- [mmcv > 1.3](https://github.com/open-mmlab/mmcv)
- [BboxToolkit 1.0](https://github.com/jbwang1997/BboxToolkit)

##### Refer to instructions below to install above requirements.
---
### Install

a. Create a conda virtual environment and activate it.

```shell
conda create -n apdetection python=3.7 -y
conda activate apdetection
```

b. Install PyTorch and torchvision following the [official instructions](https://pytorch.org/), e.g.,

If you have CUDA 11.1 installed under `/usr/local/cuda` and would like to install
PyTorch 1.9.0., you need to install the prebuilt PyTorch with CUDA 11.1.

```python
pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html
```

c. Clone the APDetection repository.

```shell
git clone https://github.com/chandlerbing65nm/APDetection.git
cd APDetection
```

d. Install build requirements and then install APDetection.

- install the BboxToolkit

```shell
git clone https://github.com/jbwang1997/BboxToolkit
cd BboxToolkit
pip install -v -e .
```

- install mmcv-full

Please refer to [mmcv-full](https://github.com/open-mmlab/mmcv) to select a compatible version of mmcv-full 
```shell
# example for pytorch 1.9.0 and cuda 11.1
pip install mmcv-full==1.3.9 -f https://download.openmmlab.com/mmcv/dist/cu111/torch1.9.0/index.html
# Here, the version of cuda and torch should be the same with your environment.
```

- install APDetection

```
pip install -r requirements/build.txt
pip install mmpycocotools
pip install -v -e .
```
