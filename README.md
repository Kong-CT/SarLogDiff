# <div align="center">  LogSaDiff: Logarithm-Domain and Structure-Aware Conditional Diffusion Model for SAR Image Generation </div>

## Abstract

<div align="justify">
Synthetic Aperture Radar (SAR) image interpretation based on deep learning heavily depends on a large number of training samples. However, the acquisition of annotated SAR image samples remains a challenge, due to the high labor costs and the stochastic nature of coherent imaging mechanisms. In this paper, a logarithm-domain and structure-aware conditional diffusion model (LogSaDiff) is proposed for SAR image generation. In the proposed LogSaDiff, to explicitly adapt to the multiplicative speckle noise, the forward diffusion and reverse denoising processes are defined in the logarithm domain. This definition makes the diffusion and denoising processes in LogSaDiff equivalent to the corresponding processes with multiplicative noise in the original SAR image space. In addition, the structure-aware guidance and the category label are introduced as conditions to constrain the generation process of LogSaDiff, allowing the high-level semantics, imposed by the conditions, to be preserved during the generation process. Extensive experiments on real and synthetic SAR data demonstrate that LogSaDiff can generate high-quality SAR images, and the synthetic training samples lead to improvement in the classification task. Notably, ablation studies verify that the logarithm domain results in an improvement in SAR image generation, and the structure-aware guidance is effective in preserving structural semantics.
</div>

<div align="center">
 <img src="/Framework.png" alt="Framework of the LogSaDiff. To explicitly adapt to the intrinsic characteristics of SAR images, LogSaDiff is defined in the logarithm domain, and
its generation process is constrained by the structure-aware guidance to preserve high-level semantics from the reference image and category label."/>
</div>
</div>

## Installation

Create a conda environment with the following command:

```
conda env create -f environment.yml
```

## Training

First, you need to do some preparations:

1. Prepare the dataset. If you want to use your own dataset, the folder structure should resemble this:

~~~
|-- data
|-- youedataset
|   |-- train
|   |   |class0
|   |   |   |0_ *.jpg/tif/png
|   |-- val
|   |   |class0
|   |   |   |0_ *.jpg/tif/png
|   |-- refrence_image
|   |   |class0
|   |   |   |0_ *.jpg/tif/png
~~~

2. Prepare the directory to save your models and samples:

```
mkdir sample
mkdir model
```

Then, you can start training:

```
make train
```

## Sampling

To get pictures, you need to excute the following command:

```
make samplepict
```

To get .npz files(used for FID calculation), you need to excute the following command:

```
make samplenpz
```

## Evaluation

To calculate the FID, you need to excute the following command:

```
python evaluate/FID.py
```

To calculate the PSNR and SSIM, you need to excute the following command:

```
python evaluate/PSNR_SSIM.py
```

To calculate the Inception Score, you need to train excute the SCNN model to extract the features:

```
python evaluate/train_scnn.py
```

Then excute the Inception Score:

```
python evaluate/IS_SCNN.py
```

## Results

<div align="center">
 <img src="/Visualization.png" alt="Visualization of the generated samples on Orchard data. The first row
shows real samples, randomly selected from the test set for each class. "/>
</div>



## Pubulication

If you use this software in your research, please cite our publication:

```

```
