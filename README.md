# <div align="center">  LogSarDiff: Logarithm-Domain and Structure-Aware Conditional Diffusion Model for SAR Image Generation </div>

## Abstract

<div align="justify">
Synthetic aperture radar (SAR) image generation using generative models is an effective way to deal with the few-shot learning problem in automatic target recognition (ATR). Recently, diffusion models have emerged as promising probabilistic generative models for image generation, which approximates the image distribution through reversing a predefined forward diffusion process. However, diffusion models remain ineffective for SAR image generation due to the intrinsic characteristics of SAR images. In this paper, we propose SarLogDiff, a logarithmic diffusion model with structure-aware guidance for SAR image synthesis. In the SarLogDiff, the forward and reverse processes are built upon the logarithmic diffusion, which enables the distribution to be approximated approaches closer towards the Gaussian distribution. In addition, SarLogDiff is controlled by the structure-aware guidance, by which the high-level semantics from conditions are preserved during generation. Extensive experiments on both real and synthetic data demonstrate that SarLogDiff can generate high-quality SAR images, and the synthetic samples by SarLogDiff improve the performance of the downstream classification task.
</div>

<div align="center">
 <img src="/Framework1.png" alt="Framework of the LogSaDiff. To explicitly adapt to the intrinsic characteristics of SAR images, LogSaDiff is defined in the logarithm domain, and
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
|-- yourdataset
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
 <img src="/Visualization.png" alt=" Visualization of the generated samples on SythRESI data. The first
row shows real samples, randomly selected from the test set for each class.
From left to right: bridge, desert, forest, freeway, and lake. "/>
</div>



