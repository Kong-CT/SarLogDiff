# <div align="center">  Logarithmic Diffusion Model with Structure-Aware Guidance for SAR Image Synthesis </div>

## Abstract

<div align="justify">
Generative models offer a promising solution for synthetic aperture radar (SAR) image generation to address the few-shot learning problem in automatic target recognition (ATR). Recently, diffusion models have emerged as a powerful class of generative models, which generate images by iteratively reversing a predefined forward diffusion process. However, applying diffusion models to SAR image generation remains challenging, due to the inherent speckle noise and non-Gaussian distribution of SAR images. To address these challenges, we propose SarLogDiff: a diffusion-based framework that explicitly adapts to SAR images through logarithmic diffusion and structure-aware guidance. SarLogDiff defines its forward and reverse processes using the logarithmic diffusion, and generates samples conditioned on the structure-aware guidance. The logarithmic diffusion enables SAR image distribution to approach a Gaussian distribution, while the structure-aware guidance preserves the high-level semantics from conditional input (e.g., a reference image) during generation. Extensive experiments demonstrate that SarLogDiff can generate high-quality SAR images, and the synthetic samples significantly enhance the ATR performance.
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



