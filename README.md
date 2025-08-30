# <div align="center">  LogSarDiff: Logarithm-Domain and Structure-Aware Conditional Diffusion Model for SAR Image Generation </div>

## Abstract

<div align="justify">
Diffusion models have recently been applied to Synthetic Aperture Radar (SAR) image generation, where the underlying distribution of SAR image is approximated by reversing a predefined forward diffusion process. However, diffusion models remain ineffective for SAR image generation, primarily because (i) the additive noise model, used in the diffusion process to degrade the clean SAR image, differs from the inherent degradation pattern of the multiplicative speckle noise; and (ii) the stochastic sampling mechanism in diffusion models poses challenges to the conditional generation. In this paper, we propose SarLogDiff, a conditional diffusion model for SAR image synthesis. In the proposed SarLogDiff, the input SAR image is first logarithmically transformed, and then the forward diffusion and reverse denoising processes are defined on this transformed data to align with the multiplicative degradation pattern. Furthermore, structure-aware guidance is presented as conditions for SarLogDiff to preserve the high-level semantics from the conditional information during the generation process. Extensive experiments on both real and synthetic data demonstrate that SarLogDiff can effectively generate high-quality SAR images, and the synthetic samples by SarLogDiff significantly enhance the performance of the downstream classification task. Notably, ablation studies confirm both the critical role of the logarithm-transformation in boosting synthesis quality and the effectiveness of structure-aware guidance in preserving high-level semantics.
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



