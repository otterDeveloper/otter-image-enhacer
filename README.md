# Otter Image Enhancer

The otter image enhancer is a batch processing tool for upscaling folders of images into higher resolution using OpenCV's super resolution and the [ESDR models](https://arxiv.org/pdf/1707.02921.pdf).

It downloads the models at runtime from [Saafke/EDSR_Tensorflow](https://github.com/Saafke/EDSR_Tensorflow)

## Requirements

- At least Python 3.8

## Usage

Firs install requirements.txt
`pip install -r requirements.txt`

You can the run the script with: `python main.py [input folder] [method]`

Example: `python main.py input/images shrink`

The methods available are:

- shrink: It first downscales the images by a factor of 2 then it upscales by 4x using ESDR, this method helps if you have very noisy images
- upscale: It skips the shrink and only upscales the images by a factor of 4
- denoise: Runs Open CV fast denoising algorithm

The script will process images *up to 4 sub-folders deep*

## Output

The processed images will be saved to the output folder in a subfolder named with timestamp of the start of the and the process used.
Example: `output/2022-07-22_21-36-00_shrink`

It will include a log and a manifest of the images in CSV.
