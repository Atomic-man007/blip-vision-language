# BLIP vision language Image Captioning

This repository contains code for generating captions for images using the BLIP (Bootstrapping Language-Image Pre-training) framework. BLIP is a Vision-Language Pre-training (VLP) framework designed to excel in both understanding and generation tasks by salesforce. It effectively utilizes noisy web data through bootstrapping, generating synthetic captions, and filtering out noise for enhanced performance.

## Prerequisites

Make sure you have the following packages installed:

```
pip install git+https://github.com/huggingface/transformers.git@main
pip install torch
pip install open_clip_torch
pip install accelerate
pip install bitsandbytes
pip install scipy
pip install gradio
```

## Usage

Clone this repository or copy the provided code.
Install the required packages as mentioned above.
Run the provided code to generate captions for images.

## How It Works

The code demonstrates how to use the BLIP framework for generating captions for images. It uses a pre-trained BLIP model to process images and generate captions.

## Load a pre-trained BLIP model and processor.

Define a function to generate captions for a given image.
Use the function to generate captions for an example image.

## Example

![Alt text](./content/cats.jpg?raw=true "Cats")

The provided code includes an example output for generating captions for the "cats.jpg" image.

### Output:

`there are two cats laying on a couch with remote controls`

## Notes

The code provided is for demonstration purposes. You can modify and integrate it into your projects as needed.
Make sure to check the original BLIP paper and documentation for more details on the framework.

## References

BLIP: Bootstrapping Language-Image Pre-training for Unified Vision-Language Understanding and Generation
Hugging Face Transformers Library
OpenAI CLIP
Gradio
