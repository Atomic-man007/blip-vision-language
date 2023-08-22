"""
## BLIP image captioning 
"""


# pip install git+https://github.com/huggingface/transformers.git@main
# pip install torch
# pip install open_clip_torch
# pip install accelerate
# pip install bitsandbytes
# pip install scipy
# pip install gradio


import torch
import open_clip
import gradio as gr
from PIL import Image
from transformers import AutoProcessor, BlipForConditionalGeneration


device = "cuda" if torch.cuda.is_available() else "cpu"
torch.hub.download_url_to_file('http://images.cocodataset.org/val2017/000000039769.jpg', 'cats.jpg')


"""
### Download and load the Pre-trained model
"""


blip_processor_large = AutoProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
blip_model_large = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large").to(device)


def generate_caption(processor, model, image, tokenizer=None, use_float_16=False):
    inputs = processor(images=image, return_tensors="pt").to(device)

    if use_float_16:
        inputs = inputs.to(torch.float16)

    generated_ids = model.generate(pixel_values=inputs.pixel_values, num_beams=3, max_length=20, min_length=5)

    if tokenizer is not None:
        generated_caption = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    else:
        generated_caption = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

    return generated_caption

def generate_captions(image):
    caption_blip_large = generate_caption(blip_processor_large, blip_model_large, image)
    return  caption_blip_large


print(generate_captions(Image.open("/content/cats.jpg")))
"""
Answer :- there are two cats laying on a couch with remote controls
"""