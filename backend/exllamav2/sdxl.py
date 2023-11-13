import torch
#from diffusers import StableDiffusionXLPipeline
from diffusers import StableDiffusionPipeline
from PIL import Image

# Load the pipeline model
#pipe = StableDiffusionXLPipeline.from_single_file("/run/media/privateserver/NECTAC1TBSSD/a1111/stable-diffusion-webui/models/Stable-diffusion/realvisxlV20_v20Bakedvae.safetensors", torch_dtype=torch.float16).to("cuda")
pipe = StableDiffusionPipeline.from_single_file("/run/media/privateserver/NECTAC1TBSSD/a1111/stable-diffusion-webui/models/Stable-diffusion/sd15.safetensors", torch_dtype=torch.float16).to("cuda")


'''
while True:
    prompt = input("Enter your prompt:")

    # Generate an image using a prompt
    pipe.safety_checker = None
    pipe.requires_safety_checker = False
    image = pipe(prompt=prompt, num_inference_steps=30).images[0]
    
    # Save the image directly as PNG
    image.save("generated_image.png")
