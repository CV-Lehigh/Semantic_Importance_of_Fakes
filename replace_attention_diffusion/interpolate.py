import torch
from PIL import Image
import argparse

from pipeline_interpolated_sd import InterpolationStableDiffusionPipeline
from prior import BetaPriorPipeline
from utils import image_grids
import numpy as np
import os
from diffusers import AutoencoderKL
from torchvision import transforms

transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

# rows = np.load('captions_most_alike.npy')

generator = torch.cuda.manual_seed(250)
dtype = torch.float16

pipe = InterpolationStableDiffusionPipeline.from_pretrained(
    "stable-diffusion-v1-5/stable-diffusion-v1-5", torch_dtype=torch.float16, safety_checker=None
)
pipe.to("cuda", dtype=dtype)
beta_pipe = BetaPriorPipeline(pipe)

vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse").to("cuda")

size = pipe.unet.config.sample_size

def main():
    parser = argparse.ArgumentParser(description="Run a Stable Diffusion experiment by licking to a configuration file.")
    parser.add_argument('--prompt_in', type=str, default="a cowboy jumping off a rocket ship.| blurry, low resolution",
                        help='prompt to be used as starting embedding.')
    parser.add_argument('--prompt_out', type=str, default="a cowboy jumping off a horse. | blurry, low resolution",
                        help='prompt to be interpolated towards')
    parser.add_argument('--latent_in', type=str, default=-1)
    parser.add_argument('--latent_out', type=str, default=-1)
    parser.add_argument('--idx', type=str, default=1111,
                        help='starting image index')
    args = parser.parse_args()

    prompt_a = args.prompt_in
    prompt_b = args.prompt_out
    p_idx = args.idx

    if args.latent_sta != -1 and args.latent_out != -1:
        latent_start = torch.load(f'YOUR DATA PATH/laion-5B/img_emb_1_5/{args.latent_in}_img2img_single-inference/embeddings/output-0_diffstep-25.pt')['latent_noise'] 
        latent_end = torch.load(f'YOUR DATA PATH/laion-5B/img_emb_1_5/{args.latent_out}_img2img_single-inference/embeddings/output-0_diffstep-25.pt')['latent_noise'] 
    else:
        latent_start = torch.randn((1, 4, size, size,), device="cuda", dtype=dtype, generator=generator)
        latent_end = latent_start




    negative_prompt = "nsfw, lowres, (bad), text, error, fewer, extra, missing, worst quality, jpeg artifacts, low quality, watermark, unfinished, displeasing, oldest, early, chromatic aberration, signature, extra digits, artistic error, username, scan, [abstract]" # "deformed, ugly, wrong proportion, low res, bad anatomy, worst quality, low quality"
    interpolation_size = 11
    num_inference_steps = 25

    images = beta_pipe.generate_interpolation(
        prompt_a,
        prompt_b,
        negative_prompt,
        latent_start,
        latent_end,
        num_inference_steps,
        exploration_size=int(interpolation_size * 1.0),
        interpolation_size=interpolation_size,
        warmup_ratio=0.5
    )
    folder_path = f'YOUR DATA PATH/prompt_close_PAID_0/{p_idx}'
    if not os.path.exists(folder_path):
        os.makedirs(f"{folder_path}")
        os.makedirs(f"{folder_path}/images/")
        os.makedirs(f"{folder_path}/embedding/")

    for i, image in enumerate(images):
        # Save the image
        image.save(f"{folder_path}/images/{i}.png")

        #Save tensor
        image_tensor = transform(image).unsqueeze(0).to("cuda")
        with torch.no_grad():
            latent = vae.encode(image_tensor).latent_dist.mean
        torch.save(latent, f"{folder_path}/embedding/{i}.pt")


if __name__ == "__main__":
    main()