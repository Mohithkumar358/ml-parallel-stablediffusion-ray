import ray
import os
import time
import requests


ray.init(address="auto")


@ray.remote(num_gpus=1)
class LatentGeneration:
    def __init__(self):
        from diffusers import UNet2DConditionModel
        from transformers import CLIPTextModel, CLIPTokenizer 
        # Load the tokenizer and text encoder to tokenize and encode the text.
        self.tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
        self.text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14")

        # The UNet model for generating the latents.
        self.unet = UNet2DConditionModel.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="unet")

    def  generate_latents_image(self):
        import torch
        from torch import autocast
        from tqdm.auto import tqdm

        from diffusers import  LMSDiscreteScheduler

        torch_device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        if "mps" == torch_device: os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = "1"
        text_encoder = self.text_encoder.to(torch_device)
        unet = self.unet.to(torch_device) 
         # The noise scheduler
        scheduler = LMSDiscreteScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", num_train_timesteps=1000) 


        # Some settings
        prompt = ["Taj Mahal in the style of monet, Pastel, Overhead light, Symmetry, Flowers, Energeti"]
        height = 512                        # default height of Stable Diffusion
        width = 512                         # default width of Stable Diffusion
        num_inference_steps = 30         # Number of denoising steps
        guidance_scale = 7.5                # Scale for classifier-free guidance
        generator = torch.manual_seed(32)   # Seed generator to create the inital latent noise
        batch_size = 1
        start_time = time.time()
        imports_time = time.time()
        print(f"imports_time:{imports_time-start_time}")
        # Prep text
        text_input = self.tokenizer(prompt, padding="max_length", max_length=self.tokenizer.model_max_length, truncation=True, return_tensors="pt")
        with torch.no_grad():
            text_embeddings = text_encoder(text_input.input_ids.to(torch_device))[0]
        max_length = text_input.input_ids.shape[-1]
        uncond_input = self.tokenizer(
            [""] * batch_size, padding="max_length", max_length=max_length, return_tensors="pt"
        )
        with torch.no_grad():
            uncond_embeddings = text_encoder(uncond_input.input_ids.to(torch_device))[0]
        text_embeddings = torch.cat([uncond_embeddings, text_embeddings])
        text_embed_time = time.time()
        print(f"text_embed_time:{text_embed_time-start_time}")

        # Prep Scheduler
        def set_timesteps(scheduler, num_inference_steps):
            scheduler.set_timesteps(num_inference_steps)
            scheduler.timesteps = scheduler.timesteps.to(torch.float16) # minor fix to ensure MPS compatibility, fixed in diffusers PR 3925

        set_timesteps(scheduler,num_inference_steps)

        # Prep latents
        latents = torch.randn(
        (batch_size, unet.in_channels, height // 8, width // 8),
        generator=generator,
        )
        latents = latents.to(torch_device)
        latents = latents * scheduler.init_noise_sigma # Scaling (previous versions did latents = latents * self.scheduler.sigmas[0]

        # Loop
        with autocast("cuda"):  # will fallback to CPU if no CUDA; no autocast for MPS
            for i, t in tqdm(enumerate(scheduler.timesteps), total=len(scheduler.timesteps)):
                # expand the latents if we are doing classifier-free guidance to avoid doing two forward passes.
                latent_model_input = torch.cat([latents] * 2)
                sigma = scheduler.sigmas[i]
                # Scale the latents (preconditioning):
                # latent_model_input = latent_model_input / ((sigma**2 + 1) ** 0.5) # Diffusers 0.3 and below
                latent_model_input = scheduler.scale_model_input(latent_model_input, t)

                # predict the noise residual
                with torch.no_grad():
                    noise_pred = unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample

                # perform guidance
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

                # compute the previous noisy sample x_t -> x_t-1
                # latents = scheduler.step(noise_pred, i, latents)["prev_sample"] # Diffusers 0.3 and below
                latents = scheduler.step(noise_pred, t, latents).prev_sample
            latent_time = time.time()
            print(f"latent_time:{latent_time-text_embed_time}")

            # scale and decode the image latents with vae
            latents = 1 / 0.18215 * latents
        return latents

class ImageGeneration:
    def __init__(self):
        from diffusers import AutoencoderKL
        # Load the autoencoder model which will be used to decode the latents into image space.
        self.vae = AutoencoderKL.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="vae")

    def generate_image(self, latents):
        import torch
        from PIL import Image

        torch_device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        if "mps" == torch_device: os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = "1"
        vae = self.vae.to(torch_device)
        with torch.no_grad():
            image = vae.decode(latents).sample
            # Display
            image = (image / 2 + 0.5).clamp(0, 1)
            image = image.detach().cpu().permute(0, 2, 3, 1).numpy()
            images = (image * 255).round().astype("uint8")
            pil_images = [Image.fromarray(image) for image in images]
            print(pil_images)
            pil_images[0].save(f'text_to_image_script.jpg')
    

if __name__ == "__main__":
    gen = LatentGeneration.remote()
    
    result = gen.generate_latents_image.remote()
    latents = ray.get(result)
    print(len(latents))
    print(latents)
    image = ImageGeneration()
    image.generate_image(latents)