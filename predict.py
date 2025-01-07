# Prediction interface for Cog ⚙️
# https://github.com/replicate/cog/blob/main/docs/python.md

from cog import BasePredictor, Input, Path
import os
import time
import torch
import subprocess
import numpy as np
from PIL import Image
from typing import List
from diffusers import (
    FluxPriorReduxPipeline,
)
from pipeline_flux_compile import FluxPipelineCompile
from torchvision import transforms

from condition_reweighting_attention_processor import ConditionReweightingAttentionProcessor

DEV_CACHE = "./FLUX.1-dev"
SCHNELL_CACHE = "./FLUX.1-schnell"
DEV_URL = "https://weights.replicate.delivery/default/black-forest-labs/FLUX.1-dev/files.tar"
SCHNELL_URL = "https://weights.replicate.delivery/default/black-forest-labs/FLUX.1-schnell/files.tar"
REDUX_CACHE = "./FLUX.1-Redux"
REDUX_URL = "https://weights.replicate.delivery/default/black-forest-labs/FLUX.1-Redux-dev/model.tar"

MAX_SEQUENCE_LENGTH = 512


def download_weights(url, dest):
    start = time.time()
    print("downloading url: ", url)
    print("downloading to: ", dest)
    subprocess.check_call(["pget", "-xf", url, dest], close_fds=False)
    print("downloading took: ", time.time() - start)


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
WEIGHT_DTYPE = torch.bfloat16


class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load the model into memory to make running multiple predictions efficient"""
        start = time.time()

        print("Loading Flux Pipeline")
        if not os.path.exists(self.MODEL_CACHE):
            download_weights(self.MODEL_URL, ".")

        self.pipe = FluxPipelineCompile.from_pretrained(
            self.MODEL_CACHE,
            torch_dtype=WEIGHT_DTYPE,
            local_files_only=True,
        ).to("cuda")
        self.pipe.transformer.set_attn_processor(ConditionReweightingAttentionProcessor())

        if not os.path.exists(REDUX_CACHE):
            download_weights(REDUX_URL, REDUX_CACHE)

        self.pipe_prior_redux = FluxPriorReduxPipeline.from_pretrained(
            REDUX_CACHE,
            tokenizer=self.pipe.tokenizer,
            text_encoder=self.pipe.text_encoder,
            tokenizer_2=self.pipe.tokenizer_2,
            text_encoder_2=self.pipe.text_encoder_2,
            local_files_only=True,
        ).to(DEVICE)

        self.compile_flux()

        print("setup took: ", time.time() - start)

    def get_image(self, image: str):
        image = Image.open(image).convert("RGB")
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Lambda(lambda x: 2.0 * x - 1.0),
            ]
        )
        img: torch.Tensor = transform(image)
        return img[None, ...]

    @staticmethod
    def make_multiple_of_16(n):
        return ((n + 15) // 16) * 16

    @torch.inference_mode()
    def predict(
        self,
        prompt: str = Input(description="Text prompt", default=None),
        redux: Path = Input(
            description="Redux image.",
            default=None,
        ),
        redux_strength: float = Input(
            description="Strength of the Redux image, values between 0.01 and 0.1 tend to have good results.",
            default=0.05,
            ge=0,
            le=1,
        ),
        width: int = Input(description="Width, in pixels", default=1024),
        height: int = Input(description="Height, in pixels", default=1024),
        num_outputs: int = Input(
            description="Number of images to output.",
            ge=1,
            le=4,
            default=1,
        ),
        num_inference_steps: int = Input(
            description="Number of inference steps",
            ge=1,
            le=50,
            default=28,
        ),
        guidance_scale: float = Input(
            description="Guidance scale for the diffusion process",
            ge=0,
            le=10,
            default=3.5,
        ),
        seed: int = Input(description="Random seed. Set for reproducible generation", default=None),
        output_format: str = Input(
            description="Format of the output images",
            choices=["webp", "jpg", "png"],
            default="webp",
        ),
        output_quality: int = Input(
            description="Quality when saving the output images, from 0 to 100. 100 is best quality, 0 is lowest quality. Not relevant for .png outputs",
            default=80,
            ge=0,
            le=100,
        ),
    ) -> List[Path]:
        """Run a single prediction on the model"""
        if seed is None:
            seed = int.from_bytes(os.urandom(2), "big")
        print(f"Using seed: {seed}")

        generator = torch.Generator("cuda").manual_seed(seed)

        prompt_embeds, pooled_prompt_embeds, joint_attention_kwargs = self.generate_embeddings(
            prompt, redux, 1.0, redux_strength, num_outputs
        )

        flux_kwargs = {
            "width": width,
            "height": height,
            "prompt_embeds": prompt_embeds,
            "pooled_prompt_embeds": pooled_prompt_embeds,
            "joint_attention_kwargs": joint_attention_kwargs,
        }
        max_sequence_length = MAX_SEQUENCE_LENGTH
        common_args = {
            "guidance_scale": guidance_scale,
            "generator": generator,
            "num_inference_steps": num_inference_steps,
            "num_images_per_prompt": num_outputs,
            "max_sequence_length": max_sequence_length,
            "output_type": "pil",
        }

        output = self.pipe(**common_args, **flux_kwargs)

        output_paths = []
        for i, image in enumerate(output.images):
            output_path = f"./out-{i}.{output_format}"
            if output_format != "png":
                image.save(output_path, quality=output_quality, optimize=True)
            else:
                image.save(output_path)
            output_paths.append(Path(output_path))

        return output_paths

    def generate_embeddings(
        self, prompt: str, redux: Path, prompt_strength: float, redux_strength: float, num_outputs: int
    ):
        txt_embedding = encode_prompt(self.pipe_prior_redux, prompt, num_outputs)
        img = Image.open(redux).convert("RGB")
        img_embedding = encode_image(self.pipe_prior_redux, img, num_outputs)

        prompt_condition = (0, txt_embedding.size(1), prompt_strength)
        img_condition = (txt_embedding.size(1), txt_embedding.size(1) + img_embedding.size(1), redux_strength)

        # Concatenate embeddings along dim=1
        prompt_embeds = torch.cat([txt_embedding, img_embedding], dim=1).to(DEVICE, dtype=WEIGHT_DTYPE)

        # For pooled_prompt_embeds, use get_clip_empty
        pooled_prompt_embeds = encode_clip_prompt(self.pipe_prior_redux, "", num_outputs)

        # Build joint_attention_kwargs
        joint_attention_kwargs = {"conditions": [prompt_condition, img_condition]}
        return prompt_embeds, pooled_prompt_embeds, joint_attention_kwargs

    def compile_flux(self):
        start = time.time()
        print("Compiling Flux Pipeline")
        prompt = "test compilation prompt"
        redux = Path("./girl.webp")
        prompt_embeds, pooled_prompt_embeds, joint_attention_kwargs = self.generate_embeddings(
            prompt, redux, 1.0, 0.05, 1
        )
        self.pipe(
            prompt_embeds=prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            joint_attention_kwargs=joint_attention_kwargs,
        )
        print("Compilation took: ", time.time() - start)


class SchnellPredictor(Predictor):
    MODEL_URL = SCHNELL_URL
    MODEL_CACHE = SCHNELL_CACHE


class DevPredictor(Predictor):
    MODEL_URL = DEV_URL
    MODEL_CACHE = DEV_CACHE


def encode_image(pipe_prior_redux, img, num_outputs):
    image_latents = pipe_prior_redux.encode_image(img, DEVICE, 1)
    image_embeds = pipe_prior_redux.image_embedder(image_latents).image_embeds
    return image_embeds


def encode_prompt(pipe_prior_redux, prompt, num_outputs, max_sequence_length=512):
    prompt_embeds = pipe_prior_redux._get_t5_prompt_embeds(prompt, 1, max_sequence_length, DEVICE)
    return prompt_embeds


def encode_clip_prompt(pipe_prior_redux, prompt, num_outputs):
    pooled_prompt_embeds = pipe_prior_redux._get_clip_prompt_embeds(prompt, 1, DEVICE)
    return pooled_prompt_embeds
