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
    FluxPipeline,
    FluxPriorReduxPipeline,
)
from torchvision import transforms
from transformers import CLIPImageProcessor
from diffusers.pipelines.stable_diffusion.safety_checker import StableDiffusionSafetyChecker

from condition import inputs_to_conditions
from condition_reweighting_attention_processor import ConditionReweightingAttentionProcessor
import monkey_patch

MAX_IMAGE_SIZE = 1440
DEV_CACHE = "./FLUX.1-dev"
SCHNELL_CACHE = "./FLUX.1-schnell"
DEV_URL = "https://weights.replicate.delivery/default/black-forest-labs/FLUX.1-dev/files.tar"
SCHNELL_URL = "https://weights.replicate.delivery/default/black-forest-labs/FLUX.1-schnell/files.tar"
REDUX_CACHE = "./FLUX.1-Redux"
REDUX_URL = "https://weights.replicate.delivery/default/black-forest-labs/FLUX.1-Redux-dev/model.tar"

ASPECT_RATIOS = {
    "1:1_small": (512, 512),
    "1:1": (1024, 1024),
    "16:9": (1344, 768),
    "21:9": (1536, 640),
    "3:2": (1216, 832),
    "2:3": (832, 1216),
    "4:5": (896, 1088),
    "5:4": (1088, 896),
    "3:4": (896, 1152),
    "4:3": (1152, 896),
    "9:16": (768, 1344),
    "9:21": (640, 1536),
}


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
            print(">>> flux weights do not exist")
            download_weights(self.MODEL_URL, ".")
        else:
            print(">>> flux weights exist: os.listdir(self.MODEL_CACHE)")

        self.pipe = FluxPipeline.from_pretrained(
            self.MODEL_CACHE,
            torch_dtype=WEIGHT_DTYPE,
            local_files_only=True,
        ).to("cuda")
        monkey_patch.monkey_patch_pipeline(self.pipe)
        self.pipe.transformer.set_attn_processor(ConditionReweightingAttentionProcessor())

        if not os.path.exists(REDUX_CACHE):
            print(">>> redux weights do not exist")
            download_weights(REDUX_URL, REDUX_CACHE)
        else:
            print(">>> redux weights exist: os.listdir(REDUX_CACHE)")
        self.pipe_prior_redux = FluxPriorReduxPipeline.from_pretrained(
            REDUX_CACHE,
            tokenizer=self.pipe.tokenizer,
            text_encoder=self.pipe.text_encoder,
            tokenizer_2=self.pipe.tokenizer_2,
            text_encoder_2=self.pipe.text_encoder_2,
            local_files_only=True,
        ).to(DEVICE)
        print("setup took: ", time.time() - start)

    def aspect_ratio_to_width_height(self, aspect_ratio: str) -> tuple[int, int]:
        return ASPECT_RATIOS[aspect_ratio]

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
        prompt: str = Input(description="First prompt for generated image", default=None),
        prompt_2: str = Input(description="Second prompt for generated image", default=None),
        redux: Path = Input(
            description="First Redux image.",
            default=None,
        ),
        redux_2: Path = Input(
            description="Second Redux image.",
            default=None,
        ),
        aspect_ratio: str = Input(
            description="Aspect ratio for the generated image", choices=list(ASPECT_RATIOS.keys()), default="1:1"
        ),
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
        prompt_single_strengths: str | list[float] = Input(description="Attention reweighting.", default=[1.0] * 38),
        prompt_double_strengths: str | list[float] = Input(description="Attention reweighting.", default=[1.0] * 19),
        prompt_2_single_strengths: str | list[float] = Input(description="Attention reweighting.", default=[1.0] * 38),
        prompt_2_double_strengths: str | list[float] = Input(description="Attention reweighting.", default=[1.0] * 19),
        redux_single_strengths: str | list[float] = Input(description="Attention reweighting.", default=[1.0] * 38),
        redux_double_strengths: str | list[float] = Input(description="Attention reweighting.", default=[1.0] * 19),
        redux_2_single_strengths: str | list[float] = Input(description="Attention reweighting.", default=[1.0] * 38),
        redux_2_double_strengths: str | list[float] = Input(description="Attention reweighting.", default=[1.0] * 19),
    ) -> List[Path]:
        """Run a single prediction on the model"""
        if seed is None:
            seed = int.from_bytes(os.urandom(2), "big")
        print(f"Using seed: {seed}")

        generator = torch.Generator("cuda").manual_seed(seed)

        prompt_embeds, pooled_prompt_embeds, joint_attention_kwargs = self.generate_embeddings(
            prompt,
            prompt_2,
            redux,
            redux_2,
            prompt_single_strengths,
            prompt_double_strengths,
            prompt_2_single_strengths,
            prompt_2_double_strengths,
            redux_single_strengths,
            redux_double_strengths,
            redux_2_single_strengths,
            redux_2_double_strengths,
        )

        width, height = self.aspect_ratio_to_width_height(aspect_ratio)
        max_sequence_length = 512

        flux_kwargs = {
            "width": width,
            "height": height,
            "prompt_embeds": prompt_embeds,
            "pooled_prompt_embeds": pooled_prompt_embeds,
            "joint_attention_kwargs": joint_attention_kwargs,
        }
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
        self,
        prompt,
        prompt_2,
        redux,
        redux_2,
        prompt_single_strengths,
        prompt_double_strengths,
        prompt_2_single_strengths,
        prompt_2_double_strengths,
        redux_single_strengths,
        redux_double_strengths,
        redux_2_single_strengths,
        redux_2_double_strengths,
    ):
        conditions = inputs_to_conditions(
            prompt,
            prompt_2,
            redux,
            redux_2,
            prompt_single_strengths,
            prompt_double_strengths,
            prompt_2_single_strengths,
            prompt_2_double_strengths,
            redux_single_strengths,
            redux_double_strengths,
            redux_2_single_strengths,
            redux_2_double_strengths,
        )

        embeddings_list = []

        current_index = 0
        for condition in conditions:
            if condition.img is not None:
                img = Image.open(condition.img).convert("RGB")
                embedding = encode_image(self.pipe_prior_redux, img)
            elif condition.txt is not None:
                embedding = encode_prompt(self.pipe_prior_redux, condition.txt)
            else:
                raise ValueError(f"txt and img both not set: {condition}")

            embeddings_list.append(embedding)

            length = embedding.size(1)
            condition.start_index = current_index
            current_index = current_index + length + 1
            condition.end_index = current_index

        # Concatenate embeddings along dim=1
        prompt_embeds = torch.cat(embeddings_list, dim=1).to(DEVICE, dtype=WEIGHT_DTYPE)

        # For pooled_prompt_embeds, use get_clip_empty
        pooled_prompt_embeds = encode_clip_prompt(self.pipe_prior_redux, "")

        # Build joint_attention_kwargs
        joint_attention_kwargs = {"condition_reweightings": conditions}
        return prompt_embeds, pooled_prompt_embeds, joint_attention_kwargs


def encode_image(pipe_prior_redux, img):
    image_latents = pipe_prior_redux.encode_image(img, DEVICE, 1)
    image_embeds = pipe_prior_redux.image_embedder(image_latents).image_embeds
    return image_embeds


def encode_prompt(pipe_prior_redux, prompt, max_sequence_length=512):
    prompt_embeds = pipe_prior_redux._get_t5_prompt_embeds(prompt, 1, max_sequence_length, DEVICE)
    return prompt_embeds


def encode_clip_prompt(pipe_prior_redux, prompt):
    pooled_prompt_embeds = pipe_prior_redux._get_clip_prompt_embeds(prompt, 1, DEVICE)
    return pooled_prompt_embeds


class SchnellPredictor(Predictor):
    MODEL_URL = SCHNELL_URL
    MODEL_CACHE = SCHNELL_CACHE


class DevPredictor(Predictor):
    MODEL_URL = DEV_URL
    MODEL_CACHE = DEV_CACHE
