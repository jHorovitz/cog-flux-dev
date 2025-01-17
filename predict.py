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
from diffusers import FluxPriorReduxPipeline
from pipeline_flux_compile import FluxPipelineCompile
from pipeline_flux_control_compile import FluxControlPipelineCompile
from torchvision import transforms
from image_gen_aux import DepthPreprocessor

from condition_reweighting_attention_processor import ConditionReweightingAttentionProcessor

MODEL_CACHE_TOP_DIR = "./model-cache"  # necessary for tars that also contain a directory.
DEV_CACHE = "./model-cache/FLUX.1-dev"
SCHNELL_CACHE = "./model-cache/FLUX.1-schnell"
DEV_URL = "https://weights.replicate.delivery/default/black-forest-labs/FLUX.1-dev/files.tar"
SCHNELL_URL = "https://weights.replicate.delivery/default/black-forest-labs/FLUX.1-schnell/files.tar"
REDUX_CACHE = "./model-cache/FLUX.1-Redux"
REDUX_URL = "https://weights.replicate.delivery/default/black-forest-labs/FLUX.1-Redux-dev/model.tar"
DEPTH_PROCESSOR_CACHE = (
    "./model-cache/models--LiheYoung--depth-anything-large-hf/snapshots/27ccb0920352c0c37b3a96441873c8d37bd52fb6"
)
DEPTH_PROCESSOR_URL = "https://weights.replicate.delivery/default/redux-slider/LiheYoung/depth-anything-large-hf.tar"
DEPTH_LORA_CACHE = (
    "./model-cache/models--black-forest-labs--FLUX.1-Depth-dev-lora/snapshots/ee9cc283d790a079d549ac0bf9ef7183082e3d90"
)
DEPTH_LORA_URL = "https://weights.replicate.delivery/default/redux-slider/black-forest-labs/FLUX.1-Depth-dev-lora.tar"

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
        """
        Note: The reason that self.control_pipe and self.pipe are separate is that loading the
        control lora actually changes the weights of the model (and not just the lora). Specifically,
        pipe.transformer.x_embedder goes from being 64 to 128 in the input dimension.
        """
        """Load the model into memory to make running multiple predictions efficient"""
        start = time.time()

        print("Loading Flux Pipeline")
        if not os.path.exists(self.MODEL_CACHE):
            download_weights(self.MODEL_URL, MODEL_CACHE_TOP_DIR)

        self.pipe = FluxPipelineCompile.from_pretrained(
            self.MODEL_CACHE,
            torch_dtype=WEIGHT_DTYPE,
            local_files_only=True,
        ).to(DEVICE)
        self.pipe.transformer.set_attn_processor(ConditionReweightingAttentionProcessor())

        self.control_pipe = FluxControlPipelineCompile.from_pretrained(
            self.MODEL_CACHE,
            torch_dtype=WEIGHT_DTYPE,
            local_files_only=True,
        ).to(DEVICE)
        self.control_pipe.transformer.set_attn_processor(ConditionReweightingAttentionProcessor())

        if not os.path.exists(DEPTH_LORA_CACHE):
            download_weights(DEPTH_LORA_URL, MODEL_CACHE_TOP_DIR)
        self.control_pipe.load_lora_weights(
            DEPTH_LORA_CACHE,
            adapter_name="depth",
            torch_dtype=WEIGHT_DTYPE,
            local_files_only=True,
            subfolder="snapshots/ee9cc283d790a079d549ac0bf9ef7183082e3d90",
            weight_name="flux1-depth-dev-lora.safetensors",
        )

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

        if not os.path.exists(DEPTH_PROCESSOR_CACHE):
            download_weights(DEPTH_PROCESSOR_URL, MODEL_CACHE_TOP_DIR)
        self.depth_processor = DepthPreprocessor.from_pretrained(DEPTH_PROCESSOR_CACHE)

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
        control_image: Path = Input(
            description="Control image (optional).",
            default=None,
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

        if control_image is not None:
            control_image = Image.open(control_image)
            control_image = self.depth_processor(control_image)[0].convert("RGB")

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

        if control_image is None:
            output = self.pipe(**common_args, **flux_kwargs)
        else:
            output = self.control_pipe(**common_args, **flux_kwargs, control_image=control_image)

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
        self.pipe.disable_lora()
        prompt = "test compilation prompt"
        redux = Path("./girl.webp")
        control_image = Image.open(redux)
        prompt_embeds, pooled_prompt_embeds, joint_attention_kwargs = self.generate_embeddings(
            prompt=prompt,
            redux=redux,
            prompt_strength=1.0,
            redux_strength=0.05,
            num_outputs=1,
        )
        self.pipe(
            prompt_embeds=prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            joint_attention_kwargs=joint_attention_kwargs,
            height=256,
            width=256,
        )

        self.control_pipe(
            prompt_embeds=prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            joint_attention_kwargs=joint_attention_kwargs,
            control_image=control_image,
            height=256,
            width=256,
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
