from predict import SchnellPredictor, DevPredictor


def schnell(height=512, width=512):
    predictor = SchnellPredictor()
    predictor.setup()
    predictor.predict(
        prompt="An anime style painting of a face.",
        redux="./sample_images/girl.webp",
        control_image=None,
        height=height,
        width=width,
        num_inference_steps=4,
        guidance_scale=3.5,
        num_outputs=2,
        seed=42,
        output_format="webp",
        output_quality=100,
        redux_strength=0.05,
    )


def dev(height=512, width=512):
    predictor = DevPredictor()
    predictor.setup()
    predictor.predict(
        prompt="An anime style painting of a face.",
        redux="./sample_images/girl.webp",
        control_image=None,
        height=height,
        width=width,
        num_inference_steps=28,
        guidance_scale=3.5,
        num_outputs=2,
        seed=42,
        output_format="webp",
        output_quality=100,
        redux_strength=0.05,
    )


def dev_depth(height=512, width=512):
    predictor = DevPredictor()
    predictor.setup()
    predictor.predict(
        prompt="A woman riding a skateboard.",
        redux="./sample_images/girl.webp",
        control_image="./sample_images/skateboard.webp",
        height=height,
        width=width,
        num_inference_steps=28,
        guidance_scale=3.5,
        num_outputs=2,
        seed=42,
        output_format="webp",
        output_quality=100,
        redux_strength=0.05,
    )


def dev_multi(height=512, width=512):
    predictor = DevPredictor()
    predictor.setup()
    predictor.predict(
        prompt="An anime style painting of a face.",
        redux="./sample_images/girl.webp",
        control_image=None,
        height=height,
        width=width,
        num_inference_steps=28,
        guidance_scale=3.5,
        num_outputs=2,
        seed=42,
        output_format="webp",
        output_quality=100,
        redux_strength=0.05,
    )
    predictor.predict(
        prompt="An anime style painting of a face.",
        redux="./sample_images/girl.webp",
        height=height * 2,
        width=width * 2,
        num_inference_steps=28,
        guidance_scale=3.5,
        num_outputs=1,
        seed=42,
        output_format="webp",
        output_quality=100,
        redux_strength=0.05,
    )


if __name__ == "__main__":
    dev()
    dev_depth()
