from predict import SchnellPredictor, DevPredictor


def main():
    # predictor = DevPredictor()
    predictor = SchnellPredictor()
    predictor.setup()
    predictor.predict(
        prompt="An anime style painting of a face.",
        prompt_2=None,
        redux="./sample_images/girl.webp",
        redux_2=None,
        aspect_ratio="1:1_small",
        num_inference_steps=4,
        # num_inference_steps=28,
        guidance_scale=3.5,
        num_outputs=1,
        seed=42,
        output_format="webp",
        output_quality=100,
        disable_safety_checker=True,
        redux_single_strengths=[0.05] * 38,
        redux_double_strengths=[0.05] * 19,
        prompt_single_strengths=[0.95] * 38,
        prompt_double_strengths=[0.95] * 19,
    )


if __name__ == "__main__":
    main()
