import gradio as gr
import imageio
import torch

from pipeline import Pipeline
import gpt


pipe = Pipeline.from_pretrained("runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16).to("cuda")

def process_text2video( input_prompt,
                        video_length,
                        fps,
                        seed,
                        motion_field_strength_x,
                        motion_field_strength_y,
                        t0,
                        t1):
    gptPrompt = gpt.get_GPTprompt(input_prompt,video_length)
    frame = pipe(   prompt=gptPrompt, 
                    generator=torch.Generator('cuda').manual_seed(seed),
                    video_length = video_length,
                    motion_field_strength_x = motion_field_strength_x,
                    motion_field_strength_y = motion_field_strength_y,
                    t0 = t0,
                    t1 = t1,
                ).images
    result = [(r * 255).astype("uint8") for r in frame]
    imageio.mimsave("output_video.mp4", result, fps=fps)
    return "output_video.mp4"

def create_demo_text_to_video():
    with gr.Blocks() as demo:
        with gr.Row():
            gr.Markdown('Text2Video: Video Generation')
        with gr.Row():
            gr.HTML(
                """
                <div style="text-align: left; auto;">
                <h2 style="font-weight: 450; font-size: 1rem; margin: 0rem">
                </h3>
                </div>
                """)

        with gr.Row():
            with gr.Column():
                model_name = gr.Textbox(
                    label="Model",
                    value="runwayml/stable-diffusion-v1-5",
                )
                input_prompt = gr.Textbox(label='Prompt')
                video_length = gr.Number(
                        label="Video length = n_gpt_prompt", value=4, precision=0)
                fps = gr.Number(
                        label="FPS", value=4, precision=0)
                seed = gr.Slider(label='Seed',
                                info="-1 for random seed on each run. Otherwise, the seed will be fixed.",
                                minimum=-1,
                                maximum=65536,
                                value=10,
                                step=1)
                motion_field_strength_x = gr.Slider(
                    label='Global Translation $\\delta_{x}$', minimum=-20, maximum=20,
                    value=2,
                    step=1)
                motion_field_strength_y = gr.Slider(
                    label='Global Translation $\\delta_{y}$', minimum=-20, maximum=20,
                    value=2,
                    step=1)
                t0 = gr.Slider(label="Timestep t0", minimum=0,
                                maximum=98, value=44, step=1,
                                info="Perform DDPM steps from t0 to t1. The larger the gap between t0 and t1, the more variance between the frames. Ensure t0 < t1 ",
                                )
                t1 = gr.Slider(label="Timestep t1", minimum=1,
                                info="Perform DDPM steps from t0 to t1. The larger the gap between t0 and t1, the more variance between the frames. Ensure t0 < t1",
                                maximum=99, value=47, step=1)
                run_button = gr.Button('Run')
            with gr.Column():
                result = gr.Video(label="Generated Video")

        inputs = [
            input_prompt,
            video_length,
            fps,
            seed,
            motion_field_strength_x,
            motion_field_strength_y,
            t0,
            t1,
        ]

        run_button.click(fn=process_text2video,
                         inputs=inputs,
                         outputs=result,)
    return demo
