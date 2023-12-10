
import gradio as gr

from app_text_to_video import create_demo_text_to_video
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--public_access', action='store_true',
                    help="if enabled, the app can be access from a public url", default=False)
args = parser.parse_args()



with gr.Blocks(css='style.css') as demo:

    with gr.Tab('Text2Video'):
        create_demo_text_to_video()


print("Generating Gradio app LINK:")
_, _, link = demo.queue(api_open=False).launch(share=args.public_access)
print(link)
