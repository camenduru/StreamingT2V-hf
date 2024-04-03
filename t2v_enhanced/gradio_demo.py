# General
import os
from os.path import join as opj
import argparse
import datetime
from pathlib import Path
import torch
import gradio as gr
import tempfile
import yaml
from t2v_enhanced.model.video_ldm import VideoLDM

# Utilities
from inference_utils import *
from model_init import *
from model_func import *


on_huggingspace = os.environ.get("SPACE_AUTHOR_NAME") == "PAIR"
parser = argparse.ArgumentParser()
parser.add_argument('--public_access', action='store_true', default=True)
parser.add_argument('--where_to_log', type=str, default="gradio_output")
parser.add_argument('--device', type=str, default="cuda")
args = parser.parse_args()


Path(args.where_to_log).mkdir(parents=True, exist_ok=True)
result_fol = Path(args.where_to_log).absolute()
device = args.device


# --------------------------
# ----- Configurations -----
# --------------------------
ckpt_file_streaming_t2v = Path("checkpoints/streaming_t2v.ckpt").absolute()
cfg_v2v = {'downscale': 1, 'upscale_size': (1280, 720), 'model_id': 'damo/Video-to-Video', 'pad': True}


# --------------------------
# ----- Initialization -----
# --------------------------
ms_model = init_modelscope(device)
# zs_model = init_zeroscope(device)
stream_cli, stream_model = init_streamingt2v_model(ckpt_file_streaming_t2v, result_fol)
msxl_model = init_v2v_model(cfg_v2v)

inference_generator = torch.Generator(device="cuda")


# -------------------------
# ----- Functionality -----
# -------------------------
def generate(prompt, num_frames, image, model_name_stage1, model_name_stage2, n_prompt, seed, t, image_guidance, where_to_log=result_fol):
    now = datetime.datetime.now()
    name = prompt[:100].replace(" ", "_") + "_" + str(now.time()).replace(":", "_").replace(".", "_")

    if num_frames == [] or num_frames is None:
        num_frames = 56
    else:
        num_frames = int(num_frames.split(" ")[0])

    n_autoreg_gen = num_frames/8-8

    inference_generator.manual_seed(seed)
    short_video = ms_short_gen(prompt, ms_model, inference_generator, t, device)
    stream_long_gen(prompt, short_video, n_autoreg_gen, n_prompt, seed, t, image_guidance, name, stream_cli, stream_model)
    video_path = opj(where_to_log, name+".mp4")
    return video_path

def enhance(prompt, input_to_enhance):
    encoded_video = video2video(prompt, input_to_enhance, result_fol, cfg_v2v, msxl_model)
    return encoded_video


# --------------------------
# ----- Gradio-Demo UI -----
# --------------------------
with gr.Blocks() as demo:
    gr.HTML(
        """
        <div style="text-align: center; max-width: 1200px; margin: 20px auto;">
        <h1 style="font-weight: 900; font-size: 3rem; margin: 0rem">
            <a href="https://github.com/Picsart-AI-Research/StreamingT2V" style="color:blue;">StreamingT2V</a> 
        </h1>
        <h2 style="font-weight: 450; font-size: 1rem; margin: 0rem">
        Roberto Henschel<sup>1*</sup>, Levon Khachatryan<sup>1*</sup>, Daniil Hayrapetyan<sup>1*</sup>, Hayk Poghosyan<sup>1</sup>, Vahram Tadevosyan<sup>1</sup>, Zhangyang Wang<sup>1,2</sup>, Shant Navasardyan<sup>1</sup>, Humphrey Shi<sup>1,3</sup>
        </h2>
        <h2 style="font-weight: 450; font-size: 1rem; margin: 0rem">
        <sup>1</sup>Picsart AI Resarch (PAIR), <sup>2</sup>UT Austin, <sup>3</sup>SHI Labs @ Georgia Tech, Oregon & UIUC
        </h2>
        <h2 style="font-weight: 450; font-size: 1rem; margin: 0rem">
        *Equal Contribution
        </h2>
        <h2 style="font-weight: 450; font-size: 1rem; margin: 0rem">
        [<a href="https://arxiv.org/abs/2403.14773" style="color:blue;">arXiv</a>] 
        [<a href="https://github.com/Picsart-AI-Research/StreamingT2V" style="color:blue;">GitHub</a>]
        </h2>
        <h2 style="font-weight: 450; font-size: 1rem; margin-top: 0.5rem; margin-bottom: 0.5rem">
        <b>StreamingT2V</b> is an advanced autoregressive technique that enables the creation of long videos featuring rich motion dynamics without any stagnation. 
        It ensures temporal consistency throughout the video, aligns closely with the descriptive text, and maintains high frame-level image quality. 
        Our demonstrations include successful examples of videos up to <b>1200 frames, spanning 2 minutes</b>, and can be extended for even longer durations. 
        Importantly, the effectiveness of StreamingT2V is not limited by the specific Text2Video model used, indicating that improvements in base models could yield even higher-quality videos.
        </h2>
        </div>
        """)

    if on_huggingspace:
        gr.HTML("""
        <p>For faster inference without waiting in queue, you may duplicate the space and upgrade to GPU in settings.
        <br/>
        <a href="https://huggingface.co/spaces/PAIR/StreamingT2V?duplicate=true">
        <img style="margin-top: 0em; margin-bottom: 0em" src="https://bit.ly/3gLdBN6" alt="Duplicate Space"></a>
        </p>""")

    with gr.Row():
        with gr.Column():
            with gr.Row():
                with gr.Column():
                    with gr.Row():
                        num_frames = gr.Dropdown(["24", "32", "40", "48", "56", "80 - only on local", "240 - only on local", "600 - only on local", "1200 - only on local", "10000 - only on local"], label="Number of Video Frames: Default is 56", info="For >80 frames use local workstation!")
                    with gr.Row():
                        prompt_stage1 = gr.Textbox(label='Textual Prompt', placeholder="Ex: Dog running on the street.")
                    with gr.Row():
                        image_stage1 = gr.Image(label='Image Prompt (only required for I2V base models)', show_label=True, scale=1, show_download_button=True)
                with gr.Column():
                    video_stage1 = gr.Video(label='Long Video Preview', show_label=True, interactive=False, scale=2, show_download_button=True)
            with gr.Row():
                run_button_stage1 = gr.Button("Long Video Preview Generation")

            with gr.Row():
                with gr.Column():
                    with gr.Accordion('Advanced options', open=False):
                        model_name_stage1 = gr.Dropdown(
                            choices=["T2V: ModelScope", "T2V: ZeroScope", "I2V: AnimateDiff"],
                            label="Base Model. Default is ModelScope",
                            info="Currently supports only ModelScope. We will add more options later!",
                        )
                        model_name_stage2 = gr.Dropdown(
                            choices=["ModelScope-XL", "Another", "Another"],
                            label="Enhancement Model. Default is ModelScope-XL",
                            info="Currently supports only ModelScope-XL. We will add more options later!",
                        )
                        n_prompt = gr.Textbox(label="Optional Negative Prompt", value='')
                        seed = gr.Slider(label='Seed', minimum=0, maximum=65536, value=33,step=1,)

                        t = gr.Slider(label="Timesteps", minimum=0, maximum=100, value=50, step=1,)
                        image_guidance = gr.Slider(label='Image guidance scale', minimum=1, maximum=10, value=9.0, step=1.0)

        with gr.Column():
            with gr.Row():
                video_stage2 = gr.Video(label='Enhanced Long Video', show_label=True, interactive=False, height=473, show_download_button=True)
            with gr.Row():
                run_button_stage2 = gr.Button("Long Video Enhancement")
    '''
    '''
    gr.HTML(
        """
        <div style="text-align: justify; max-width: 1200px; margin: 20px auto;">
        <h3 style="font-weight: 450; font-size: 0.8rem; margin: 0rem">
        <b>Version: v1.0</b>
        </h3>
        <h3 style="font-weight: 450; font-size: 0.8rem; margin: 0rem">
        <b>Caution</b>: 
        We would like the raise the awareness of users of this demo of its potential issues and concerns.
        Like previous large foundation models, StreamingT2V could be problematic in some cases, partially we use pretrained ModelScope, therefore StreamingT2V can Inherit Its Imperfections.
        So far, we keep all features available for research testing both to show the great potential of the StreamingT2V framework and to collect important feedback to improve the model in the future.
        We welcome researchers and users to report issues with the HuggingFace community discussion feature or email the authors.
        </h3>
        <h3 style="font-weight: 450; font-size: 0.8rem; margin: 0rem">
        <b>Biases and content acknowledgement</b>:
        Beware that StreamingT2V may output content that reinforces or exacerbates societal biases, as well as realistic faces, pornography, and violence. 
        StreamingT2V in this demo is meant only for research purposes.
        </h3>
        </div>
        """)

    inputs_t2v = [prompt_stage1, num_frames, image_stage1, model_name_stage1, model_name_stage2, n_prompt, seed, t, image_guidance]
    run_button_stage1.click(fn=generate, inputs=inputs_t2v, outputs=video_stage1,)

    inputs_v2v = [prompt_stage1, video_stage1]
    run_button_stage2.click(fn=enhance, inputs=inputs_v2v, outputs=video_stage2,)


if on_huggingspace:
    demo.queue(max_size=20)
    demo.launch(debug=True)
else:
    _, _, link = demo.queue(api_open=False).launch(share=args.public_access)
    print(link)