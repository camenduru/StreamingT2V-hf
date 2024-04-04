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
from t2v_enhanced.inference_utils import *
from t2v_enhanced.model_init import *
from t2v_enhanced.model_func import *


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--prompt', type=str, default="A cat running on the street", help="The prompt to guide video generation.")
    parser.add_argument('--image', type=str, default="", help="Path to image conditioning.")
    # parser.add_argument('--video', type=str, default="", help="Path to video conditioning.")
    parser.add_argument('--base_model', type=str, default="ModelscopeT2V", help="Base model to generate first chunk from", choices=["ModelscopeT2V", "AnimateDiff", "SVD"])
    parser.add_argument('--num_frames', type=int, default=24, help="The number of video frames to generate.")
    parser.add_argument('--negative_prompt', type=str, default="", help="The prompt to guide what to not include in video generation.")
    parser.add_argument('--num_steps', type=int, default=50, help="The number of denoising steps.")
    parser.add_argument('--image_guidance', type=float, default=9.0, help="The guidance scale.")

    parser.add_argument('--output_dir', type=str, default="results", help="Path where to save the generated videos.")
    parser.add_argument('--device', type=str, default="cuda")
    parser.add_argument('--seed', type=int, default=33, help="Random seed")
    args = parser.parse_args()


    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    result_fol = Path(args.output_dir).absolute()
    device = args.device


    # --------------------------
    # ----- Configurations -----
    # --------------------------
    ckpt_file_streaming_t2v = Path("checkpoints/streaming_t2v.ckpt").absolute()
    cfg_v2v = {'downscale': 1, 'upscale_size': (1280, 720), 'model_id': 'damo/Video-to-Video', 'pad': True}


    # --------------------------
    # ----- Initialization -----
    # --------------------------
    stream_cli, stream_model = init_streamingt2v_model(ckpt_file_streaming_t2v, result_fol)
    if args.base_model == "ModelscopeT2V":
        model = init_modelscope(device)
    elif args.base_model == "AnimateDiff":
        model = init_animatediff(device)
    elif args.base_model == "SVD":
        model = init_svd(device)
        sdxl_model = init_sdxl(device)
    

    inference_generator = torch.Generator(device="cuda")


    # ------------------
    # ----- Inputs -----
    # ------------------
    now = datetime.datetime.now()
    name = args.prompt[:100].replace(" ", "_") + "_" + str(now.time()).replace(":", "_").replace(".", "_")

    inference_generator = torch.Generator(device="cuda")
    inference_generator.manual_seed(args.seed)
    
    if args.base_model == "ModelscopeT2V":
        short_video = ms_short_gen(args.prompt, model, inference_generator)
    elif args.base_model == "AnimateDiff":
        short_video = ad_short_gen(args.prompt, model, inference_generator)
    elif args.base_model == "SVD":
        short_video = svd_short_gen(args.image, args.prompt, model, sdxl_model, inference_generator)

    n_autoreg_gen = args.num_frames // 8 - 8
    stream_long_gen(args.prompt, short_video, n_autoreg_gen, args.negative_prompt, args.seed, args.num_steps, args.image_guidance, name, stream_cli, stream_model)
    video2video(args.prompt, opj(result_fol, name+".mp4"), result_fol, cfg_v2v, msxl_model)
