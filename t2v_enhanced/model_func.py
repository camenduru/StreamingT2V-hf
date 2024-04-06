# General
import os
from os.path import join as opj
import datetime
import torch
from einops import rearrange, repeat

# Utilities
from t2v_enhanced.inference_utils import *

from modelscope.outputs import OutputKeys
import imageio
from PIL import Image
import numpy as np

import torch.nn.functional as F
import torchvision.transforms as transforms
from diffusers.utils import load_image
transform = transforms.Compose([ 
    transforms.PILToTensor() 
])


def ms_short_gen(prompt, ms_model, inference_generator, t=50, device="cuda"):
    frames = ms_model(prompt, 
                    num_inference_steps=t, 
                    generator=inference_generator, 
                    eta=1.0, 
                    height=256, 
                    width=256, 
                    latents=None).frames
    frames = torch.stack([torch.from_numpy(frame) for frame in frames])
    frames = frames.to(device).to(torch.float32)
    return rearrange(frames[0], "F W H C -> F C W H")

def ad_short_gen(prompt, ad_model, inference_generator, t=25, device="cuda"):
    frames = ad_model(prompt, 
                    negative_prompt="bad quality, worse quality",
                    num_frames=16,
                    num_inference_steps=t, 
                    generator=inference_generator, 
                    guidance_scale=7.5).frames[0]
    frames = torch.stack([transform(frame) for frame in frames])
    frames = frames.to(device).to(torch.float32)
    frames = F.interpolate(frames, size=256)
    frames = frames/255.0
    return frames

def sdxl_image_gen(prompt, sdxl_model):
    image = sdxl_model(prompt=prompt).images[0]
    return image

def svd_short_gen(image, prompt, svd_model, sdxl_model, inference_generator, t=25, device="cuda"):
    if image is None:
        image = sdxl_image_gen(prompt, sdxl_model)
        image = image.resize((576, 576))
        image = add_margin(image, 0, 224, 0, 224, (0, 0, 0))
    elif type(image) is str:
        image = load_image(image)
        image = resize_and_keep(image)
        image = center_crop(image)
        image = add_margin(image, 0, 224, 0, 224, (0, 0, 0))
    else:
        image = Image.fromarray(np.uint8(image))
        image = resize_and_keep(image)
        image = center_crop(image)
        image = add_margin(image, 0, 224, 0, 224, (0, 0, 0))

    frames = svd_model(image, decode_chunk_size=4, generator=inference_generator).frames[0]
    frames = torch.stack([transform(frame) for frame in frames])
    frames = frames.to(device).to(torch.float32)
    frames = frames[:16,:,:,224:-224]
    frames = F.interpolate(frames, size=256)
    frames = frames/255.0
    return frames


def stream_long_gen(prompt, short_video, n_autoreg_gen, seed, t, image_guidance, result_file_stem, stream_cli, stream_model):
    trainer = stream_cli.trainer
    trainer.limit_predict_batches = 1

    trainer.predict_cfg = {
        "predict_dir": stream_cli.config["result_fol"].as_posix(),
        "result_file_stem": result_file_stem,
        "prompt": prompt,
        "video": short_video,
        "seed": seed,
        "num_inference_steps": t,
        "guidance_scale": image_guidance,
        'n_autoregressive_generations': n_autoreg_gen,
    }

    trainer.predict(model=stream_model, datamodule=stream_cli.datamodule)


def video2video(prompt, video, where_to_log, cfg_v2v, model_v2v, square=True):
    downscale = cfg_v2v['downscale']
    upscale_size = cfg_v2v['upscale_size']
    pad = cfg_v2v['pad']

    now = datetime.datetime.now()
    now = str(now.time()).replace(":", "_").replace(".", "_")
    name = prompt[:100].replace(" ", "_") + "_" + now
    enhanced_video_mp4 = opj(where_to_log, name+"_enhanced.mp4")

    video_frames = imageio.mimread(video)
    h, w, _ = video_frames[0].shape

    # Downscale video, then resize to fit the upscale size
    video = [Image.fromarray(frame).resize((w//downscale, h//downscale)) for frame in video_frames]
    video = [resize_to_fit(frame, upscale_size) for frame in video]

    if pad:
        video = [pad_to_fit(frame, upscale_size) for frame in video]
    # video = [np.array(frame) for frame in video]

    imageio.mimsave(opj(where_to_log, 'temp_'+now+'.mp4'), video, fps=8)

    p_input = {
        'video_path': opj(where_to_log, 'temp_'+now+'.mp4'),
        'text': prompt
    }
    output_video_path = model_v2v(p_input, output_video=enhanced_video_mp4)[OutputKeys.OUTPUT_VIDEO]

    # Remove padding
    video_frames = imageio.mimread(enhanced_video_mp4)
    video_frames_square = []
    for frame in video_frames:
        frame = frame[:, 280:-280, :]
        video_frames_square.append(frame)
    imageio.mimsave(enhanced_video_mp4, video_frames_square)

    return enhanced_video_mp4
