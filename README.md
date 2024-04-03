---
title: StreamingT2V
emoji: 🔥
colorFrom: blue
colorTo: purple
sdk: gradio
sdk_version: 4.25.0
app_file: app.py
pinned: false
short_description: Consistent, Dynamic, and Extendable Long Video Generation fr
---



# StreamingT2V

This repository is the official implementation of [StreamingT2V](https://streamingt2v.github.io/).


**[StreamingT2V: Consistent, Dynamic, and Extendable Long Video Generation from Text](https://arxiv.org/abs/2403.14773)**
</br>
Roberto Henschel,
Levon Khachatryan,
Daniil Hayrapetyan,
Hayk Poghosyan,
Vahram Tadevosyan,
Zhangyang Wang, Shant Navasardyan, Humphrey Shi
</br>

[arXiv preprint](https://arxiv.org/abs/2403.14773) | [Video](https://twitter.com/i/status/1770909673463390414) | [Project page](https://streamingt2v.github.io/)


<p align="center">
<img src="__assets__/github/teaser/teaser_final.png" width="800px"/>  
<br>
<br>
<em>StreamingT2V is an advanced autoregressive technique that enables the creation of long videos featuring rich motion dynamics without any stagnation. It ensures temporal consistency throughout the video, aligns closely with the descriptive text, and maintains high frame-level image quality. Our demonstrations include successful examples of videos up to 1200 frames, spanning 2 minutes, and can be extended for even longer durations. Importantly, the effectiveness of StreamingT2V is not limited by the specific Text2Video model used, indicating that improvements in base models could yield even higher-quality videos.</em>
</p>

## News

* [03/21/2024] Paper [StreamingT2V](https://arxiv.org/abs/2403.14773) released!
* [04/03/2024] Code and [model](https://huggingface.co/PAIR/StreamingT2V) released!


## Setup



1. Clone this repository and enter:

``` shell
git clone https://github.com/Picsart-AI-Research/StreamingT2V.git
cd StreamingT2V/
```
2. Install requirements using Python 3.10 and CUDA >= 11.6
``` shell
conda create -n st2v python=3.10
conda activate st2v
pip install -r requirements.txt
```
3. (Optional) Install FFmpeg if it's missing on your system
``` shell
conda install conda-forge::ffmpeg
```
4. Download the weights from [HF](https://huggingface.co/PAIR/StreamingT2V) and put them into the `t2v_enhanced/checkpoints` directory.

---  


## Inference



### For Text-to-Video

``` shell
cd StreamingT2V/
python inference.py --prompt="A cat running on the street"
```
To use other base models add the `--base_model=AnimateDiff` argument. Use `python inference.py --help` for more options.

### For Image-to-Video

``` shell
cd StreamingT2V/
python inference.py --image=../examples/underwater.png --base_model=SVD
```



## Results
Detailed results can be found in the [Project page](https://streamingt2v.github.io/).

## License
Our code is published under the CreativeML Open RAIL-M license.

We include [ModelscopeT2V](https://github.com/modelscope/modelscope), [AnimateDiff](https://github.com/guoyww/AnimateDiff), [DynamiCrafter](https://github.com/Doubiiu/DynamiCrafter) in the demo for research purposes and to demonstrate the flexibility of the StreamingT2V framework to include different T2V/I2V models. For commercial usage of such components, please refer to their original license.




## BibTeX
If you use our work in your research, please cite our publication:
```
@article{henschel2024streamingt2v,
  title={StreamingT2V: Consistent, Dynamic, and Extendable Long Video Generation from Text},
  author={Henschel, Roberto and Khachatryan, Levon and Hayrapetyan, Daniil and Poghosyan, Hayk and Tadevosyan, Vahram and Wang, Zhangyang and Navasardyan, Shant and Shi, Humphrey},
  journal={arXiv preprint arXiv:2403.14773},
  year={2024}
}
```


