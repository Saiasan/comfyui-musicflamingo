# ComfyUI Custom Node for Music Flamingo

**Custom node for [ComfyUI](https://github.com/comfyanonymous/ComfyUI) that allows you to analyze songs using NVIDIA's [Music Flamingo](https://musicflamingo.github.io/).**

![Screenshot](/example_workflow/screenshot.png?raw=true)


## Installation

1. Clone this repo into `custom_modules`:
    ```
    cd ComfyUI/custom_nodes
    git clone https://github.com/C0untFloyd/comfyui-musicflamingo
    ```
2. Make sure your installed transformers version is at least 5.0.0

The first run will download the necessary models from [NVIDIA](https://huggingface.co/nvidia/music-flamingo-hf/tree/main) to your comfymodels/checkpoints folder.
Depending on your internet connection this might take a while, they are about 16 Gb in size.
