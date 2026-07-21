# Anaconda Environment Setup
Create and activate an Anaconda environment using
```bash
conda create -n SDXL python=3.11
conda activate SDXL
```
Install PyTorch using:
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
```
Newer CUDA versions are available on the PyTorch [website](https://pytorch.org/get-started/locally/). Note that a GPU is a must for reasonable inference time. Inference using SDXL requires around **15 GB of VRAM**. You can also use Google Colab.

Install other packages:
```bash
pip install -U accelerate diffusers transformers
pip install pandas
```

# Inference
To generate 300 images using 300 prompts, run:
```bash
bash run.sh
```
The images will be saved in ```./inf/```. The prompts are written in ```metadata.csv```. The fine-tuned UNet for SDXL [kumo24/sdxl_nuclear](https://huggingface.co/kumo24/sdxl_nuclear) is available on HuggingFace.

# Note on Reproducibility
To get a fully reproducible pipeline, you should use deterministic algorithms as explained [here](https://huggingface.co/docs/diffusers/main/en/using-diffusers/reproducibility). However, deterministic algorithms may be slower than nondeterministic ones, and you may observe a decrease in performance. Therefore, you should expect different images on different GPU hardware or different pipeline initializations.
