from diffusers import DiffusionPipeline, UNet2DConditionModel
import torch
from pathlib import Path
import argparse
import pandas as pd
from PIL import Image

parser = argparse.ArgumentParser()
parser.add_argument('--unet_path', type=str, help="trained unet. Provide this if --unet is provided")
parser.add_argument('--prompt_path', type=str, required=True, help="Prompt to generate the image")
parser.add_argument('--seed', type=int, default=42, help="seed to control images selection")
parser.add_argument('--n_inf_steps', type=int, default=50, help="Number of inference steps (int), default = 50")
parser.add_argument('--inference_path', type=str, required=True, help="path to save images generated")
parser.add_argument('--device', type=str, required=True, help="GPU (cuda) or CPU (cpu)")
args = parser.parse_args()

df=pd.read_csv(args.prompt_path)
prompts = df['prompt'].tolist()
names = df['file_name'].tolist()

custom_unet = UNet2DConditionModel.from_pretrained(args.unet_path, 
                                                   torch_dtype=torch.float16,
                                                   use_safetensors=True)

base = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0",
                                          unet=custom_unet,    
                                          torch_dtype=torch.float16,
                                          variant="fp16",
                                          use_safetensors=True).to(args.device)


p = Path(args.inference_path)
p.mkdir(parents=True, exist_ok=True)

generator = torch.Generator(device=args.device).manual_seed(args.seed)

for i in range (len(prompts)):
    print ('prompt', str(i+1)+'/'+str(len(prompts)))
    print(names[i])
    print (prompts[i])
    image = base(prompt=prompts[i],
                 num_inference_steps=args.n_inf_steps,
                 generator=generator).images[0].save(args.inference_path+names[i])


