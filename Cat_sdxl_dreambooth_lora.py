from google.colab import drive
drive.mount('/content/drive')

# Check the GPU
!nvidia-smi

# Install dependencies.
!pip install bitsandbytes transformers accelerate peft -q

"""Make sure to install `diffusers` from `main`."""

!pip install git+https://github.com/huggingface/diffusers.git -q

"""Download diffusers SDXL DreamBooth training script."""

!wget https://raw.githubusercontent.com/huggingface/diffusers/main/examples/dreambooth/train_dreambooth_lora_sdxl.py

from google.colab import drive
import os
import shutil

gdrive_cat_dir = "/content/drive/MyDrive/Cat"
local_dir = "/content/Cat_dataset"

# 複製整個資料夾
shutil.copytree(gdrive_cat_dir, local_dir, dirs_exist_ok=True)

print("已複製到：", local_dir)
print(os.listdir(local_dir))

from PIL import Image

def image_grid(imgs, rows, cols, resize=256):

    if resize is not None:
        imgs = [img.resize((resize, resize)) for img in imgs]
    w, h = imgs[0].size
    grid = Image.new("RGB", size=(cols * w, rows * h))
    grid_w, grid_h = grid.size

    for i, img in enumerate(imgs):
        grid.paste(img, box=(i % cols * w, i // cols * h))
    return grid

import glob

# change path to display images from your local dir
img_paths = "./Cat_dataset/*.jpg"
imgs = [Image.open(path) for path in glob.glob(img_paths)]

num_imgs_to_preview = 15
image_grid(imgs[:num_imgs_to_preview], 1, num_imgs_to_preview)

import requests
from transformers import AutoProcessor, BlipForConditionalGeneration
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"

# load the processor and the captioning model
blip_processor = AutoProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base",torch_dtype=torch.float16).to(device)

# captioning utility
def caption_images(input_image):
    inputs = blip_processor(images=input_image, return_tensors="pt").to(device, torch.float16)
    pixel_values = inputs.pixel_values

    generated_ids = blip_model.generate(pixel_values=pixel_values, max_length=50)
    generated_caption = blip_processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return generated_caption

import glob
from PIL import Image

# create a list of (Pil.Image, path) pairs
local_dir = "./Cat_dataset/"
imgs_and_paths = [(path,Image.open(path)) for path in glob.glob(f"{local_dir}*.jpg")]

import json

caption_prefix = "a photo of Lilly cat, " #@param
with open(f'{local_dir}metadata.jsonl', 'w') as outfile:
  for img in imgs_and_paths:
      caption = caption_prefix + caption_images(img[1]).split("\n")[0]
      entry = {"file_name":img[0].split("/")[-1], "prompt": caption}
      json.dump(entry, outfile)
      outfile.write('\n')

import gc

# delete the BLIP pipelines and free up some memory
del blip_processor, blip_model
gc.collect()
torch.cuda.empty_cache()

import locale
locale.getpreferredencoding = lambda: "UTF-8"

!accelerate config default

from huggingface_hub import login

login(token="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")

!pip install datasets -q

mv Cat_dataset/metadata.jsonl .

import os
import shutil
from pathlib import Path

# 設定原始資料夾和新的乾淨資料夾名稱
original_dir = "Cat_dataset"
new_dir = "Cat_dataset_clean"

# 建立新資料夾（如果已存在則忽略）
os.makedirs(new_dir, exist_ok=True)

# 定義允許的圖片格式
allowed_extensions = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

print(f"正在將圖片從 {original_dir} 複製到 {new_dir} ...")

count = 0
# 遍歷原始資料夾
for file_path in Path(original_dir).iterdir():
    # 這裡的邏輯會過濾掉 .ipynb_checkpoints 和 metadata.jsonl
    if file_path.is_file() and file_path.suffix.lower() in allowed_extensions:
        shutil.copy(file_path, os.path.join(new_dir, file_path.name))
        count += 1

print(f"成功！已複製 {count} 張圖片到 '{new_dir}' 資料夾。")
print("請使用這個新資料夾進行訓練。")

#!/usr/bin/env bash
!accelerate launch train_dreambooth_lora_sdxl.py \
  --pretrained_model_name_or_path="stabilityai/stable-diffusion-xl-base-1.0" \
  --pretrained_vae_model_name_or_path="madebyollin/sdxl-vae-fp16-fix" \
  --instance_data_dir="./Cat_dataset_clean" \
  --output_dir="Lilly_cat_50steps" \
  --caption_column="prompt"\
  --mixed_precision="fp16" \
  --instance_prompt="a photo of Lilly cat" \
  --resolution=1024 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=3 \
  --gradient_checkpointing \
  --learning_rate=1e-4 \
  --snr_gamma=5.0 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --mixed_precision="fp16" \
  --use_8bit_adam \
  --max_train_steps=50 \
  --checkpointing_steps=717 \
  --seed="0"

from huggingface_hub import whoami
from pathlib import Path

output_dir = "Lilly_cat_50steps" #@param
username = whoami(token=Path("/root/.cache/huggingface/"))["name"]
repo_id = f"{username}/{output_dir}"


# push to the hub🔥
from train_dreambooth_lora_sdxl import save_model_card
from huggingface_hub import upload_folder, create_repo

repo_id = create_repo(repo_id, exist_ok=True).repo_id

# change the params below according to your training arguments
save_model_card(
    repo_id = repo_id,
    images=[],
    base_model="stabilityai/stable-diffusion-xl-base-1.0",
    train_text_encoder=False,
    instance_prompt="a photo of Lilly cat",
    validation_prompt=None,
    repo_folder=output_dir,
    vae_path="madebyollin/sdxl-vae-fp16-fix",
    use_dora=False
)

upload_folder(
    repo_id=repo_id,
    folder_path=output_dir,
    commit_message="End of training",
    ignore_patterns=["step_*", "epoch_*"],
)

from IPython.display import display, Markdown

link_to_model = f"https://huggingface.co/{repo_id}"
display(Markdown("### Your model has finished training.\nAccess it here: {}".format(link_to_model)))

import torch
from diffusers import DiffusionPipeline, AutoencoderKL

vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16)
pipe = DiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    vae=vae,
    torch_dtype=torch.float16,
    variant="fp16",
    use_safetensors=True
)
pipe.load_lora_weights(repo_id)
_ = pipe.to("cuda")

prompt = "a photo of Lilly cat lie on the beach" # @param

image = pipe(prompt=prompt, num_inference_steps=50).images[0]
image