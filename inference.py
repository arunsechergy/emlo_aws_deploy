
import gradio as gr
from typing import Dict
import boto3
import urllib
import torch
import timm
import numpy as np

from PIL import Image
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform


# ENVIRONMENT VARIABLES TO BE DECLARED
BUCKET_NAME = "emlo-session-5"
PREFIX_KEY = "timm_resnet/run_5/scripted_model_ckpt_2022.pth"
MODEL_CKPTH = "resnet_checkpoint_file.pth"
MODEL = 'resnet18'

# Download S3 model checkpoint file
s3 = boto3.client('s3')
s3.download_file(BUCKET_NAME, PREFIX_KEY, MODEL_CKPTH)

# Load the model
ckpt = torch.load(MODEL_CKPTH)

# Load the state dict of trained model
# initialize the model architecture
model = timm.create_model(MODEL, pretrained=False)
model.load_state_dict(ckpt)
model.eval()

# convert to scripted model
scripted_model = torch.jit.script(model)

# get the classnames
url, filename = (
    "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt",
    "imagenet_classes.txt",
)
urllib.request.urlretrieve(url, filename)
with open("imagenet_classes.txt", "r") as f:
    categories = [s.strip() for s in f.readlines()]

def predict(inp_img: Image) -> Dict[str, float]:
    config = resolve_data_config({}, model=MODEL)
    transform = create_transform(**config)

    img_tensor = transform(inp_img).unsqueeze(0)  # transform and add batch dimension

    # inference
    with torch.no_grad():
        out = scripted_model(img_tensor)
        probabilities = torch.nn.functional.softmax(out[0], dim=0)
        confidences = {categories[i]: float(probabilities[i]) for i in range(1000)}

    return confidences


if __name__ == "__main__":
    gr.Interface(
        fn=predict, inputs=gr.Image(type="pil"), outputs=gr.Label(num_top_classes=10)
    ).launch(server_name="0.0.0.0")