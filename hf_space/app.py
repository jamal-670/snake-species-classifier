import gradio as gr
import torch
import torch.nn as nn
import timm
import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image
import numpy as np
import json

with open("classes.json") as f:
    CLASS_NAMES = json.load(f)
NUM_CLASSES = len(CLASS_NAMES)

VENOMOUS = [
    "indian_cobra","spectacled_cobra","common_krait","sind_krait",
    "russells_viper","saw-scaled_viper","sindh_saw-scaled_viper",
    "sochurek_saw-scaled_viper","himalayan_pit_viper","blunt-nosed_viper",
    "caspian_cobra","beaked_sea_snake","hydrophis_ornatus",
    "hydrophis_spiralis","stokes_sea_snake","yellow_bellied_sea_snake"
]

class SnakeClassifier(nn.Module):
    def __init__(self, num_classes, dropout=0.4):
        super().__init__()
        self.backbone = timm.create_model(
            'efficientnet_b3', pretrained=False, num_classes=0)
        in_f = self.backbone.num_features
        self.head = nn.Sequential(
            nn.BatchNorm1d(in_f), nn.Dropout(dropout),
            nn.Linear(in_f, 512),  nn.SiLU(),
            nn.BatchNorm1d(512),   nn.Dropout(dropout/2),
            nn.Linear(512, num_classes))
    def forward(self, x):
        return self.head(self.backbone(x))

DEVICE = torch.device("cpu")
model  = SnakeClassifier(NUM_CLASSES)
ckpt   = torch.load("best_model.pt", map_location=DEVICE)
model.load_state_dict(ckpt["model_state"])
model.eval()
print(f"Model loaded — {NUM_CLASSES} classes")

transform = A.Compose([
    A.Resize(300, 300),
    A.CenterCrop(260, 260),
    A.Normalize(mean=[0.485,0.456,0.406],
                std=[0.229,0.224,0.225]),
    ToTensorV2(),
])

def predict(image):
    if image is None:
        return {}, "Please upload an image"
    img_arr = np.array(image.convert("RGB"))
    tensor  = transform(image=img_arr)["image"]
    tensor  = tensor.unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        probs = torch.softmax(model(tensor), dim=1)[0]
    top5_vals, top5_idx = probs.topk(5)
    results = {}
    for i in range(5):
        name = CLASS_NAMES[top5_idx[i].item()].replace("_"," ").title()
        results[name] = round(top5_vals[i].item(), 4)
    top_species = CLASS_NAMES[top5_idx[0].item()]
    is_venomous = any(v in top_species for v in VENOMOUS)
    top_name    = top_species.replace("_"," ").title()
    top_conf    = round(top5_vals[0].item()*100, 1)
    if is_venomous:
        warning = f"⚠️ WARNING: {top_name} is potentially VENOMOUS ({top_conf}%). Do not handle!"
    else:
        warning = f"✅ {top_name} — Non-venomous ({top_conf}% confidence)"
    return results, warning

with gr.Blocks(title="SnakeID") as demo:
    gr.Markdown("# 🐍 SnakeID — Snake Species Classifier")
    gr.Markdown("EfficientNet-B3 · 39 species · 81.6% accuracy")
    with gr.Row():
        with gr.Column():
            image_input = gr.Image(type="pil", label="Upload snake image", height=320)
            submit_btn  = gr.Button("🔍 Identify Species", variant="primary", size="lg")
        with gr.Column():
            label_output   = gr.Label(label="Top 5 predictions", num_top_classes=5)
            warning_output = gr.Textbox(label="Result", interactive=False)
    submit_btn.click(fn=predict, inputs=image_input,
                     outputs=[label_output, warning_output])
    image_input.change(fn=predict, inputs=image_input,
                       outputs=[label_output, warning_output])

demo.launch()
