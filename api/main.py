from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import torch
import torch.nn as nn
import timm
import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image
import numpy as np
import json, io, os

app = FastAPI(
    title="Snake Species Classifier",
    version="1.0.0"
)

# ── CORS — allows browser to call API ─────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Load model ─────────────────────────────────────────────
DEVICE      = torch.device("cpu")
CLASS_NAMES = json.load(open("../data/splits/classes.json"))
NUM_CLASSES = len(CLASS_NAMES)

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
    def forward(self, x): return self.head(self.backbone(x))

model = SnakeClassifier(NUM_CLASSES)
ckpt  = torch.load("../checkpoints/best_model.pt", map_location=DEVICE)
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

@app.get("/")
def root():
    return {"message": "Snake Classifier API running!",
            "classes": NUM_CLASSES}

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/classes")
def get_classes():
    return {"classes": CLASS_NAMES}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        return JSONResponse(status_code=400,
            content={"error": "File must be an image"})
    contents = await file.read()
    image    = Image.open(io.BytesIO(contents)).convert("RGB")
    tensor   = transform(image=np.array(image))["image"]
    tensor   = tensor.unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        probs = torch.softmax(model(tensor), dim=1)[0]
    top5_vals, top5_idx = probs.topk(5)
    return JSONResponse({
        "prediction": {
            "species":    CLASS_NAMES[top5_idx[0].item()],
            "confidence": round(top5_vals[0].item() * 100, 2),
        },
        "top5": [
            {"rank": i+1,
             "species":    CLASS_NAMES[top5_idx[i].item()],
             "confidence": round(top5_vals[i].item() * 100, 2)}
            for i in range(5)
        ]
    })
