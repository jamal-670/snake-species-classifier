# 🐍 SnakeID — Snake Species Classifier

> A complete end-to-end deep learning project for identifying 39 snake species found in Pakistan and South Asia.

![Python](https://img.shields.io/badge/Python-3.11-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.7-orange)
![Accuracy](https://img.shields.io/badge/Accuracy-81.6%25-green)
![Species](https://img.shields.io/badge/Species-39-yellowgreen)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

---

## 🌐 Live Demo

👉 **[Try it on Hugging Face Spaces](https://huggingface.co/spaces/IbrahimJamal2005/snake-classifier)**

Upload any snake photo and get the predicted species with confidence score.

---

## 📌 Project Overview

| Detail | Value |
|--------|-------|
| Model | EfficientNet-B3 |
| Classes | 39 snake species |
| Accuracy | 81.6% |
| Training Images | 8,800 (200 per class) |
| Training Platform | Kaggle Tesla P100 GPU |
| Training Time | ~2 hours |
| Model Size | 45 MB |
| Deployment | Hugging Face Spaces |

---

## 🗂️ Project Structure

```
snake_classifier/
├── src/                   # Core source code
│   ├── data/
│   │   ├── audit.py       # Data cleaning and deduplication
│   │   ├── augment.py     # Offline augmentation
│   │   ├── split.py       # Stratified splits
│   │   └── dataset.py     # PyTorch Dataset class
│   ├── models/
│   │   └── classifier.py  # EfficientNet-B3 + custom head
│   ├── training/
│   │   └── trainer.py     # 3-stage training loop
│   └── evaluation/
│       └── evaluate.py    # Metrics + confusion matrix
├── configs/
│   └── base.yaml          # All hyperparameters
├── api/
│   ├── main.py            # FastAPI REST endpoint
│   └── Dockerfile
├── kaggle_kernel/         # Kaggle training notebook
├── hf_space/              # Hugging Face deployment
├── reports/
│   └── confusion_matrix.png
└── requirements.txt
```

---

## 🧠 Model Architecture

```
Input (260x260x3)
      |
EfficientNet-B3 Backbone (ImageNet pretrained)
      |
BatchNorm -> Dropout(0.4)
      |
Linear(1536 -> 512) + SiLU
      |
BatchNorm -> Dropout(0.2)
      |
Linear(512 -> 39)
      |
Predicted Species
```

---

## 🔄 Training Pipeline

### 3-Stage Transfer Learning

| Stage | Epochs | LR | What trains |
|-------|--------|----|-------------|
| Stage 1 | 5 | 1e-3 | Head only — backbone frozen |
| Stage 2 | 15 | 3e-4 | Top 3 backbone blocks |
| Stage 3 | 20 | 5e-5 | Full model fine-tune |

**Techniques used:**

- Mixed precision training (AMP)
- Label smoothing (0.1)
- Cosine LR scheduler
- Gradient clipping (max norm 1.0)
- Early stopping (patience 7)
- Offline augmentation for class balancing

---

## 📊 Dataset

- **Source:** Student-collected images from multiple platforms
- **Original:** 44 species folders, heavily imbalanced (8 to 513 images per class)
- **After cleaning:** 39 species (merged duplicates, fixed encoding issues)
- **After augmentation:** 200 images per class

| Split | Images | Percentage |
|-------|--------|------------|
| Train | 5,460 | 70% |
| Validation | 1,170 | 15% |
| Test | 1,170 | 15% |

---

## 🚀 Run Locally

```bash
# Clone the repository
git clone https://github.com/jamal-670/snake-species-classifier.git
cd snake-species-classifier

# Create environment
conda create -n snake_clf python=3.11 -y
conda activate snake_clf
pip install -r requirements.txt

# Run the API
cd api
uvicorn main:app --reload --port 8000
```

Open your browser at `http://localhost:8000/docs` to test the API.

---

## 📦 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | API info |
| GET | `/health` | Health check |
| GET | `/classes` | List all 39 species |
| POST | `/predict` | Upload image and get prediction |

**Example response:**

```json
{
  "prediction": {
    "species": "indian_cobra",
    "confidence": 94.32
  },
  "top5": [
    {"rank": 1, "species": "indian_cobra",     "confidence": 94.32},
    {"rank": 2, "species": "spectacled_cobra", "confidence": 3.21},
    {"rank": 3, "species": "common_krait",     "confidence": 1.12},
    {"rank": 4, "species": "indian_python",    "confidence": 0.89},
    {"rank": 5, "species": "rat_snake",        "confidence": 0.46}
  ]
}
```

---

## 🛠️ Tech Stack

| Category | Tool |
|----------|------|
| Deep Learning | PyTorch 2.7 |
| Model | EfficientNet-B3 (timm) |
| Augmentation | Albumentations |
| API | FastAPI + Uvicorn |
| Web UI | Gradio |
| GPU Training | Kaggle P100 |
| Hosting | Hugging Face Spaces |
| OS | Linux Mint |

---

## 🐍 Supported Species

1. Banded Kukri Snake
2. Beaked Sea Snake
3. Black-headed Royal Snake
4. Blunt-nosed Viper
5. Brahminy Blind Snake
6. Buff-striped Keelback
7. Caspian Cobra
8. Checkered Keelback
9. Coluber Ventromaculatus
10. Common Bronze Snake
11. Common Cat Snake
12. Common Krait
13. Common Sand Boa
14. Common Wolf Snake
15. Diadem Snake
16. Dice Snake
17. Gamma Snake
18. Glossy-bellied Racer
19. Hemorrhois Ravergieri
20. Himalayan Pit Viper
21. Hydrophis Ornatus
22. Hydrophis Spiralis
23. Indian Cobra
24. Indian Python
25. Indian Wolf Snake
26. Jan's Cliff Racer
27. Lycodon Striatus
28. Oriental Rat Snake
29. Rat Snake
30. Red Sand Boa
31. Russell's Viper
32. Saharan Horned Viper
33. Sand Boa
34. Saw-scaled Viper
35. Schokari Sand Racer
36. Sind Krait
37. Sindh Saw-scaled Viper
38. Sochurek's Saw-scaled Viper
39. Spectacled Cobra

---

## ⚠️ Disclaimer

This model is for **educational purposes only**. Do not use it to make safety decisions about real snakes. Always consult a professional when dealing with potentially dangerous snakes.

---

## 👤 Author

**Ibrahim Khan**

- GitHub: [@jamal-670](https://github.com/jamal-670)
- Hugging Face: [IbrahimJamal2005](https://huggingface.co/IbrahimJamal2005)

---

## 📄 License

MIT License
