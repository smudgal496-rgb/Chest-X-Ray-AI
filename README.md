# ChestAI — Chest X-Ray Diagnostic Tool

An AI-powered web application for chest X-ray analysis. Upload an X-ray and get per-condition probability scores alongside a Grad-CAM heatmap showing which image regions drove each prediction.

**Built by:** Shivam Mudgal & Vraddhi Srivastava  
**Contact:** smudgal496@gmail.com · vraddhisrivastava@gmail.com

> ⚠️ For educational and research purposes only. Not a substitute for professional medical diagnosis. Always consult a qualified radiologist.

---

## Live Demo

Hosted on Hugging Face Spaces (backend) + GitHub Pages (frontend).  
→ **[Try ChestAI live](https://shivammudgal-chestxrayai.hf.space)**

---

## Conditions Detected

| Condition | Model AUC | Notes |
|---|---|---|
| Cardiomegaly | **0.91** | Strongest signal; clear spatial boundary aids GradCAM |
| Effusion | **0.88** | High contrast pleural fluid makes this learnable |
| Atelectasis | **0.79** | Moderate; overlaps visually with other opacities |
| Pneumonia | **0.73** | Variable presentation reduces consistency |
| Infiltration | **0.70** | Weakest; NIH label noise is highest for this class |

**Mean AUC: 0.80** across all five conditions.

---

## How It Works

1. A **DenseNet121** backbone (pretrained on ImageNet) is fine-tuned on the NIH ChestX-ray14 dataset — 112,120 frontal-view X-rays across 14 pathology labels. We use 5 of the 14 labels.
2. The final classifier is replaced with a custom head: `Linear(1024→512) → ReLU → Dropout(0.3) → Linear(512→5)`.
3. Output passes through `sigmoid` — each condition is scored independently (multi-label, not multi-class).
4. **Grad-CAM** is applied to `denseblock4` (the deepest convolutional block), producing a heatmap that highlights the image regions most responsible for the top-predicted condition.

### Training Setup

| Parameter | Value |
|---|---|
| Dataset | NIH ChestX-ray14 |
| Training images | ~90,000 |
| Validation images | ~11,000 |
| Test images | ~11,000 |
| Input resolution | 224 × 224 |
| Backbone | DenseNet121 (ImageNet pretrained) |
| Optimizer | Adam |
| Loss | Binary Cross-Entropy with logits |
| Epochs | Phase 1 (frozen backbone) + Phase 2 (fine-tuned) |

---

## Features

- **Multi-label prediction** — five conditions scored independently in one forward pass
- **Grad-CAM visualization** — spatial explanation overlaid on the original X-ray
- **Adjustable threshold** — slider from 0.10 to 0.90 to tune sensitivity vs. specificity
- **Patient records** — save analyses to a per-user Supabase database
- **PDF report generation** — downloadable diagnostic report per patient
- **CSV export** — bulk download of all saved records
- **Auth system** — email/password login via Supabase Auth

---

## Tech Stack

| Layer | Technology |
|---|---|
| Model | PyTorch, DenseNet121, `pytorch-grad-cam` |
| Backend | Python, Gradio (hosted on HF Spaces) |
| Frontend | Vanilla HTML/CSS/JS, Chart.js, jsPDF |
| Database | Supabase (PostgreSQL) |
| Auth | Supabase Auth |

---

## Local Setup

### Backend (HF Space / local)

```bash
# Clone and install
pip install -r requirements.txt

# Place model weights at project root
# File: best_model_phase2.pth

# Run backend
python app.py
```

`requirements.txt`:
```
gradio
torch
torchvision
pillow
numpy
pytorch-grad-cam
```

### Frontend

The frontend is a single `index.html` file. Configure your own Supabase project:

```html
<!-- In index.html, replace these two values -->
const SUPABASE_URL  = 'YOUR_SUPABASE_PROJECT_URL';
const SUPABASE_ANON = 'YOUR_SUPABASE_ANON_KEY';
```

Then update `HF_SPACE` to point to your own deployed Gradio backend:

```html
const HF_SPACE = 'https://your-hf-username-your-space-name.hf.space';
```

Open `index.html` directly in a browser or serve it via any static host.

---

## Known Limitations

- **Infiltration AUC is 0.70** — the NIH dataset has well-documented label noise for Infiltration, where inter-radiologist agreement is lowest. The model reflects this uncertainty.
- **5 of 14 conditions** — the NIH dataset covers 14 pathologies; we trained on 5. The model will not flag conditions outside this set.
- **No lateral view** — the model was trained exclusively on frontal (PA) views. Lateral or AP views may produce unreliable scores.
- **No DICOM support** — input must be a standard image file (PNG/JPG). Clinical DICOM preprocessing is not implemented.
- **Threshold is user-controlled** — the default 0.50 threshold is not clinically validated. Adjusting it changes sensitivity/specificity tradeoffs in ways that depend on the use case.
- **Model images not stored** — due to Supabase row size limits, only numerical scores are saved per record. The PDF report does not include the GradCAM image.

---

## Repository Structure

```
chestai/
├── app.py              # Gradio backend — model loading, inference, GradCAM
├── index.html          # Full frontend — auth, upload, visualization, records
├── requirements.txt    # Python dependencies
└── README.md
```

---

## License

MIT License. See `LICENSE` for details.
