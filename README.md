# Health Insurance Premium Pricing (GLM + Gradient Boosting)



[![Open In Colab]
https://colab.research.google.com/drive/1oogzDyU_Rd2WEqNJeh2T8zzTqjDuoyPZ?usp=sharing

**What this does (in one line):**  
Given a few details about a person (age, sex, BMI, children, smoker, region), the model estimates their annual medical insurance charges.

---

## Why this is useful
- **Pricing assistant:** get a quick, data-driven estimate to support rating discussions.
- **Underwriting triage:** flag obvious high/low risk cases early (e.g., smokers tend to have much higher predicted cost).
- **Learning tool:** shows how classical actuarial GLMs compare with modern machine-learning (Gradient Boosting).

> ⚠️ This is **for demonstration/education**. It’s not a final rating plan and should not be used as-is for binding prices.

---

## The data (small, public demo set)
- `insurance.csv` — 1.3k rows of de-identified records with columns:
  `age, sex, bmi, children, smoker, region, charges`  
- No PHI/PII. You can replace it with your own data (same columns) to re-train.

---

## Results at a glance (on a held-out test set)

| Model | What it is | MAE ↓ | RMSE ↓ | R² ↑ |
|---|---|---:|---:|---:|
| **GLM** | Classic actuarial model (Gamma with log link) | **7174.06** | 9990.77 | 0.357 |
| **Gradient Boosting** | Modern ML, captures interactions/non-linearity | **2404.90** | **4328.15** | **0.879** |

**How to read this:**
- **MAE** (average absolute error) ~ how far predictions are from actual charges, on average. Lower is better.  
- **RMSE** is similar but penalizes big mistakes more. Lower is better.  
- **R²** ~ “how much of the variation we explain.” Higher is better; 0.88 is strong for this toy dataset.

---

## What drives cost (intuition, not code)
- **Smoker** status is the **strongest driver** (biggest jump in predicted charges).
- **BMI** and **Age** matter next.
- **Children**, **sex**, and **region** are weaker in this small dataset.

You can regenerate a feature-importance table by running the training script; it saves `outputs/gbr_permutation_importance.csv`.

---

## Try it without installing anything
1. Click the **Colab** badge at the top.  
2. Upload your `insurance.csv` (or use the demo one).  
3. Run the notebook cells — you’ll get metrics, a calibration plot, and a saved model.

---

## Quick start (local)
```bash
# 1) Setup
python -m venv .venv
# Windows: .\.venv\Scripts\Activate.ps1
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt

# 2) Put your CSV at: data/insurance.csv

# 3) Train & generate artifacts (metrics, plots, model)
python train.py --csv data/insurance.csv --out outputs
