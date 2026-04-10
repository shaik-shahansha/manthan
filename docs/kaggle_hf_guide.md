# Kaggle + HuggingFace Setup Guide

Complete guide to run Genesis Manthan training on Kaggle free GPU and push results to HuggingFace Hub.

---

## Part 1 — HuggingFace Setup

### 1. Create account
Go to [huggingface.co/join](https://huggingface.co/join) and register with email.

### 2. Create a Write token
1. Go to [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
2. Click **New token**
3. Name: `manthan-write`, Role: **Write**
4. Click **Generate token** — copy it immediately (shown only once)

> **Security**: Never commit tokens to git or paste them in shared chats. If exposed, revoke immediately and create a new one.

### 3. Push the training dataset (run once locally)

```powershell
.venv\Scripts\python.exe -c "
from datasets import load_from_disk
ds = load_from_disk('data/processed/manthan_dataset')
ds.push_to_hub('Shahansha/manthan-tool-reasoning-v1', token='hf_YOUR_TOKEN')
print('Done:', ds)
"
```

> Your HuggingFace username is case-sensitive. Check it at [huggingface.co/settings/profile](https://huggingface.co/settings/profile).

Dataset will be live at: `https://huggingface.co/datasets/Shahansha/manthan-tool-reasoning-v1`

---

## Part 2 — Kaggle Setup

### 1. Create account
Go to [kaggle.com](https://kaggle.com) → **Register**. Free, no credit card needed.

### 2. Verify phone (required for GPU access)
1. Go to [kaggle.com/settings](https://kaggle.com/settings)
2. Scroll to **Phone Verification** → verify your number
3. Without this step, GPU option will not appear

### 3. Add HF_TOKEN secret
> The secret lives in the **notebook editor**, NOT in account Settings.

1. Go to [kaggle.com/code](https://kaggle.com/code) → **New Notebook**
2. In the editor top menu: **Add-ons → Secrets → Add a new secret**
   - Label: `HF_TOKEN`
   - Value: your `hf_...` write token
3. Toggle **Attach to notebook** to ON

### 4. Enable GPU and Internet
- Right panel → **Session Options → Accelerator → GPU T4 x2**
- Right panel → **Internet → On** (required to download models and push to Hub)

---

## Part 3 — Run Training on Kaggle

### Phase 1 — SFT (~2-3 hours)

1. Go to [kaggle.com/code](https://kaggle.com/code) → **New Notebook**
2. **File → Import Notebook** → upload `notebooks/02_sft_kaggle.ipynb`
3. Set GPU T4 x2 + Internet ON + HF_TOKEN secret attached
4. **Run All**
5. Output: SFT adapter pushed to `Shahansha/Manthan-1.5B-sft-v0.1` on HuggingFace Hub

### Phase 2 — GRPO (~8-10 hours, run across multiple sessions)

1. **New Notebook** → upload `notebooks/03_grpo_kaggle.ipynb`
2. Same settings: GPU T4 x2 + Internet ON + HF_TOKEN attached
3. **Run All**
4. If session times out (12 hr Kaggle limit): re-run with `RESUME_FROM_CHECKPOINT = True` in Cell 3
5. Checkpoints are saved to HuggingFace Hub every 50 steps automatically

---

## GPU Quota

| Platform | Free GPU | Session limit | Weekly limit | Notes |
|---|---|---|---|---|
| Kaggle | T4 x2 (16 GB each) | 12 hours | 30 hours | Most reliable |
| Google Colab | T4 (15 GB) | ~12 hours | Varies | Less stable, may disconnect |

**Kaggle recommended** — 30 hrs/week, stable sessions, no random disconnects.

---

## Troubleshooting

| Error | Fix |
|---|---|
| `403 Forbidden: You don't have the rights` | Token is Read-only — create a new **Write** token |
| `Repo id must be in the form 'repo_name'` | Path starts with `./` — use absolute path or correct Hub ID |
| GPU shows 0% utilization | Normal during GRPO reward computation between rollouts — check VRAM usage instead |
| `ModuleNotFoundError: kaggle_secrets` | Running outside Kaggle — the notebook falls back to `os.environ.get('HF_TOKEN')` automatically |
| Session timed out mid-GRPO | Set `RESUME_FROM_CHECKPOINT = True` in Cell 3 and re-run |
| HF push fails with wrong username | Username is case-sensitive — check exact casing at huggingface.co/settings/profile |
