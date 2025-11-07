#!/usr/bin/env python3
# ======================================================================
# Neural Fine-Tuning of CLIP-SAE-ViT-L-14 (CPU-safe, top-K slicing)
# ======================================================================

import os, numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F
from pathlib import Path
from tqdm import tqdm
from safetensors import safe_open
from huggingface_hub import snapshot_download
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr
import argparse

# ---------------------------------------------------------------
# Args
# ---------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--topk", type=int, default=8192,
                    help="Number of top SAE latents to keep (default: 8192)")
parser.add_argument("--epochs", type=int, default=25)
args = parser.parse_args()

# ---------------------------------------------------------------
# Config
# ---------------------------------------------------------------
DEVICE = "cpu"      # ✅ safe default
SEED = 42
torch.manual_seed(SEED); np.random.seed(SEED)

DATA_PATH = "/home/maria/LuckyMouse/pixel_transformer_neuro/data/processed/hybrid_neural_responses.npy"
IMG_DIR   = Path("/home/maria/MITNeuralComputation/vit_embeddings/images")
SAE_REPO  = "zer0int/CLIP-SAE-ViT-L-14"
LOCAL_SAE_DIR = "./clip_sae_vitl14_weights"

EPOCHS         = args.epochs
LR             = 1e-3
BATCH_SIZE     = 4
LAMBDA_SPARSE  = 1e-3
LAMBDA_NEURAL  = 5e-2
VAL_SPLIT      = 0.2
TOPK           = args.topk

print(f"🔹 Device: {DEVICE} | TOPK={TOPK}")

# ---------------------------------------------------------------
# Load neural data
# ---------------------------------------------------------------
dat = np.load(DATA_PATH)
Y_binary = (dat > 0).astype(np.float32)
n_neurons, n_samples = Y_binary.shape
N_IMAGES = 118
N_TRIALS = n_samples // N_IMAGES
img_ids_full = np.repeat(np.arange(N_IMAGES), N_TRIALS)
Y_image_mean = np.array([Y_binary[:, img_ids_full == i].mean(axis=1)
                         for i in range(N_IMAGES)], dtype=np.float32)
n_neurons = Y_image_mean.shape[1]
print(f"✅ Neural data: {n_neurons} neurons × {N_IMAGES} images")

# ---------------------------------------------------------------
# Download SAE weights
# ---------------------------------------------------------------
snapshot_download(repo_id=SAE_REPO, local_dir=LOCAL_SAE_DIR)

# ---------------------------------------------------------------
# Load CLIP features (precompute if needed)
# ---------------------------------------------------------------
FEAT_PATH = "clip_vitl14_feats.npy"
if Path(FEAT_PATH).exists():
    X_clip = np.load(FEAT_PATH)
else:
    print("⏳ Extracting CLIP features ...")
    clip = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(DEVICE)
    clip.eval(); processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
    feats = []
    for p in tqdm(sorted(IMG_DIR.glob("scene_*.png"))):
        img = Image.open(p).convert("RGB")
        inputs = processor(images=img, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            f = clip.get_image_features(**inputs).squeeze().cpu().numpy()
        feats.append(f.astype(np.float32))
    X_clip = np.stack(feats)
    np.save(FEAT_PATH, X_clip)
    del clip
print("✅ CLIP features:", X_clip.shape)

# ---------------------------------------------------------------
# Load SAE encoder matrix
# ---------------------------------------------------------------
print("\n🔍 Scanning for encoder matrices ...")
cands = []
for f in Path(LOCAL_SAE_DIR).glob("*.safetensors"):
    with safe_open(f, framework="pt", device="cpu") as sf:
        for k in sf.keys():
            t = sf.get_tensor(k)
            if t.ndim == 2:
                cands.append((f, k, t.shape))
sae_path, enc_key, enc_shape = max(cands, key=lambda x: x[2][0]*x[2][1])
print(f"→ Using {sae_path.name} | key={enc_key} | shape={enc_shape}")

with safe_open(sae_path, framework="pt", device="cpu") as sf:
    W = sf.get_tensor(enc_key)
    b = None
    for k in sf.keys():
        if "bias" in k.lower():
            tb = sf.get_tensor(k)
            if tb.ndim == 1: b = tb; break

A, B = W.shape
if A > B: in_dim, out_dim, W_use = B, A, W
else:     in_dim, out_dim, W_use = A, B, W.T
if (b is None) or (b.shape[0] != out_dim): b = torch.zeros(out_dim)
print(f"✓ SAE dims: {in_dim} → {out_dim}")

# ---------------------------------------------------------------
# Slice top-K latents
# ---------------------------------------------------------------
if TOPK < out_dim:
    print(f"🔹 Slicing SAE: keeping top {TOPK}/{out_dim} latents")
    W_use = W_use[:TOPK, :]
    b = b[:TOPK]
    out_dim = TOPK
else:
    print(f"🔹 Using all {out_dim} latents")

# ---------------------------------------------------------------
# Modules
# ---------------------------------------------------------------
class SAEEncoder(nn.Module):
    def __init__(self, in_dim, out_dim, W, b):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        with torch.no_grad():
            self.linear.weight.copy_(W)
            self.linear.bias.copy_(b)
    def forward(self, x): return F.relu(self.linear(x))

class NeuralFineTuneSAE(nn.Module):
    """Decoder frozen; only sparsity + neural losses."""
    def __init__(self, sae: SAEEncoder, n_neurons, λs=1e-3, λn=5e-2):
        super().__init__()
        self.encoder = sae.linear
        self.neural_head = nn.Linear(sae.linear.out_features, n_neurons, bias=False)
        self.λs, self.λn = λs, λn
    def forward(self, x, y=None):
        z = F.relu(self.encoder(x))
        loss_s = z.abs().mean()
        loss_n = torch.tensor(0.0)
        if y is not None:
            y_pred = self.neural_head(z)
            loss_n = F.mse_loss(y_pred, y)
        return self.λs*loss_s + self.λn*loss_n, z

# ---------------------------------------------------------------
# Split train/val
# ---------------------------------------------------------------
X_train, X_val, Y_train, Y_val = train_test_split(
    X_clip, Y_image_mean, test_size=VAL_SPLIT, random_state=SEED
)
X_train_t, X_val_t = torch.tensor(X_train), torch.tensor(X_val)
Y_train_t, Y_val_t = torch.tensor(Y_train), torch.tensor(Y_val)
print(f"Train: {len(X_train)} | Val: {len(X_val)}")

# ---------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------
sae = SAEEncoder(in_dim, out_dim, W_use, b)
model = NeuralFineTuneSAE(sae, n_neurons, LAMBDA_SPARSE, LAMBDA_NEURAL)
optim = torch.optim.SGD(model.parameters(), lr=LR, momentum=0.9, nesterov=True)

def batches(X, Y, bs):
    for i in range(0, len(X), bs):
        yield X[i:i+bs], Y[i:i+bs]

print("\n🚀 Starting training ...")
for epoch in range(1, EPOCHS+1):
    model.train(); total = 0.0
    for xb, yb in batches(X_train_t, Y_train_t, BATCH_SIZE):
        optim.zero_grad(set_to_none=True)
        loss, _ = model(xb, yb)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optim.step()
        total += loss.item() * len(xb)
    train_loss = total / len(X_train_t)
    model.eval()
    with torch.no_grad():
        val_loss, _ = model(X_val_t, Y_val_t)
    print(f"Epoch {epoch:02d} | Train {train_loss:.6f} | Val {val_loss.item():.6f}")

# ---------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------
model.eval()
with torch.no_grad():
    _, z_val = model(X_val_t)
    Y_pred = model.neural_head(z_val).numpy()
    Y_true = Y_val
corrs = []
for i in range(n_neurons):
    xi, yi = Y_pred[:, i], Y_true[:, i]
    if np.std(xi) < 1e-8 or np.std(yi) < 1e-8:
        corrs.append(np.nan)
    else:
        corrs.append(pearsonr(xi, yi)[0])
corrs = np.array(corrs)
mean_corr = np.nanmean(corrs)
brier = mean_squared_error(Y_true.flatten(), Y_pred.flatten())

print(f"\n✅ Mean neuron corr: {mean_corr:.4f}")
print(f"✅ Brier (MSE): {brier:.6f}")
print(f"✅ Trainable params: "
      f"{sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

# ---------------------------------------------------------------
# Save
# ---------------------------------------------------------------
torch.save(model.state_dict(), f"neural_finetuned_sae_top{TOPK}_cpu.pth")
pd.DataFrame({"neuron_corr": corrs}).to_csv(
    f"neural_finetuned_corrs_top{TOPK}_cpu.csv", index=False)
print(f"\n💾 Saved model → neural_finetuned_sae_top{TOPK}_cpu.pth")
print(f"💾 Saved correlations → neural_finetuned_corrs_top{TOPK}_cpu.csv")
