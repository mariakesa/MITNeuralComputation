# ======================================================================
#  Deep Nonlinear Sparse Autoencoder (SwishGLU SAE)
#  - 5-layer encoder/decoder with SwishGLU activations + dropout
#  - Overcomplete expansion, strong sparsity, no decoder bias
#  - Encourages true nonlinear manifold
# ======================================================================

import os, pickle, numpy as np, pandas as pd, torch, torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from sklearn.linear_model import Ridge
from skbio.stats.composition import clr
from scipy.special import softmax

# --------------------------
# Config
# --------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
RNG_SEED = 42
np.random.seed(RNG_SEED)
torch.manual_seed(RNG_SEED)

# Data
VIT_PATH = "/home/maria/Documents/HuggingMouseData/MouseViTEmbeddings/google_vit-base-patch16-224_embeddings_logits.pkl"
OUT_DIR = "./sae_deep_swiglu_outputs"
os.makedirs(OUT_DIR, exist_ok=True)

# Hyperparameters
SAE_LR = 1e-4
SAE_EPOCHS = 800
SAE_BATCH = 64
SAE_L1 = 0.5
WEIGHT_DECAY = 1e-5
DROPOUT_P = 0.4
NORMALIZE_DECODER = True

print(f"Using device: {DEVICE}")

# --------------------------
# Load data (CLR of ViT softmax)
# --------------------------
with open(VIT_PATH, "rb") as f:
    vit_dict = pickle.load(f)["natural_scenes"]

X = np.stack(list(vit_dict))
X = softmax(X, axis=1)
X = clr(X + 1e-12)
scaler = StandardScaler().fit(X)
X_std = scaler.transform(X).astype(np.float32)
n_images, d_in = X_std.shape
print(f"Loaded {n_images} images, input dim {d_in}")

# --------------------------
# Activation: SwishGLU
# --------------------------
class SwishGLU(nn.Module):
    """x * sigmoid(gate(x)) variant."""
    def __init__(self):
        super().__init__()
        self.act = nn.Sigmoid()
    def forward(self, x):
        return x * self.act(x)

# --------------------------
# Deep SAE model
# --------------------------
class DeepSwishGLUSAE(nn.Module):
    def __init__(self, d_in: int, d_latent: int):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Linear(d_in, 2048),
            SwishGLU(),
            nn.Dropout(DROPOUT_P),
            nn.Linear(2048, 4096),
            SwishGLU(),
            nn.Dropout(DROPOUT_P),
            nn.Linear(4096, d_latent),
            SwishGLU(),
        )
        self.dec = nn.Sequential(
            nn.Linear(d_latent, 4096, bias=False),
            SwishGLU(),
            nn.Dropout(DROPOUT_P),
            nn.Linear(4096, 2048, bias=False),
            SwishGLU(),
            nn.Dropout(DROPOUT_P),
            nn.Linear(2048, d_in, bias=False),
        )

        # Kaiming init for deep ReLU/Swish
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, a=np.sqrt(5))
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        z = self.enc(x)
        x_hat = self.dec(z)
        return x_hat, z

# --------------------------
# Training setup
# --------------------------
model = DeepSwishGLUSAE(d_in=d_in, d_latent=1024).to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=SAE_LR, weight_decay=WEIGHT_DECAY)
mse = nn.MSELoss()
dl = DataLoader(TensorDataset(torch.from_numpy(X_std)), batch_size=SAE_BATCH, shuffle=True)

# --------------------------
# Train loop
# --------------------------
print("\n=== Training Deep SwishGLU SAE ===")
for ep in range(1, SAE_EPOCHS + 1):
    model.train()
    total = 0.0
    for (xb,) in dl:
        xb = xb.to(DEVICE)
        x_hat, z = model(xb)
        loss = mse(x_hat, xb) + SAE_L1 * torch.mean(torch.abs(z))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total += loss.item() * xb.size(0)

    # normalize decoder columns
    if NORMALIZE_DECODER:
        with torch.no_grad():
            for layer in [l for l in model.dec if isinstance(l, nn.Linear)]:
                W = layer.weight.data
                norms = torch.norm(W, dim=1, keepdim=True) + 1e-8
                layer.weight.data = W / norms

    if ep % 20 == 0 or ep == SAE_EPOCHS:
        print(f"Epoch {ep:3d}/{SAE_EPOCHS} | loss={total/len(dl.dataset):.6f}")

# --------------------------
# Diagnostics
# --------------------------
@torch.no_grad()
def encode_decode(X):
    model.eval()
    xb = torch.from_numpy(X.astype(np.float32)).to(DEVICE)
    x_hat, z = model(xb)
    return x_hat.cpu().numpy(), z.cpu().numpy()

Xhat, Z = encode_decode(X_std)
recon_r2 = r2_score(X_std, Xhat, multioutput="variance_weighted")

ridge = Ridge(alpha=1.0, fit_intercept=False)
ridge.fit(Z, X_std)
r2_lin = r2_score(X_std, ridge.predict(Z), multioutput="variance_weighted")

print(f"\nReconstruction R²: {recon_r2:.4f}")
print(f"Linear map R²(Z→X): {r2_lin:.4f}")

frac_nonzero = (Z > 0).mean(axis=0)
l0 = (Z > 0).sum(axis=1)
print(f"Active units per image: {l0.mean():.1f} ± {l0.std():.1f}")
print(f"Mean fraction nonzero across units: {frac_nonzero.mean():.3f}")

# --------------------------
# Save artifacts
# --------------------------
np.save(os.path.join(OUT_DIR, "Z.npy"), Z)
torch.save(model.state_dict(), os.path.join(OUT_DIR, "deep_swiglu_sae.pt"))
with open(os.path.join(OUT_DIR, "scaler.pkl"), "wb") as f:
    pickle.dump(scaler, f)
pd.DataFrame([{
    "recon_r2": float(recon_r2),
    "r2_lin": float(r2_lin),
    "mean_active_per_img": float(l0.mean()),
    "mean_frac_nonzero": float(frac_nonzero.mean())
}]).to_csv(os.path.join(OUT_DIR, "diagnostics.csv"), index=False)

print(f"\nSaved results → {OUT_DIR}")
