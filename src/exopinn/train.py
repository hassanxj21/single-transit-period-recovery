"""Training loop for the hybrid CNN-PINN."""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch

from .model import HybridCNNPINN, Normalizer, gaussian_nll, physics_loss


@dataclass
class TrainConfig:
    # The 300-epoch run peaked at epoch 15 and then diverged: train NLL kept
    # falling to -1.42 while val NLL blew up to +6.07, because the Gaussian NLL
    # lets the model shrink sigma toward zero on training points. Early stopping
    # plus a sigma floor is the fix; 60 epochs is already generous.
    epochs: int = 60
    batch_size: int = 512
    lr: float = 1.5e-3
    weight_decay: float = 1e-4
    lambda_physics: float = 0.1
    val_fraction: float = 0.15
    seed: int = 0
    device: str = "cpu"
    physics_warmup: int = 5      # epochs before the physics term is switched on
    patience: int = 12           # stop after this many epochs with no val improvement
    min_log_sigma: float = -2.5  # floor on predicted log sigma (normalised units)
    use_lightcurve: bool = True  # False -> scalars-only ablation

    def to_dict(self):
        return asdict(self)


def _split(n, val_fraction, seed):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    n_val = int(n * val_fraction)
    return idx[n_val:], idx[:n_val]


def train(data, cfg: TrainConfig | None = None, out_dir: str | Path = "results/models", verbose=True):
    cfg = cfg or TrainConfig()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(cfg.seed)
    dev = torch.device(cfg.device)

    lc, scalars, period = data["lc"], data["scalars"], data["period"]
    tr, va = _split(len(period), cfg.val_fraction, cfg.seed)

    norm = Normalizer().fit(lc[tr], scalars[tr], period[tr])

    def to_t(idx):
        return (
            torch.tensor(norm.transform_lc(lc[idx]), device=dev).unsqueeze(1),
            torch.tensor(norm.transform_scalars(scalars[idx]), device=dev),
            torch.tensor(norm.transform_y(period[idx]), device=dev).unsqueeze(1),
            torch.tensor(scalars[idx], dtype=torch.float32, device=dev),  # raw, for physics
        )

    LC_tr, SC_tr, Y_tr, RAW_tr = to_t(tr)
    LC_va, SC_va, Y_va, RAW_va = to_t(va)

    model = HybridCNNPINN(n_points=lc.shape[1], n_scalars=scalars.shape[1],
                          use_lightcurve=cfg.use_lightcurve,
                          min_log_sigma=cfg.min_log_sigma).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.epochs, eta_min=cfg.lr * 0.02)

    n_tr = len(tr)
    history = {"epoch": [], "train_total": [], "train_data": [], "train_physics": [],
               "val_nll": [], "val_median_pct": [], "lr": []}
    best = (np.inf, None, -1)
    since_improved = 0
    t_start = time.time()

    for epoch in range(cfg.epochs):
        model.train()
        perm = torch.randperm(n_tr, device=dev)
        lam = cfg.lambda_physics if epoch >= cfg.physics_warmup else 0.0
        acc = np.zeros(3)
        n_batches = 0

        for s in range(0, n_tr, cfg.batch_size):
            idx = perm[s : s + cfg.batch_size]
            mu, log_sigma = model(LC_tr[idx], SC_tr[idx])

            l_data = gaussian_nll(mu, log_sigma, Y_tr[idx])

            if lam > 0:
                p_pred = torch.pow(10.0, norm.inverse_y(mu))
                raw = RAW_tr[idx]
                l_phys = physics_loss(p_pred, raw[:, 0:1], raw[:, 2:3], raw[:, 3:4], raw[:, 1:2])
            else:
                l_phys = torch.zeros((), device=dev)

            loss = l_data + lam * l_phys
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()

            acc += [loss.item(), l_data.item(), l_phys.detach().item()]
            n_batches += 1

        sched.step()
        acc /= max(n_batches, 1)

        model.eval()
        with torch.no_grad():
            mu_v, ls_v = model(LC_va, SC_va)
            val_nll = gaussian_nll(mu_v, ls_v, Y_va).item()
            p_pred = 10.0 ** norm.inverse_y(mu_v.cpu().numpy().ravel())
            p_true = period[va]
            pct = np.abs(p_pred - p_true) / p_true * 100.0
            val_med = float(np.median(pct))

        history["epoch"].append(epoch)
        history["train_total"].append(acc[0])
        history["train_data"].append(acc[1])
        history["train_physics"].append(acc[2])
        history["val_nll"].append(val_nll)
        history["val_median_pct"].append(val_med)
        history["lr"].append(opt.param_groups[0]["lr"])

        if val_nll < best[0] - 1e-5 and epoch >= cfg.physics_warmup:
            best = (val_nll, {k: v.detach().clone() for k, v in model.state_dict().items()}, epoch)
            since_improved = 0
            torch.save({"model_state": model.state_dict(),
                        "normalizer": norm.state_dict(),
                        "train_config": cfg.to_dict(),
                        "n_points": int(lc.shape[1]), "n_scalars": int(scalars.shape[1]),
                        "val_indices": va, "epoch": epoch},
                       out_dir / "cnn_pinn.pt")
        elif epoch >= cfg.physics_warmup:
            since_improved += 1

        if since_improved >= cfg.patience:
            if verbose:
                print(f"  early stop at epoch {epoch}: no val improvement for "
                      f"{cfg.patience} epochs (best was epoch {best[2]})")
            break

        if verbose and (epoch % 5 == 0 or epoch == cfg.epochs - 1):
            print(f"  epoch {epoch:4d}  total {acc[0]:8.4f}  data {acc[1]:8.4f}  "
                  f"phys {acc[2]:8.4f}  val_nll {val_nll:8.4f}  val_med_err {val_med:6.1f}%")

    if best[1] is not None:
        model.load_state_dict(best[1])

    ckpt = {
        "model_state": model.state_dict(),
        "normalizer": norm.state_dict(),
        "train_config": cfg.to_dict(),
        "n_points": int(lc.shape[1]),
        "n_scalars": int(scalars.shape[1]),
        "val_indices": va,
        "best_epoch": best[2],
    }
    torch.save(ckpt, out_dir / "cnn_pinn.pt")
    (out_dir / "history.json").write_text(json.dumps(history, indent=2))

    if verbose:
        print(f"[train] done in {time.time() - t_start:.1f}s  "
              f"best val NLL {best[0]:.4f} at epoch {best[2]}")
        print(f"[train] saved {out_dir / 'cnn_pinn.pt'}")
    return model, norm, history, (tr, va)


def load_model(path, device="cpu"):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    tc = ckpt.get("train_config", {})
    model = HybridCNNPINN(n_points=ckpt["n_points"], n_scalars=ckpt["n_scalars"],
                          use_lightcurve=tc.get("use_lightcurve", True),
                          min_log_sigma=tc.get("min_log_sigma", -6.0)).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, Normalizer.from_state_dict(ckpt["normalizer"]), ckpt


@torch.no_grad()
def predict(model, norm, lc, scalars, device="cpu", n_sigma=1.0):
    """Return (period_days, lo, hi) where lo/hi are the +/- n_sigma band in log space."""
    lc = np.atleast_2d(np.asarray(lc, np.float32))
    scalars = np.atleast_2d(np.asarray(scalars, np.float32))
    lc_t = torch.tensor(norm.transform_lc(lc), device=device).unsqueeze(1)
    sc_t = torch.tensor(norm.transform_scalars(scalars), device=device)
    mu, log_sigma = model(lc_t, sc_t)
    mu = mu.cpu().numpy().ravel()
    sig = np.exp(log_sigma.cpu().numpy().ravel())
    center = 10.0 ** norm.inverse_y(mu)
    lo = 10.0 ** norm.inverse_y(mu - n_sigma * sig)
    hi = 10.0 ** norm.inverse_y(mu + n_sigma * sig)
    return center, lo, hi
