"""
Stage B fine-tuning for DLS (MultipathModelBL).

Reads the train/valid CSVs produced by build_finetune_matched.py:
  - feat128_path             absolute path to {id}.npy of shape (5, 128)
  - 8 biomarker cols          age, education, bmi, phist, fhist, smo_status, quit_time, pkyr
  - with_image, with_marker   per-sample modality flags
  - lung_cancer (or any binary label via --label_col)

Each sample is routed into one of three model paths based on its flags:
  - both modalities  -> bothPred (joint head)
  - image only       -> imgPred  (image-only head, exercised when biomarkers absent)
  - biomarkers only  -> clicPred (clinical-only head, exercised when image absent)

For "both" samples we also train the image-only and clinical-only heads on the
joint inputs (auxiliary losses, weighted 0.5x) so all three heads get gradients
even when most data has both modalities.

Outputs go to --output_dir:
  best.pth              best-val-AUC checkpoint
  val_pred_best.csv     per-sample predictions at the best epoch
  train_log.csv         per-epoch metrics
  tb/                   tensorboard logs
"""

import os, sys, time, argparse, json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
from model import MultipathModelBL

BIOMARKERS = ["age", "education", "bmi", "phist", "fhist",
              "smo_status", "quit_time", "pkyr"]

NODULE_SIZE_DENOM = 30.0   # normalize nodule_size by dividing by this

CLINICAL_MODES = ("all_biomarkers", "image_only", "nodule_size_only",
                  "image_plus_optional_size")


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class DLSFinetuneDataset(Dataset):
    """
    clinical_mode controls how the 12-dim biomarker vector is filled and which
    DLS path each sample routes through:

      'all_biomarkers' (default, original DLS):
          biomarker[0] = with_image, [1] = with_marker (from CSV)
          biomarker[2:10] = 8 DLS biomarkers (from CSV)
          biomarker[10:12] = 0 (plco/kaggle placeholders)

      'image_only':
          biomarker[0] = 1, [1] = 0  -> routes every sample through imgOnly path
          biomarker[2:12] = 0        -> unused
          Trains only imgPred. Used for the image-only ablation.

      'nodule_size_only':
          biomarker[0] = 1, [1] = 1  -> routes every sample through 'both' path
          biomarker[2] = nodule_size / NODULE_SIZE_DENOM (0 if NaN/missing)
          biomarker[3:12] = 0
          Trains bothPred (joint head). Used for the image+size deployment model.

      'image_plus_optional_size' (the deployment model):
          Mirrors the real application, where each scan is EITHER image-only OR
          image + a measured max nodule size. Routing is per-sample:
            - size present -> [0]=1, [1]=1, [2]=size/DENOM  -> 'both' path (bothPred)
            - size absent  -> [0]=1, [1]=0                  -> imgOnly path (imgPred)
          A genuinely-missing size is NEVER encoded as size=0; it routes through
          the image-only head, which structurally ignores the clinical vector,
          so there is no 0-mm placeholder ambiguity.
          With size_dropout_p > 0 and apply_dropout=True (training only), a random
          fraction of size-present samples are demoted to image-only each epoch
          (modality dropout). This manufactures genuine image-only training signal
          so the shared trunk + image-only head stay deployment-grade even when the
          training cohort almost always has a size. One model serves both modes;
          at inference the head is chosen deterministically by size availability.
    """
    def __init__(self, csv_path, label_col="lung_cancer",
                 clinical_mode="all_biomarkers",
                 size_dropout_p=0.0, apply_dropout=False):
        assert clinical_mode in CLINICAL_MODES, f"clinical_mode must be one of {CLINICAL_MODES}"
        df = pd.read_csv(csv_path, low_memory=False)
        for c in BIOMARKERS:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)
            else:
                df[c] = 0.0
        if "nodule_size" in df.columns:
            raw_size = pd.to_numeric(df["nodule_size"], errors="coerce")
            df["_has_size"] = (raw_size.notnull() & (raw_size > 0))
            df["nodule_size"] = raw_size.fillna(0.0)
        else:
            df["_has_size"] = False
            df["nodule_size"] = 0.0
        df["with_image"]  = df.get("with_image", 1)
        df["with_marker"] = df.get("with_marker", 0)
        df["with_image"]  = pd.to_numeric(df["with_image"], errors="coerce").fillna(0).astype(int)
        df["with_marker"] = pd.to_numeric(df["with_marker"], errors="coerce").fillna(0).astype(int)
        df[label_col]     = pd.to_numeric(df[label_col], errors="coerce").fillna(0).astype(int)
        self.df = df.reset_index(drop=True)
        self.label_col = label_col
        self.clinical_mode = clinical_mode
        self.size_dropout_p = float(size_dropout_p)
        self.apply_dropout = bool(apply_dropout)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        feat = np.load(row["feat128_path"]).astype(np.float32)            # (5, 128)
        biomarker = np.zeros(12, dtype=np.float32)
        if self.clinical_mode == "image_only":
            biomarker[0] = 1.0    # with_image
            biomarker[1] = 0.0    # with_marker -> imgOnly routing
        elif self.clinical_mode == "nodule_size_only":
            biomarker[0] = 1.0    # with_image
            biomarker[1] = 1.0    # with_marker -> 'both' routing
            biomarker[2] = float(row["nodule_size"]) / NODULE_SIZE_DENOM
        elif self.clinical_mode == "image_plus_optional_size":
            biomarker[0] = 1.0    # with_image
            has_size = bool(row["_has_size"])
            if self.apply_dropout and has_size and np.random.rand() < self.size_dropout_p:
                has_size = False  # modality dropout: demote to image-only this epoch
            if has_size:
                biomarker[1] = 1.0    # with_marker -> 'both' routing
                biomarker[2] = float(row["nodule_size"]) / NODULE_SIZE_DENOM
            else:
                biomarker[1] = 0.0    # -> imgOnly routing (clinical vector ignored)
        else:  # all_biomarkers (original convention)
            biomarker[0]    = float(row["with_image"])
            biomarker[1]    = float(row["with_marker"])
            biomarker[2:10] = row[BIOMARKERS].values.astype(np.float32)
        label = float(row[self.label_col])
        return (
            torch.from_numpy(feat),
            torch.from_numpy(biomarker),
            torch.tensor(label, dtype=torch.float32),
        )


# ---------------------------------------------------------------------------
# Forward / loss with three-path routing
# ---------------------------------------------------------------------------
def weighted_bce(pred, target, pos_weight):
    """BCE with per-sample weighting for class imbalance. Model outputs prob (sigmoid)."""
    eps = 1e-7
    pred = pred.clamp(eps, 1 - eps)
    loss = -(pos_weight * target * torch.log(pred) + (1 - target) * torch.log(1 - pred))
    return loss.mean()


def split_batch_by_modality(feats, biomarkers):
    """Return masks for the three sub-batches based on with_image/with_marker."""
    wi = biomarkers[:, 0].bool()
    wm = biomarkers[:, 1].bool()
    return (wi & wm), (wi & ~wm), (~wi & wm)


def forward_and_loss(model, feats, biomarkers, labels, pos_weight, device,
                     aux_weight=0.5):
    """Forward each sub-batch through the appropriate path, sum weighted losses.
    Returns: total loss, prediction tensor, label tensor (joint pred per sample
    where possible, falling back to imgPred or clicPred for partial-modality)."""
    both_mask, img_only_mask, fac_only_mask = split_batch_by_modality(feats, biomarkers)

    total_loss = torch.tensor(0.0, device=device)
    n_total = 0
    all_pred, all_lbl = [], []

    Z3D = torch.zeros((0, 5, 128), device=device, dtype=feats.dtype)
    Z1D = torch.zeros((0, 12), device=device, dtype=biomarkers.dtype)

    # --- both-modality samples (main training signal) ---
    if both_mask.any():
        f, b, y = feats[both_mask], biomarkers[both_mask], labels[both_mask]
        out = model(Z3D, Z1D, f, b)
        # MultipathModelBL.forward returns:
        #   (imgPred, clicPred, bothImgPred, bothClicPred, bothPred)
        _, _, bothImgPred, bothClicPred, bothPred = out
        loss = (weighted_bce(bothPred, y, pos_weight)
                + aux_weight * weighted_bce(bothImgPred.squeeze(-1), y, pos_weight)
                + aux_weight * weighted_bce(bothClicPred.squeeze(-1), y, pos_weight))
        total_loss = total_loss + loss * y.shape[0]
        n_total += y.shape[0]
        all_pred.append(bothPred.detach())
        all_lbl.append(y.detach())

    # --- image-only samples ---
    if img_only_mask.any():
        f, y = feats[img_only_mask], labels[img_only_mask]
        out = model(f, Z1D, Z3D, Z1D)
        # when both == 0: forward returns (imgPred, clicPred, 0, 0, 0)
        imgPred, _, _, _, _ = out
        loss = weighted_bce(imgPred, y, pos_weight)
        total_loss = total_loss + loss * y.shape[0]
        n_total += y.shape[0]
        all_pred.append(imgPred.detach())
        all_lbl.append(y.detach())

    # --- biomarker-only samples ---
    if fac_only_mask.any():
        b, y = biomarkers[fac_only_mask], labels[fac_only_mask]
        out = model(Z3D, b, Z3D, Z1D)
        _, clicPred, _, _, _ = out
        loss = weighted_bce(clicPred, y, pos_weight)
        total_loss = total_loss + loss * y.shape[0]
        n_total += y.shape[0]
        all_pred.append(clicPred.detach())
        all_lbl.append(y.detach())

    if n_total == 0:
        return None, None, None
    avg_loss = total_loss / n_total
    preds  = torch.cat(all_pred)
    lbls   = torch.cat(all_lbl)
    return avg_loss, preds, lbls


def evaluate(model, loader, pos_weight, device, aux_weight, route_override=None):
    """Run one no-grad pass and return (loss, auc, lbls, preds).

    route_override:
      None         -> route each sample as its with_image/with_marker flags say.
      "image_only" -> force every sample through the image-only head (zero the
                      with_marker flag). Used to measure the image-only
                      deployment mode of an image_plus_optional_size model.
    """
    model.eval()
    loss_sum, n_seen = 0.0, 0
    preds_all, lbls_all = [], []
    with torch.no_grad():
        for feats, biomarkers, labels in loader:
            feats      = feats.to(device, non_blocking=True)
            biomarkers = biomarkers.to(device, non_blocking=True).clone()
            labels     = labels.to(device, non_blocking=True)
            if route_override == "image_only":
                biomarkers[:, 1] = 0.0
            loss, preds, lbls = forward_and_loss(model, feats, biomarkers, labels,
                                                 pos_weight, device, aux_weight)
            if loss is None:
                continue
            n = preds.shape[0]
            loss_sum += loss.item() * n
            n_seen   += n
            preds_all.append(preds.cpu().numpy())
            lbls_all.append(lbls.cpu().numpy())
    if n_seen == 0:
        return float("nan"), float("nan"), np.zeros(0), np.zeros(0)
    lbls = np.concatenate(lbls_all)
    preds = np.concatenate(preds_all)
    try:
        auc = roc_auc_score(lbls, preds)
    except ValueError:
        auc = float("nan")
    return loss_sum / n_seen, auc, lbls, preds


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--train_csv",     required=True)
    p.add_argument("--valid_csv",     required=True)
    p.add_argument("--output_dir",    required=True)
    p.add_argument("--pretrain_pth",  default=None,
                   help="Initialize from this checkpoint; omit to train from scratch.")
    p.add_argument("--label_col",     default="lung_cancer")
    p.add_argument("--batch_size",    type=int, default=256)
    p.add_argument("--lr",            type=float, default=1e-4)
    p.add_argument("--weight_decay",  type=float, default=1e-4)
    p.add_argument("--epochs",        type=int, default=50)
    p.add_argument("--patience",      type=int, default=10,
                   help="Early stop if val AUC doesn't improve for this many epochs.")
    p.add_argument("--num_workers",   type=int, default=4)
    p.add_argument("--aux_weight",    type=float, default=0.5,
                   help="Weight on imgPred/clicPred auxiliary losses for both-modality samples.")
    p.add_argument("--clinical_mode", choices=list(CLINICAL_MODES), default="all_biomarkers",
                   help="How to build the per-sample clinical input (see DLSFinetuneDataset docstring).")
    p.add_argument("--size_dropout_p", type=float, default=0.3,
                   help="image_plus_optional_size only: probability of demoting a "
                        "size-present TRAIN sample to image-only each epoch (modality "
                        "dropout). Ignored by other modes and never applied to valid.")
    p.add_argument("--seed",          type=int, default=42)
    args = p.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # save args for reproducibility
    with open(os.path.join(args.output_dir, "args.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    # --- data ---
    print(f"Clinical mode: {args.clinical_mode}")
    train_ds = DLSFinetuneDataset(args.train_csv, args.label_col, args.clinical_mode,
                                  size_dropout_p=args.size_dropout_p, apply_dropout=True)
    valid_ds = DLSFinetuneDataset(args.valid_csv, args.label_col, args.clinical_mode,
                                  size_dropout_p=0.0, apply_dropout=False)
    print(f"Train: {len(train_ds)} samples  |  Valid: {len(valid_ds)} samples")
    if args.clinical_mode == "image_plus_optional_size":
        n_size = int(train_ds.df["_has_size"].sum())
        print(f"  train size-present: {n_size}/{len(train_ds)}  "
              f"(size_dropout_p={args.size_dropout_p})")

    pos = int((train_ds.df[args.label_col] == 1).sum())
    neg = len(train_ds) - pos
    pos_weight = float(neg) / max(pos, 1)
    print(f"Train class balance: pos={pos}  neg={neg}  pos_weight={pos_weight:.3f}")
    pos_weight_t = torch.tensor(pos_weight, device=device, dtype=torch.float32)

    # --- modality breakdown ---
    for name, ds in [("train", train_ds), ("valid", valid_ds)]:
        both = ((ds.df["with_image"] == 1) & (ds.df["with_marker"] == 1)).sum()
        img_only = ((ds.df["with_image"] == 1) & (ds.df["with_marker"] == 0)).sum()
        fac_only = ((ds.df["with_image"] == 0) & (ds.df["with_marker"] == 1)).sum()
        print(f"  {name}: both={both}  img_only={img_only}  fac_only={fac_only}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True)
    valid_loader = DataLoader(valid_ds, batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers, pin_memory=True)

    # --- model ---
    model = MultipathModelBL(1).to(device)
    if args.pretrain_pth:
        sd = torch.load(args.pretrain_pth, map_location="cpu", weights_only=False)
        if isinstance(sd, dict) and "state_dict" in sd:
            sd = sd["state_dict"]
        missing, unexpected = model.load_state_dict(sd, strict=False)
        print(f"Loaded pretrain: {args.pretrain_pth}")
        if missing:    print(f"  missing keys ({len(missing)}): {missing[:3]}...")
        if unexpected: print(f"  unexpected keys ({len(unexpected)}): {unexpected[:3]}...")
    else:
        print("Training from scratch (no pretrain).")

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # --- training loop ---
    try:
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(os.path.join(args.output_dir, "tb"))
    except ImportError:
        writer = None
        print("Note: tensorboard not available — skipping TB logging.")

    log_rows = []
    best_val_auc = -1.0
    best_epoch   = -1
    no_improve   = 0

    for epoch in range(args.epochs):
        # ----- train -----
        model.train()
        t0 = time.time()
        loss_sum, n_seen = 0.0, 0
        preds_all, lbls_all = [], []
        for feats, biomarkers, labels in train_loader:
            feats      = feats.to(device, non_blocking=True)
            biomarkers = biomarkers.to(device, non_blocking=True)
            labels     = labels.to(device, non_blocking=True)
            optimizer.zero_grad()
            loss, preds, lbls = forward_and_loss(model, feats, biomarkers, labels,
                                                 pos_weight_t, device, args.aux_weight)
            if loss is None:
                continue
            loss.backward()
            optimizer.step()
            n = preds.shape[0]
            loss_sum += loss.item() * n
            n_seen   += n
            preds_all.append(preds.cpu().numpy())
            lbls_all.append(lbls.cpu().numpy())
        train_loss = loss_sum / max(n_seen, 1)
        try:
            train_auc = roc_auc_score(np.concatenate(lbls_all), np.concatenate(preds_all))
        except ValueError:
            train_auc = float("nan")

        # ----- validation -----
        # "natural" routing reflects the deployment mix (size-present -> joint head,
        # size-absent -> image-only head). For image_plus_optional_size we also
        # measure the forced image-only mode, and select the checkpoint on the mean
        # of the two so the single model stays strong in BOTH deployment cases.
        val_loss, val_auc, val_lbls, val_preds = evaluate(
            model, valid_loader, pos_weight_t, device, args.aux_weight, route_override=None)
        if args.clinical_mode == "image_plus_optional_size":
            _, val_auc_img, _, _ = evaluate(
                model, valid_loader, pos_weight_t, device, args.aux_weight,
                route_override="image_only")
            sel_metric = float(np.nanmean([val_auc, val_auc_img]))
        else:
            val_auc_img = float("nan")
            sel_metric = val_auc

        scheduler.step()
        elapsed = time.time() - t0

        img_str = f" val_auc_img={val_auc_img:.4f}" if not np.isnan(val_auc_img) else ""
        print(f"Ep {epoch+1:3d}/{args.epochs}  "
              f"train_loss={train_loss:.4f} train_auc={train_auc:.4f}  "
              f"val_loss={val_loss:.4f} val_auc={val_auc:.4f}{img_str}  "
              f"({elapsed:.1f}s)")

        log_rows.append(dict(epoch=epoch+1, train_loss=train_loss, train_auc=train_auc,
                             val_loss=val_loss, val_auc=val_auc, val_auc_img=val_auc_img,
                             sel_metric=sel_metric,
                             lr=optimizer.param_groups[0]["lr"], time_s=elapsed))
        if writer:
            writer.add_scalar("train/loss", train_loss, epoch)
            writer.add_scalar("train/auc",  train_auc,  epoch)
            writer.add_scalar("val/loss",   val_loss,   epoch)
            writer.add_scalar("val/auc",    val_auc,    epoch)
            if not np.isnan(val_auc_img):
                writer.add_scalar("val/auc_image_only", val_auc_img, epoch)
                writer.add_scalar("val/sel_metric", sel_metric, epoch)
            writer.add_scalar("lr", optimizer.param_groups[0]["lr"], epoch)

        # ----- checkpoint best (on sel_metric) -----
        if sel_metric > best_val_auc:
            best_val_auc = sel_metric
            best_epoch   = epoch + 1
            no_improve   = 0
            torch.save(
                {"state_dict": model.state_dict(),
                 "epoch": epoch + 1, "val_auc": val_auc, "val_auc_img": val_auc_img,
                 "sel_metric": sel_metric, "args": vars(args)},
                os.path.join(args.output_dir, "best.pth"),
            )
            pd.DataFrame({"label": val_lbls, "prob": val_preds}).to_csv(
                os.path.join(args.output_dir, "val_pred_best.csv"), index=False)
        else:
            no_improve += 1
        if no_improve >= args.patience:
            print(f"Early stop: no val improvement for {args.patience} epochs.")
            break

    # ----- finalize -----
    pd.DataFrame(log_rows).to_csv(os.path.join(args.output_dir, "train_log.csv"), index=False)
    if writer: writer.close()
    print()
    print(f"DONE. Best selection metric: {best_val_auc:.4f} at epoch {best_epoch}")
    print(f"  checkpoint: {os.path.join(args.output_dir, 'best.pth')}")
    print(f"  val preds:  {os.path.join(args.output_dir, 'val_pred_best.csv')}")
    print(f"  train log:  {os.path.join(args.output_dir, 'train_log.csv')}")


if __name__ == "__main__":
    main()
