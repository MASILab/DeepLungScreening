"""
Stage B inference: run a fine-tuned DLS model on a matched CSV (held-out, or
any other) and write per-sample predictions + summary metrics.

Inputs:
  --input_csv    DLS-ready CSV (output of build_holdout_matched.py /
                 build_finetune_matched.py); must include feat128_path,
                 with_image, with_marker, the 8 biomarkers, and a label col.
  --checkpoint   Path to the fine-tuned best.pth produced by step4_train.py.
  --output_csv   Where to write per-sample predictions.
  --label_col    Default 'lung_cancer'.  Pass '' to predict unlabeled data.

Outputs:
  - <output_csv>           per-row: pid, scandate, label, prob, pred_path
  - <output_csv>.metrics.json   AUC, AP, n_pos, n_neg, n_total (if labeled)

Run example:
    python3 4_co_learning/step4_predict.py \
        --input_csv  cohorts/finetune_harmonized/biodesix_holdout_matched.csv \
        --checkpoint data/finetune_harmonized/stageB/best.pth \
        --output_csv data/finetune_harmonized/stageB/holdout_pred.csv
"""

import os, sys, json, argparse
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, average_precision_score

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
from model import MultipathModelBL
from step4_train import (DLSFinetuneDataset, BIOMARKERS, CLINICAL_MODES,
                         split_batch_by_modality)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input_csv",   required=True)
    p.add_argument("--checkpoint",  required=True)
    p.add_argument("--output_csv",  required=True)
    p.add_argument("--label_col",   default="lung_cancer",
                   help="Empty string disables metric computation.")
    p.add_argument("--clinical_mode", choices=list(CLINICAL_MODES), default="all_biomarkers",
                   help="Must match the clinical_mode used at training time.")
    p.add_argument("--batch_size",  type=int, default=256)
    p.add_argument("--num_workers", type=int, default=4)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Input:      {args.input_csv}")
    print(f"Checkpoint: {args.checkpoint}")

    # Drop rows that have no feat — they can't be scored anyway.
    raw_df = pd.read_csv(args.input_csv, low_memory=False)
    n_raw  = len(raw_df)
    raw_df = raw_df[raw_df.get("feat_ready", True).astype(bool)].reset_index(drop=True)
    print(f"Rows after dropping feat_ready=False: {len(raw_df)} / {n_raw}")

    tmp_csv = args.input_csv + ".filtered.csv"
    raw_df.to_csv(tmp_csv, index=False)

    # Reuse the training Dataset (which fills NaN biomarkers, etc.)
    label_col = args.label_col if args.label_col else "lung_cancer"
    if args.label_col == "":
        # If label is absent in the CSV, add a dummy so Dataset doesn't crash.
        if "lung_cancer" not in raw_df.columns:
            raw_df["lung_cancer"] = 0
            raw_df.to_csv(tmp_csv, index=False)
    print(f"Clinical mode: {args.clinical_mode}")
    ds = DLSFinetuneDataset(tmp_csv, label_col, args.clinical_mode)

    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, pin_memory=True)

    # Model + load fine-tuned weights
    model = MultipathModelBL(1).to(device)
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    sd = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing:    print(f"  WARN missing keys: {len(missing)}")
    if unexpected: print(f"  WARN unexpected:  {len(unexpected)}")
    if isinstance(ckpt, dict) and "val_auc" in ckpt:
        print(f"Checkpoint val_auc was: {ckpt['val_auc']:.4f} at epoch {ckpt.get('epoch', '?')}")
    model.eval()

    # Forward — same routing as the trainer.
    Z3D = torch.zeros((0, 5, 128), device=device)
    Z1D = torch.zeros((0, 12), device=device)
    all_probs = np.full(len(ds), np.nan, dtype=np.float32)

    sample_idx = 0
    with torch.no_grad():
        for feats, biomarkers, _labels in loader:
            B = feats.shape[0]
            feats      = feats.to(device, non_blocking=True)
            biomarkers = biomarkers.to(device, non_blocking=True)

            both_m, img_m, fac_m = split_batch_by_modality(feats, biomarkers)
            batch_probs = torch.zeros(B, device=device)

            if both_m.any():
                f, b = feats[both_m], biomarkers[both_m]
                out = model(Z3D, Z1D, f, b)
                _, _, _, _, bothPred = out
                batch_probs[both_m] = bothPred
            if img_m.any():
                f = feats[img_m]
                out = model(f, Z1D, Z3D, Z1D)
                imgPred, _, _, _, _ = out
                batch_probs[img_m] = imgPred
            if fac_m.any():
                b = biomarkers[fac_m]
                out = model(Z3D, b, Z3D, Z1D)
                _, clicPred, _, _, _ = out
                batch_probs[fac_m] = clicPred

            all_probs[sample_idx:sample_idx + B] = batch_probs.cpu().numpy()
            sample_idx += B

    os.remove(tmp_csv)

    # Assemble output table.
    out = pd.DataFrame({
        "cohort_name": raw_df["cohort_name"].values if "cohort_name" in raw_df else "",
        "subject_id":  raw_df["subject_id"].values  if "subject_id"  in raw_df else "",
        "session_date":raw_df["session_date"].values if "session_date" in raw_df else "",
        "id":          raw_df["id"].values         if "id" in raw_df else "",
        "feat128_path":raw_df["feat128_path"].values,
        "label":       raw_df[label_col].values if label_col in raw_df.columns else np.nan,
        "prob":        all_probs,
        "with_image":  raw_df["with_image"].values if "with_image" in raw_df else 1,
        "with_marker": raw_df["with_marker"].values if "with_marker" in raw_df else 1,
    })
    os.makedirs(os.path.dirname(os.path.abspath(args.output_csv)) or ".", exist_ok=True)
    out.to_csv(args.output_csv, index=False)
    print(f"Wrote {len(out)} predictions -> {args.output_csv}")

    # Metrics if labels are present
    metrics = {"n_total": int(len(out))}
    if args.label_col and label_col in raw_df.columns:
        valid = out["label"].notna() & out["prob"].notna()
        y = out.loc[valid, "label"].astype(int).values
        p = out.loc[valid, "prob"].values
        if len(np.unique(y)) > 1:
            metrics.update({
                "auc": float(roc_auc_score(y, p)),
                "ap":  float(average_precision_score(y, p)),
                "n_pos": int((y == 1).sum()),
                "n_neg": int((y == 0).sum()),
                "n_scored": int(len(y)),
            })
            print()
            print(f"Held-out metrics:  AUC={metrics['auc']:.4f}  AP={metrics['ap']:.4f}  "
                  f"n_pos={metrics['n_pos']}  n_neg={metrics['n_neg']}")
            # per-cohort breakdown
            if "cohort_name" in out.columns:
                print("Per-cohort AUC:")
                for c, sub in out.groupby("cohort_name"):
                    sv = sub["label"].notna() & sub["prob"].notna()
                    sy = sub.loc[sv, "label"].astype(int).values
                    sp = sub.loc[sv, "prob"].values
                    if len(np.unique(sy)) > 1:
                        a = roc_auc_score(sy, sp)
                        print(f"  {c:<10} n={len(sy):>4}  pos={int((sy==1).sum()):>3}  AUC={a:.4f}")
                    else:
                        print(f"  {c:<10} n={len(sy):>4}  (single class — AUC undefined)")
        else:
            print("WARN: only one class present in labels — AUC undefined.")

    with open(args.output_csv + ".metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Wrote metrics -> {args.output_csv}.metrics.json")


if __name__ == "__main__":
    main()
