"""
Match the 4-30mm-nodule fine-tuning spreadsheet against existing Stage A
features.

The spreadsheet
(compatible_cohorts_finetune_split_lung_cancer_labeled_nodule_size_4_to_30mm_*.xlsx)
already has its own train/validation split.  We derive Stage A ids per row
using the same unharmonized->harmonized convention as build_holdout_matched.py
and produce two DLS-ready CSVs that step4_train.py can ingest directly.

Outputs (in cohorts/finetune_harmonized/):
  - biodesix_finetune_train_4to30.csv
  - biodesix_finetune_valid_4to30.csv
  - biodesix_finetune_4to30_matched_all.csv   (audit)
"""

import os, sys
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SPREADSHEET = os.environ.get(
    "NODULE_4TO30_XLSX",
    os.path.join(
        SCRIPT_DIR,
        "compatible_cohorts_finetune_split_lung_cancer_labeled_nodule_size_4_to_30mm_20260512.xlsx",
    ),
)
FEAT_ROOT = os.environ.get(
    "BIODESIX_FEAT_ROOT",
    "/valiant02/masi/zuol1/projects/biodesix/DeepLungScreen/data/finetune_harmonized",
)
TRAIN_CSV = os.path.join(SCRIPT_DIR, "biodesix_finetune_train_4to30.csv")
VALID_CSV = os.path.join(SCRIPT_DIR, "biodesix_finetune_valid_4to30.csv")
MERGED_CSV = os.path.join(SCRIPT_DIR, "biodesix_finetune_4to30_matched_all.csv")

# Spreadsheet cohort_name -> Stage A folder.
COHORT_FOLDER_OVERRIDES = {"tho1496": "1496"}

# Suffix appended to the unharmonized basename to get the harmonized basename
# (which is also the Stage A id).
HARMONIZED_SUFFIX = {
    "nlst":     "_resampled_fov_extended_orig_res",
    "mcl":      "_harmonized_fov_extended_orig_res",
    "bronch":   "_harmonized_fov_extended_orig_res",
    "nodulevu": "_harmonized_fov_extended_orig_res",
    "tho1496":  "_harmonized_fov_extended_orig_res",
    "vlsp":     "_harmonized_fov_extended_orig_res",
    "Reliant1": "_harmonized_fov_extended_orig_res",
    "Reliant2": "_harmonized_fov_extended_orig_res",
    "Veritas":  "_harmonized_fov_extended_orig_res",
}

BIOMARKER_RENAME = {
    "personal_cancer_history": "phist",
    "family_cancer_history":   "fhist",
    "smoking_status":          "smo_status",
    "years_since_quitting":    "quit_time",
    "pack_years":              "pkyr",
}
DLS_BIOMARKERS = ["age", "education", "bmi",
                  "phist", "fhist", "smo_status", "quit_time", "pkyr"]


def strip_nii(fname):
    if fname.endswith(".nii.gz"): return fname[:-7]
    if fname.endswith(".nii"):    return fname[:-4]
    return fname


def derive_id(row):
    if pd.isna(row["fpath"]):
        return None
    base = strip_nii(os.path.basename(str(row["fpath"])))
    suffix = HARMONIZED_SUFFIX.get(row["cohort_name"], "_harmonized_fov_extended_orig_res")
    return base + suffix


def to_folder(cohort_name):
    if pd.isna(cohort_name): return None
    return COHORT_FOLDER_OVERRIDES.get(str(cohort_name), str(cohort_name).lower())


def main():
    if not os.path.isfile(SPREADSHEET):
        sys.exit(f"ERROR: spreadsheet not found: {SPREADSHEET}\n"
                 f"       set NODULE_4TO30_XLSX=/path/to/file.xlsx if it lives elsewhere.")
    if not os.path.isdir(FEAT_ROOT):
        sys.exit(f"ERROR: feat root not found: {FEAT_ROOT}")

    print(f"Reading {SPREADSHEET}")
    df = pd.read_excel(SPREADSHEET, sheet_name="data")
    print(f"  rows: {len(df)}")

    df["id"]            = df.apply(derive_id, axis=1)
    df["cohort_folder"] = df["cohort_name"].apply(to_folder)
    df["feat128_path"]  = df.apply(
        lambda r: (
            os.path.join(FEAT_ROOT, r["cohort_folder"], "feat128", f"{r['id']}.npy")
            if pd.notna(r["id"]) and pd.notna(r["cohort_folder"])
            else None
        ),
        axis=1,
    )
    df["feat_ready"] = df["feat128_path"].apply(
        lambda p: os.path.isfile(p) if isinstance(p, str) else False
    )

    df_renamed = df.rename(columns=BIOMARKER_RENAME)
    df_renamed["with_image"]  = df_renamed["feat_ready"].astype(int)
    df_renamed["with_marker"] = df_renamed[DLS_BIOMARKERS].notnull().all(axis=1).astype(int)

    print()
    print("Match rate by cohort:")
    rep = df.groupby("cohort_name").agg(rows=("id", "count"),
                                        feat_ready=("feat_ready", "sum"))
    rep["pct"] = (rep["feat_ready"] / rep["rows"] * 100).round(1)
    print(rep.to_string())
    print()

    print("Train/valid x label composition (feat_ready only):")
    ready = df_renamed[df_renamed["feat_ready"]]
    print(pd.crosstab(ready["train_validation_split"], ready["lung_cancer"],
                      margins=True, margins_name="All"))
    print()

    print("nodule_size distribution (feat_ready):")
    print(ready["nodule_size"].describe())
    print()

    # write outputs
    df_renamed.to_csv(MERGED_CSV, index=False)
    print(f"Wrote audit CSV: {MERGED_CSV}  ({len(df_renamed)} rows)")

    clean = df_renamed[df_renamed["feat_ready"]].copy()
    train_df = clean[clean["train_validation_split"] == "train"]
    valid_df = clean[clean["train_validation_split"] == "validation"]
    train_df.to_csv(TRAIN_CSV, index=False)
    valid_df.to_csv(VALID_CSV, index=False)
    print(f"Wrote train: {len(train_df)} rows -> {TRAIN_CSV}")
    print(f"Wrote valid: {len(valid_df)} rows -> {VALID_CSV}")


if __name__ == "__main__":
    main()
