"""
Match Stage A feature outputs against the curated train/valid QA spreadsheet,
and emit Stage-B-ready train / valid CSVs.

Inputs:
  - SPREADSHEET (xlsx): biodesix_train_valid_harmonized_20260512.xlsx
        Must contain columns:
            cohort_name, subject_id, session_date, harmonized_fpath,
            train_validation_split, QA_status, lung_cancer,
            age, education, bmi,
            personal_cancer_history, family_cancer_history,
            smoking_status, years_since_quitting, pack_years.
  - FEAT_ROOT: Stage A output root, with per-cohort feat128/{id}.npy files.

Outputs (in cohorts/finetune_harmonized/):
  - biodesix_finetune_matched_all.csv   every spreadsheet row + matching status
  - biodesix_finetune_train.csv         feat_ready & QA_status==yes & train
  - biodesix_finetune_valid.csv         feat_ready & QA_status==yes & validation

Run on the lab server (needs read access to /valiant02 feat128 dirs):
    python3 cohorts/finetune_harmonized/build_finetune_matched.py
"""

import os, sys, re
import pandas as pd

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
SPREADSHEET = os.environ.get(
    "BIODESIX_QA_XLSX",
    os.path.join(SCRIPT_DIR, "biodesix_train_valid_harmonized_20260512.xlsx"),
)
FEAT_ROOT   = os.environ.get(
    "BIODESIX_FEAT_ROOT",
    "/valiant02/masi/zuol1/projects/biodesix/DeepLungScreen/data/finetune_harmonized",
)
OUT_DIR     = SCRIPT_DIR

# Spreadsheet cohort name -> Stage A output folder (lowercase by default).
COHORT_FOLDER_OVERRIDES = {
    "tho1496": "1496",
}

# Spreadsheet biomarker column -> DLS-expected column name.
BIOMARKER_RENAME = {
    "personal_cancer_history": "phist",
    "family_cancer_history":   "fhist",
    "smoking_status":          "smo_status",
    "years_since_quitting":    "quit_time",
    "pack_years":              "pkyr",
    # age, education, bmi stay as-is
}

DLS_BIOMARKERS = ["age", "education", "bmi",
                  "phist", "fhist", "smo_status", "quit_time", "pkyr"]

SUFFIX_RE = re.compile(
    r"_harmonized_fov_extended_orig_res\.nii(?:\.gz)?$"
    r"|_resampled_fov_extended_orig_res\.nii(?:\.gz)?$"
    r"|_harmonized_fov_extended\.nii(?:\.gz)?$"
    r"|\.nii(?:\.gz)?$"
)


def derive_id(path):
    if pd.isna(path):
        return None
    fname = os.path.basename(str(path))
    m = SUFFIX_RE.search(fname)
    if m:
        return fname[:m.start()]
    return fname


def to_folder(cohort_name):
    if pd.isna(cohort_name):
        return None
    s = str(cohort_name)
    return COHORT_FOLDER_OVERRIDES.get(s, s.lower())


def main():
    if not os.path.isfile(SPREADSHEET):
        sys.exit(f"ERROR: spreadsheet not found: {SPREADSHEET}\n"
                 f"       set BIODESIX_QA_XLSX=/path/to/file.xlsx or place it at the default.")
    if not os.path.isdir(FEAT_ROOT):
        sys.exit(f"ERROR: feat root not found: {FEAT_ROOT}")

    print(f"Reading {SPREADSHEET}")
    df = pd.read_excel(SPREADSHEET)
    print(f"  rows: {len(df)}")

    # --- derive id + cohort folder + feat128 path ---
    df["id"]            = df["harmonized_fpath"].apply(derive_id)
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

    # --- rename biomarker columns to DLS conventions ---
    df_renamed = df.rename(columns=BIOMARKER_RENAME)

    # --- per-DLS multi-path flags ---
    df_renamed["with_image"]  = df_renamed["feat_ready"].astype(int)
    df_renamed["with_marker"] = df_renamed[DLS_BIOMARKERS].notnull().all(axis=1).astype(int)

    # --- reports ---
    print()
    print("Feat-ready by cohort (count of rows whose feat128/{id}.npy exists):")
    rep = df.groupby("cohort_name").agg(
        rows=("id", "count"),
        feat_ready=("feat_ready", "sum"),
    )
    rep["feat_pct"] = (rep["feat_ready"] / rep["rows"] * 100).round(1)
    print(rep.to_string())
    print()
    print("Feat-ready & QA=yes by cohort:")
    sub = df[df["QA_status"] == "yes"]
    rep2 = sub.groupby("cohort_name").agg(
        rows=("id", "count"),
        feat_ready=("feat_ready", "sum"),
    )
    rep2["feat_pct"] = (rep2["feat_ready"] / rep2["rows"] * 100).round(1)
    print(rep2.to_string())
    print()
    print("Final train/valid composition (QA=yes & feat_ready):")
    final = df[df["feat_ready"] & (df["QA_status"] == "yes")]
    print(pd.crosstab(
        final["train_validation_split"],
        final["lung_cancer"],
        margins=True, margins_name="All",
    ))
    print()
    print("With-marker breakdown of final cohort (1 = all 8 biomarkers present):")
    f_renamed = df_renamed[df_renamed["feat_ready"] & (df_renamed["QA_status"] == "yes")]
    print(pd.crosstab(
        f_renamed["train_validation_split"],
        f_renamed["with_marker"],
        margins=True, margins_name="All",
    ))

    # --- write the merged audit CSV (all rows, full info) ---
    merged_path = os.path.join(OUT_DIR, "biodesix_finetune_matched_all.csv")
    df_renamed.to_csv(merged_path, index=False)
    print()
    print(f"Wrote {len(df_renamed)} rows -> {merged_path}")

    # --- write filtered train/valid CSVs ---
    keep = df_renamed["feat_ready"] & (df_renamed["QA_status"] == "yes")
    clean = df_renamed[keep].copy()

    train_df = clean[clean["train_validation_split"] == "train"]
    valid_df = clean[clean["train_validation_split"] == "validation"]

    train_path = os.path.join(OUT_DIR, "biodesix_finetune_train.csv")
    valid_path = os.path.join(OUT_DIR, "biodesix_finetune_valid.csv")
    train_df.to_csv(train_path, index=False)
    valid_df.to_csv(valid_path, index=False)
    print(f"Wrote {len(train_df)} rows -> {train_path}")
    print(f"Wrote {len(valid_df)} rows -> {valid_path}")


if __name__ == "__main__":
    main()
