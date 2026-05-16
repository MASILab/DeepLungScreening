"""
Match held-out test subjects against existing Stage A feature outputs.

The held-out spreadsheet (selected_subjects_exclude_*.xlsx) lists subjects by
their UNHARMONIZED fpath. Stage A features were computed on the HARMONIZED
images, so we derive each row's expected Stage A id from cohort-specific
filename conventions and check whether feat128/{id}.npy exists on disk.

Same DLS-ready output schema as build_finetune_matched.py — Stage B's
step4_predict.py reads either CSV interchangeably.

Run on the lab server:
    python3 cohorts/finetune_harmonized/build_holdout_matched.py
"""

import os, sys, re
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SPREADSHEET = os.environ.get(
    "HOLDOUT_XLSX",
    os.path.join(SCRIPT_DIR, "selected_subjects_exclude_20260515.xlsx"),
)
FEAT_ROOT = os.environ.get(
    "BIODESIX_FEAT_ROOT",
    "/valiant02/masi/zuol1/projects/biodesix/DeepLungScreen/data/finetune_harmonized",
)
OUT_CSV = os.path.join(SCRIPT_DIR, "biodesix_holdout_matched.csv")

# Spreadsheet cohort name -> Stage A output folder.
COHORT_FOLDER_OVERRIDES = {"tho1496": "1496"}

# Harmonized id = unharmonized_basename + this suffix (per cohort).
# An empty string means "no extra suffix; id == unharmonized stem".
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
    if fname.endswith(".nii.gz"):
        return fname[:-7]
    if fname.endswith(".nii"):
        return fname[:-4]
    return fname


def derive_id(row):
    """unharmonized fpath -> expected Stage A id."""
    if pd.isna(row["fpath"]):
        return None
    base = strip_nii(os.path.basename(str(row["fpath"])))
    suffix = HARMONIZED_SUFFIX.get(row["cohort_name"], "_harmonized_fov_extended_orig_res")
    return base + suffix


def to_folder(cohort_name):
    if pd.isna(cohort_name):
        return None
    s = str(cohort_name)
    return COHORT_FOLDER_OVERRIDES.get(s, s.lower())


def main():
    if not os.path.isfile(SPREADSHEET):
        sys.exit(f"ERROR: spreadsheet not found: {SPREADSHEET}\n"
                 f"       set HOLDOUT_XLSX=/path/to/file.xlsx if it lives elsewhere.")
    if not os.path.isdir(FEAT_ROOT):
        sys.exit(f"ERROR: feat root not found: {FEAT_ROOT}")

    print(f"Reading {SPREADSHEET}")
    df = pd.read_excel(SPREADSHEET)
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
    rep = df.groupby("cohort_name").agg(
        rows=("id", "count"),
        feat_ready=("feat_ready", "sum"),
    )
    rep["pct"] = (rep["feat_ready"] / rep["rows"] * 100).round(1)
    print(rep.to_string())

    print()
    print("Label distribution in matched-and-ready subset:")
    ready = df[df["feat_ready"]]
    if len(ready):
        print(ready["lung_cancer"].value_counts().to_string())

    print()
    print("Examples of UNMATCHED rows (first 5) — check the derived id against disk:")
    unmatched = df[~df["feat_ready"]]
    for _, r in unmatched.head(5).iterrows():
        print(f"  cohort={r['cohort_name']:<10}  id={r['id']}")
        print(f"     expected: {r['feat128_path']}")

    df_renamed.to_csv(OUT_CSV, index=False)
    print()
    print(f"Wrote {len(df_renamed)} rows -> {OUT_CSV}")


if __name__ == "__main__":
    main()
