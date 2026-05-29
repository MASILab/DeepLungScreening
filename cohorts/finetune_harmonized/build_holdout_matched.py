"""
Match held-out test subjects against existing Stage A feature outputs.

The held-out spreadsheet (selected_subjects_exclude_*.xlsx) lists subjects by
their UNHARMONIZED fpath. Stage A features were computed on (possibly
harmonized) images, and the resulting feat128 filename varies by cohort:
some cohorts keep the bare unharmonized stem (`<basename>.npy`), NLST appends
`_resampled[_masked]_fov_extended_orig_res`, others append
`_harmonized_fov_extended_orig_res`. Rather than guess the per-cohort suffix
(which silently misses files when the convention differs), we GLOB the cohort's
feat128/ directory for `<basename>*.npy` and take the actual file on disk.

Same DLS-ready output schema as build_finetune_matched.py — Stage B's
step4_predict.py reads either CSV interchangeably.

Run on the lab server:
    python3 cohorts/finetune_harmonized/build_holdout_matched.py
"""

import os, sys, re, glob
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


def to_folder(cohort_name):
    if pd.isna(cohort_name):
        return None
    s = str(cohort_name)
    return COHORT_FOLDER_OVERRIDES.get(s, s.lower())


def _pick_feat(matches):
    """Given candidate feat128 .npy paths for one subject, pick one.

    Prefer the non-masked variant when both masked and non-masked exist
    (NLST produces `_resampled_fov_extended_orig_res` AND
    `_resampled_masked_fov_extended_orig_res`; the non-masked one matches
    what Stage B was trained on). Otherwise take the shortest name (closest
    to the bare stem), deterministically.
    """
    if not matches:
        return None
    non_masked = [m for m in matches if "_masked" not in os.path.basename(m)]
    pool = non_masked if non_masked else matches
    # Shortest basename first, then lexicographic — deterministic.
    pool = sorted(pool, key=lambda m: (len(os.path.basename(m)), m))
    return pool[0]


def resolve_feat(row):
    """unharmonized fpath -> (id, feat128_path, feat_ready) by globbing disk.

    Looks in <FEAT_ROOT>/<cohort_folder>/feat128/ for the subject's stem.
    Tries an exact `<stem>.npy` first, then `<stem>*.npy` (which catches the
    various harmonization/resample suffixes), then `<stem>_*.npy`.
    """
    if pd.isna(row["fpath"]) or pd.isna(row["cohort_name"]):
        return pd.Series({"id": None, "feat128_path": None, "feat_ready": False})
    folder = to_folder(row["cohort_name"])
    stem = strip_nii(os.path.basename(str(row["fpath"])))
    feat_dir = os.path.join(FEAT_ROOT, folder, "feat128")

    exact = os.path.join(feat_dir, f"{stem}.npy")
    if os.path.isfile(exact):
        chosen = exact
    else:
        matches = glob.glob(os.path.join(feat_dir, f"{stem}*.npy"))
        chosen = _pick_feat(matches)

    if chosen is None:
        # Report the bare stem so unmatched diagnostics are readable.
        return pd.Series({"id": stem, "feat128_path": exact, "feat_ready": False})
    fid = os.path.splitext(os.path.basename(chosen))[0]
    return pd.Series({"id": fid, "feat128_path": chosen, "feat_ready": True})


def main():
    if not os.path.isfile(SPREADSHEET):
        sys.exit(f"ERROR: spreadsheet not found: {SPREADSHEET}\n"
                 f"       set HOLDOUT_XLSX=/path/to/file.xlsx if it lives elsewhere.")
    if not os.path.isdir(FEAT_ROOT):
        sys.exit(f"ERROR: feat root not found: {FEAT_ROOT}")

    print(f"Reading {SPREADSHEET}")
    df = pd.read_excel(SPREADSHEET)
    print(f"  rows: {len(df)}")

    df["cohort_folder"] = df["cohort_name"].apply(to_folder)
    resolved = df.apply(resolve_feat, axis=1)
    df[["id", "feat128_path", "feat_ready"]] = resolved[["id", "feat128_path", "feat_ready"]]

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
