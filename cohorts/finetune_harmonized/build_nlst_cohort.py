"""
Convert the NLST master metadata CSV into the same schema other harmonized
cohorts already use:  id, cohort_name, fpath.

Why a separate script: NLST images aren't laid out in
{cohort}/{cohort}_fov_original_resolution/ like the other cohorts; their
harmonized paths are listed inside the master spreadsheet directly.

Usage (on the lab server):
    python3 cohorts/finetune_harmonized/build_nlst_cohort.py
    # or with custom source CSV:
    python3 cohorts/finetune_harmonized/build_nlst_cohort.py /path/to/source.csv

Output: cohorts/finetune_harmonized/finetune_stageA_harmonized_nlst.csv
"""

import os, sys
import pandas as pd

DEFAULT_SRC = (
    "/valiant02/masi/zuol1/projects/biodesix/DeepLungScreen/"
    "cohorts/finetune_harmonized/"
    "nlst_cohort_with_all_metadata_and_harmonized_file_paths_20260506_v2.csv"
)
DST_DIR = os.path.dirname(os.path.abspath(__file__))
DST = os.path.join(DST_DIR, "finetune_stageA_harmonized_nlst.csv")

FPATH_COL = "harmonized_file_path"


def derive_id(path):
    """The harmonized basename (minus .nii.gz) is unique per row and safe
    as a filename — keep it verbatim as the id."""
    fname = os.path.basename(str(path))
    if fname.endswith(".nii.gz"):
        return fname[:-7]
    if fname.endswith(".nii"):
        return fname[:-4]
    return fname


def main():
    src = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_SRC
    if not os.path.isfile(src):
        sys.exit(f"ERROR: source CSV not found: {src}")

    print(f"Reading {src}")
    df = pd.read_csv(src, low_memory=False)
    n_total = len(df)
    print(f"  total rows:                    {n_total}")

    if FPATH_COL not in df.columns:
        sys.exit(f"ERROR: expected column '{FPATH_COL}' not found.  Got: {list(df.columns)[:5]}...")

    # Drop rows where harmonization didn't produce a path.
    mask = df[FPATH_COL].notnull() & (df[FPATH_COL].astype(str).str.strip() != "")
    df = df[mask].copy()
    print(f"  after dropping empty fpath:    {len(df)}")

    # Build id from filename basename.
    df["id"] = df[FPATH_COL].apply(derive_id)

    # Sanity-check: ids should be unique.  If not, we'd have collisions on
    # symlinks / outputs.
    dup = df["id"].duplicated()
    if dup.any():
        n_dup = int(dup.sum())
        print(f"  WARN: {n_dup} duplicate ids; keeping first occurrence each")
        df = df[~dup].reset_index(drop=True)

    # Standard cohort-CSV schema: id, cohort_name, fpath.
    out = pd.DataFrame({
        "id":          df["id"].values,
        "cohort_name": "nlst",
        "fpath":       df[FPATH_COL].values,
    })

    out.to_csv(DST, index=False)
    print(f"  wrote {len(out)} rows -> {DST}")
    print()
    print("Sample:")
    print(out.head(3).to_string(index=False))


if __name__ == "__main__":
    main()
