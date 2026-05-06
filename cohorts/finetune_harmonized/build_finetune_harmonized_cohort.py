"""
Inventory the harmonized + FOV-extended NIfTI files and emit one CSV per cohort.

Runs on the lab server (paths under /valiant02 are not visible from Mac).

For each cohort folder under SRC_ROOT, walks {cohort}/{cohort}_fov_original_resolution/
and turns each *.nii.gz into a row:
    id           = <filename stripped of harmonized-fov suffix>
    cohort_name  = <cohort folder name>
    fpath        = absolute path to the NIfTI

Outputs:
    finetune_stageA_harmonized_{cohort}.csv  (one per cohort)
    finetune_stageA_harmonized_all.csv       (concatenation)
    finetune_stageA_harmonized_summary.csv   (per-cohort counts, unparseable filenames)
"""

import os, re, glob, sys
import pandas as pd

SRC_ROOT = "/valiant02/masi/lung_data/Biodesix_2026/harmonized_and_fov_extended_cohorts"
DST_DIR  = os.path.dirname(os.path.abspath(__file__))

COHORTS = ["1496", "bronch", "mcl", "nodulevu",
           "reliant1", "reliant2", "veritas", "vlsp"]

# Strip any of these suffixes (in order) from the basename to derive `id`.
# Add new patterns as new naming conventions appear.
SUFFIX_PATTERNS = [
    re.compile(r"_harmonized_fov_extended_orig_res\.nii(?:\.gz)?$"),
    re.compile(r"_harmonized_fov_extended\.nii(?:\.gz)?$"),
    re.compile(r"_harmonized\.nii(?:\.gz)?$"),
    re.compile(r"\.nii(?:\.gz)?$"),  # last-resort fallback
]


def derive_id(fname):
    for pat in SUFFIX_PATTERNS:
        m = pat.search(fname)
        if m:
            return fname[:m.start()], pat.pattern
    return None, None


def inventory_cohort(cohort):
    img_dir = os.path.join(SRC_ROOT, cohort, f"{cohort}_fov_original_resolution")
    if not os.path.isdir(img_dir):
        return None, [], f"missing dir: {img_dir}"
    files = sorted(glob.glob(os.path.join(img_dir, "*.nii.gz")))
    rows, bad = [], []
    pat_counts = {}
    for fp in files:
        fname = os.path.basename(fp)
        sid, pat = derive_id(fname)
        if sid is None:
            bad.append(fname); continue
        pat_counts[pat] = pat_counts.get(pat, 0) + 1
        rows.append({"id": sid, "cohort_name": cohort, "fpath": fp})
    return pd.DataFrame(rows), bad, pat_counts


def main():
    os.makedirs(DST_DIR, exist_ok=True)
    summary_rows = []
    all_dfs = []
    for cohort in COHORTS:
        df, bad, info = inventory_cohort(cohort)
        if df is None:
            print(f"WARN: {cohort}: {info}")
            summary_rows.append({"cohort": cohort, "n_kept": 0, "n_unparseable": 0, "note": info})
            continue
        out_csv = os.path.join(DST_DIR, f"finetune_stageA_harmonized_{cohort}.csv")
        df.to_csv(out_csv, index=False)
        all_dfs.append(df)
        suffix_str = ", ".join(f"{p}:{c}" for p, c in info.items()) if isinstance(info, dict) else ""
        summary_rows.append({
            "cohort": cohort,
            "n_kept": len(df),
            "n_unparseable": len(bad),
            "suffix_pattern_counts": suffix_str,
            "first_unparseable": bad[0] if bad else "",
        })
        print(f"  {cohort:>10}: kept={len(df):>6}  unparseable={len(bad):>3}  -> {out_csv}")
        if bad:
            print(f"             first unparseable: {bad[0]}")

    if all_dfs:
        big = pd.concat(all_dfs, ignore_index=True)
        big_csv = os.path.join(DST_DIR, "finetune_stageA_harmonized_all.csv")
        big.to_csv(big_csv, index=False)
        print(f"  -> all: {len(big):>6}  -> {big_csv}")

    summary = pd.DataFrame(summary_rows)
    summary_csv = os.path.join(DST_DIR, "finetune_stageA_harmonized_summary.csv")
    summary.to_csv(summary_csv, index=False)
    print(f"  -> summary -> {summary_csv}")


if __name__ == "__main__":
    main()
