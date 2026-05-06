#!/bin/bash
# Quick status snapshot for the distributed Stage A run.
# Shows per-cohort progress: total chunks vs claimed vs done, plus output file counts.

set -euo pipefail

ROOT=/valiant02/masi/zuol1/projects/biodesix/DeepLungScreen/data/finetune_harmonized
CHUNK_ROOT=${ROOT}/chunks

if [ ! -d "${CHUNK_ROOT}" ]; then
    echo "No chunk dir at ${CHUNK_ROOT} (nothing started yet)."
    exit 0
fi

echo "Cohort        chunks  claimed  done  pct"
echo "------------  ------  -------  ----  ----"
for d in "${CHUNK_ROOT}"/*/; do
    cohort=$(basename "${d}")
    total=$(ls "${d}"chunk_*.csv 2>/dev/null | wc -l)
    [ "${total}" -eq 0 ] && continue
    claimed=$(ls -d "${d}"chunk_*.lock 2>/dev/null | wc -l)
    done=$(ls "${d}"chunk_*.lock/done 2>/dev/null | wc -l)
    pct=$(awk "BEGIN { if (${total}==0) print 0; else printf \"%.0f\", 100*${done}/${total} }")
    printf "%-12s  %6d  %7d  %4d  %3s%%\n" "${cohort}" "${total}" "${claimed}" "${done}" "${pct}"
done

echo ""
echo "Output file counts:"
for sub in nifti prep bbox feat64 feat128; do
    if [ -d "${ROOT}/${sub}" ]; then
        n=$(ls "${ROOT}/${sub}" 2>/dev/null | wc -l)
        printf "  %-8s %d\n" "${sub}" "${n}"
    fi
done

# Stale locks (claimed but not done, older than 6 hours)
stale=$(find "${CHUNK_ROOT}" -maxdepth 3 -name '*.lock' -type d -mmin +360 2>/dev/null | while read l; do
    [ -f "${l}/done" ] || echo "${l}"
done | wc -l)
if [ "${stale}" -gt 0 ]; then
    echo ""
    echo "WARN: ${stale} lock(s) older than 6h without done — likely stalled workers."
    echo "      They will be auto-reclaimed by the next worker that polls for chunks."
fi
