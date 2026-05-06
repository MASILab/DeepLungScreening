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

count_glob () {
    local n=0
    for f in $1; do
        [ -e "$f" ] && n=$((n + 1))
    done
    echo "$n"
}

echo "Cohort        chunks  claimed  done  pct"
echo "------------  ------  -------  ----  ----"
for d in "${CHUNK_ROOT}"/*/; do
    [ -d "${d}" ] || continue
    cohort=$(basename "${d}")
    total=$(count_glob "${d}chunk_*.csv")
    [ "${total}" -eq 0 ] && continue
    claimed=$(count_glob "${d}chunk_*.lock")
    done=$(count_glob "${d}chunk_*.lock/done")
    pct=$(awk "BEGIN { if (${total}==0) print 0; else printf \"%.0f\", 100*${done}/${total} }")
    printf "%-12s  %6d  %7d  %4d  %3s%%\n" "${cohort}" "${total}" "${claimed}" "${done}" "${pct}"
done

echo ""
echo "Per-cohort output file counts:"
printf "  %-12s %8s %8s %8s %8s %8s\n" "cohort" "nifti" "prep" "bbox" "feat64" "feat128"
for d in "${CHUNK_ROOT}"/*/; do
    [ -d "${d}" ] || continue
    cohort=$(basename "${d}")
    counts=()
    for sub in nifti prep bbox feat64 feat128; do
        if [ -d "${ROOT}/${cohort}/${sub}" ]; then
            counts+=( "$(count_glob "${ROOT}/${cohort}/${sub}/*")" )
        else
            counts+=( "-" )
        fi
    done
    printf "  %-12s %8s %8s %8s %8s %8s\n" "${cohort}" "${counts[@]}"
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
