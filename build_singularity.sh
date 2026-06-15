#!/usr/bin/env bash
# Build the DeepLungScreening Singularity/Apptainer image from a CLEAN copy.
#
# Why a staging dir: deeplung.def uses `%files . /app`, which copies the
# build context wholesale. Building straight from the working tree would also
# pull in gitignored fine-tuning data/outputs (potentially many GB of NFS
# volumes) into the .sif. This script stages only git-tracked files first
# (which DOES include the committed *.ckpt / *.pth model weights).
#
# Usage:
#   ./build_singularity.sh                # -> deeplung.sif
#   ./build_singularity.sh my_image.sif   # custom output name
#
# Requires apptainer (or singularity) and either root or --fakeroot support.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT="${1:-${REPO_ROOT}/deeplung.sif}"

# Pick whichever container runtime is available.
if command -v apptainer >/dev/null 2>&1; then
    RUNTIME=apptainer
elif command -v singularity >/dev/null 2>&1; then
    RUNTIME=singularity
else
    echo "ERROR: neither apptainer nor singularity found on PATH." >&2
    exit 1
fi

# Resolve the absolute path. sudo's secure_path often differs from the user's
# PATH, so `sudo singularity` can fail with "command not found" even though the
# binary is on the user's PATH. Always invoke it by full path under sudo.
RUNTIME_BIN="$(command -v "$RUNTIME")"

# Decide how to get root for the build. Order of preference:
#   1. already root          -> build directly
#   2. sudo available        -> sudo build, then chown the .sif back to us
#                               (most reliable on singularity 3.8.6, where
#                               rootless --fakeroot often isn't configured)
#   3. fall back to --fakeroot
# Override with BUILD_MODE=fakeroot|sudo|root if you want to force one.
BUILD_MODE="${BUILD_MODE:-}"
if [ -z "$BUILD_MODE" ]; then
    if [ "$(id -u)" -eq 0 ]; then
        BUILD_MODE=root
    elif command -v sudo >/dev/null 2>&1; then
        BUILD_MODE=sudo
    else
        BUILD_MODE=fakeroot
    fi
fi

STAGE="$(mktemp -d)"
trap 'rm -rf "$STAGE"' EXIT

# Stage as the invoking user (NOT under sudo) so git doesn't trip over
# "dubious ownership" on a repo owned by you.
echo ">> Staging tracked files into $STAGE"
git -C "$REPO_ROOT" archive --format=tar HEAD | tar -x -C "$STAGE"

# git archive omits the definition file if it isn't committed yet — copy it in.
cp "$REPO_ROOT/deeplung.def" "$STAGE/deeplung.def"

# Build to a LOCAL temp path first, then move into place.
# Final destinations are often on NFS, where root_squash maps the (sudo) root
# build process to 'nobody' and the SIF write is denied. $STAGE lives under
# $TMPDIR (default /tmp, local), so root can write there; the move into $OUT is
# done as the invoking user, who DOES own the NFS dir.
# If /tmp is too small for the ~6-8 GB image, run with TMPDIR pointed at a local
# scratch dir, e.g.  TMPDIR=/local/scratch ./build_singularity.sh
BUILD_SIF="$STAGE/image.sif"

echo ">> Building (mode: $BUILD_MODE) -> $BUILD_SIF"
case "$BUILD_MODE" in
    root)
        ( cd "$STAGE" && "$RUNTIME_BIN" build "$BUILD_SIF" deeplung.def )
        ;;
    sudo)
        # Full path + preserve PATH so singularity can find its own helpers.
        ( cd "$STAGE" && sudo env "PATH=$PATH" "$RUNTIME_BIN" build "$BUILD_SIF" deeplung.def )
        # SIF is root-owned; hand it back before the (non-root) move.
        sudo chown "$(id -u):$(id -g)" "$BUILD_SIF"
        ;;
    fakeroot)
        ( cd "$STAGE" && "$RUNTIME_BIN" build --fakeroot "$BUILD_SIF" deeplung.def )
        ;;
    *)
        echo "ERROR: unknown BUILD_MODE '$BUILD_MODE'" >&2
        exit 1
        ;;
esac

echo ">> Moving image into place: $OUT"
mv -f "$BUILD_SIF" "$OUT"

echo ">> Done: $OUT"
