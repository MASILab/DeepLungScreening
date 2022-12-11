import os
import nibabel as nib
import sys
import glob
sys.path.append("/home/local/VANDERBILT/litz/github/MASILab/thoraxtools/func")
import vis.paral_clip_overlay_mask as overlay

def main(data_dir, clip_dir):
    """clip nii for visualization"""
    images = glob.glob(os.path.join(data_dir, "*.nii.gz"))[:100]
    for img in images:
        out_png = os.path.join(clip_dir, f"{os.path.basename(img).split('_clean.nii.gz')[0]}_axial.png")
        overlay.multiple_clip_overlay(img, out_png, img_vrange=(0,255))

if __name__ == "__main__":
    args = sys.argv[1:]
    main(*args)