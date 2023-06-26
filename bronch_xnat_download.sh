#!/bin/bash

cat /home/local/VANDERBILT/litz/data/bronch/bronch_xnat_sessions.txt | xargs -n 1 -I {} \
Xnatdownload -p MCL -d /home/local/VANDERBILT/litz/data/bronch/xnat20221202_bronch \
--sess {} -s all --rs NIFTI,DICOM


cat /home/local/VANDERBILT/litz/github/MASILab/DeepLungScreening/cohorts/xnat/xnat_unmatched_v2.txt | xargs -n 1 -I {} Xnatdownload -p MCL,Atwater,CANARY,HealthMyne,THO1292,Moffitt,Pitt,VLR-VUVA,CTDNA,GGO,MafeCANARY,OptellumAneri,Optellum,TMA34,UW,VLR -d . --rs NIFTI,DICOM --sess {} -s all