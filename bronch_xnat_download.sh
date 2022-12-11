#!/bin/bash

cat /home/local/VANDERBILT/litz/data/bronch/bronch_xnat_sessions.txt | xargs -n 1 -I {} \
Xnatdownload -p MCL -d /home/local/VANDERBILT/litz/data/bronch/xnat20221202_bronch \
--sess {} -s all --rs NIFTI,DICOM
