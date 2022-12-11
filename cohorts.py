"""
Pipeline to scope/define cohorts and preprocess the clinical data from data dictionaries
"""

import os
import numpy as np
import pandas as pd
import glob


def nlst(meta):
    root_dir = meta["root_dir"]
    raw_cli = os.path.join(root_dir, "participant.data.d100517.csv")
    raw_df = pd.read_csv(raw_cli, dtype={"pid": str})
    
    scan_dir = meta["scan_dir"]
    rows = []
    for scan in glob.glob(os.path.join(scan_dir, "*_clean.nii.gz")):
        scanid = os.path.basename(scan).split("_clean.nii.gz")[0]
        pid, year = scanid.split("time")
        rows.append({"pid": pid, "year": year, "id": scanid})

    # extract desired features - matching 
    demo_ft = ['pid', 'age', 'educat', 'race', 'ethnic', 'height', 'weight',]
    copd_ft = ['diagcopd']
    pmh_ft = ['cancblad', 'cancbrea', 'canccerv', 'canccolo', 'cancesop', 'canckidn', 'canclary',
        'cancnasa', 'cancoral', 'cancpanc', 'cancphar', 'cancstom', 'cancthyr', 'canctran'] # past medical history features
    plc_ft = ['canclung']
    fmh_ft = ['fambrother', 'famchild', 'famfather', 'fammother', 'famsister'] # family history of lung cancer
    smoking_ft = ['cigsmok', 'age_quit', 'pkyr']
    lc_ft = ['conflc', 'cancyr', 'candx_days'] # biopsy-confirmed lung cancer
    sc_ft = demo_ft + copd_ft+pmh_ft + plc_ft+fmh_ft+smoking_ft+lc_ft
    sc_df = raw_df[sc_ft]

    # apply inclusion/exclusion criteria
    sc_df = sc_df[(sc_df['age']>=55) & (sc_df['age'] <= 74)] # age
    sc_df = sc_df[sc_df['pkyr'] >=30] # pkr
    sc_df['canclung'] = sc_df['canclung'].fillna(0) # prior lung cancer
    sc_df = sc_df[sc_df['canclung']==0]
    # assume NLST excluded subjects w/ chest ct within 18 mo, hemoptysis, weight loss

    # demo
    sc_df['age'] = sc_df['age']
    def parse_edu(x):
        mapper = {
            1:1, # less than high school
            2:1,
            3:2, # high school
            4:3, # post high school training
            5:4, # some college
            6:5, # college degree
            7:6, # graduate degree
            8:0, # other, missing, decline to answer
            95:0,
            99:0,
            98:0,
        }
        return mapper[x]    
    sc_df['educat'] = sc_df['educat'].apply(lambda x: parse_edu(x))
    def parse_race(x):
        mapper= {
            1:1, #White
            2:2, #Black
            3:4, #Asian
            4:5, #American Indian or Alaskan Native
            5:6, #Native Hawaiian or Other Pacific Islander
            6:0, #Mixed, other, missing, unkown, decline to answer
            7:0,
            95:0,
            96:0,
            98:0,
            99:0
        }
        return mapper[x]
    sc_df['race'] = sc_df['race'].apply(lambda x: parse_race(x))

    # set race to 3 if ethnicity is hispanic
    def parse_ethnic(x):
        mapper = {
            1:1, # hispanic
            2:0, # neither hispanic nor latino, missing, decline to answer
            7:0,
            95:0,
            98:0,
            99:0
        }
        return mapper[x]
    sc_df['ethnic'] = sc_df['ethnic'].apply(lambda x: parse_ethnic(x))
    sc_df.loc[sc_df['ethnic']==1, 'race'] = 3

    # calculate bmi = kg/m^2
    sc_df['weight'] = 0.45359237*sc_df['weight'] # lb to kg
    sc_df['height'] = 0.0254*sc_df['height'] # in to m
    sc_df['bmi'] = sc_df['weight'].div(np.power(sc_df['height'], 2))
    sc_df['race'].value_counts(dropna=False)



    return cohort_df

if __name__ == "__main__":
    
    meta = {
        "root_dir": "/nfs/masi/NLST/package-nlst-7-2018.09.24",
        "scan_dir": "/home/local/VANDERBILT/litz/data/nlst/DeepLungScreening/prep",
    }
