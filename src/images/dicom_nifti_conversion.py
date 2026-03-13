import numpy as np
import os
import shutil
import argparse

from glob import glob
from tqdm import tqdm

import nibabel as nib
import dicom2nifti

if __name__ == "__main__":

    parser = argparse.ArgumentParser("Dicom to NIfTI conversion")

    parser.add_argument("dicom_dir", help="path to dicom directories")
    parser.add_argument("nifti_dir", help="path to nifti directory")
    parser.add_argument("-c","--compression", default=True, help="compress nifti file")

    args = parser.parse_args()

    root_ct_dir = args.dicom_dir
    nifti_dir = args.nifti_dir

    patient_dir_list = (glob(root_ct_dir+"\*"))

    for p in tqdm(patient_dir_list):
        try:
            id = p.split("\\")[-1]
            print(f"\nProcessing Case number: {id}")
            dicom_dirs=glob(p+"\*")
            for q in dicom_dirs:
                print(f" Converting directory {q}")
                patient_nifti_dir = os.path.join(nifti_dir, id)
                os.makedirs(patient_nifti_dir, exist_ok=True)
                dicom2nifti.convert_directory(q,patient_nifti_dir, compression=args.compression)
            print(f"Case: {id} dicom to nifti conversion complete")
        except:
            print(f"Error processing {p}")


