# Radiomics predictor for hepatocellular carcinoma immunotherapy treatment 
This is a repository with the code and workflow for the machine learning based radiomic predictor for immunotherapy response in patients with hepatocellular carcinoma published as:<br/><br/>
Vithayathil M, Koku D, Campani C, Nault JC, Sutter O, Ganne-Carrié N, Aboagye EO, Sharma R. Machine learning based radiomic models outperform clinical biomarkers in predicting outcomes after immunotherapy for hepatocellular carcinoma. J Hepatol. 2025 Oct;83(4):959-970. [doi: 10.1016/j.jhep.2025.04.017](https://www.journal-of-hepatology.eu/article/S0168-8278(25)00244-2/fulltext)
. Epub 2025 Apr 17. PMID: 40246150.

## Features<br/>
- <ins>Image preprocessing</ins>
  - dicom to nifti conversion<br/>
  - voxel resampling<br/><br/>
- <ins>Data preprocessing</ins><br/>
  - Feature harmonisation<br/>
  - Feature engineering (z-normalisationand one-hot encoding)<br/>
  - Imputation<br/><br/>
- <ins>Model training</ins><br/>
  - Feature selection<br/>
  - Supervised machine learning prediction<br/>
  - Ensemble learning<br/><br/>
- <ins>Model evaluation</ins><br/>
  - AUC against clinical benchmarks<br/>
  - Survival analysis with Kaplan-Meier, Cox Regression and C-index<br/>
  - Calibration curve<br/>
  - Decision curve analysis<br/>
  - Feature importance<br/>

## Directory structure
```
project_root/
├── src/                # Core implementation
│   ├── images/           # Image preprocessing codes
│   ├── data/         # Data preprocessing codes
│   ├── model_builder/       # Model builder and feature selection codes
│   └── model_evaluation/          # Model evaluation 
├── notebooks/            # Notebooks for radiomic model construction and evaluation
│   └── 2024_Nov_11_ICL_Paris_ICI_ML_OS_JH.ipynb         # Notebook predicting 1-year mortality 
└── requirements.txt
```
## Installation
Uses Python 3.8.8 
```bash
pip install -r requirements.txt
```

## Usage
### <ins>Dicom-to-nifti conversion</ins>
Python file to convert dicom files to nifti files<br/>

Ensure the dicom directory stucture is as follows:
```
dicom directory/
├── 001/                # Scan number
│   ├── xxxxxxxx.dcm       # dicom files for each scan in the directory
│   └── xxxxxxxx.dcm      
│── 002/
│   ├── xxxxxxxx.dcm       
│   └── xxxxxxxx.dcm
│
...
└──  
```
Code
```bash
python src/images/dicom_nifti_conversion.py -dicom_directory_path -nifti_directory_path --compression=True
```
- Required arguments:
  - dicom_dir - path to dicom directory
  - nifti_dir - path to nifti directory

- Optional arguments:
  - compression - compression of nifti file (.nii.gz), default=True

Output<br/>
```
Directory with subdirectory for each case with nifti files for each dicom series
```

### <ins>Voxel Resamping</ins>
Python file to resample nifti images and masks to specified voxel size <br/>

Ensure the nifti directory stucture is as follows:
```
nifti directory/
├── 001/                # Case number
│   ├── 001.nii.gz      # Image file same name as case number directory
│   └── 001_seg.nii.gz  # Segmentation file image name plus <any label> - e.g. "_seg"  
│── 002/
│   ├── 002.nii.gz       
│   └── 002_seg.nii.gz
│
...
└──  
```
Code

```bash
python src/images/image_resample.py -nifti_directory_path -output_directory_path --voxel_size
```
- Required arguments:
  - nifti_dir - path to nifti directory
  - output_dir - path to output directory with resampled images
    
- Optional arguments:
  - voxel_size (-vs) -voxel size for resampled image in format [x,y,z], default=average voxel size
  - image_interpolator (-ip) -image interpolation method, choices=["neighbours","trilinear","cubic"], default="cubic"
  - mask_interpolator (-mp) -mask interpolation method, choices=["neighbours","trilinear","cubic"], default="neighbours"

### <ins>Machine Learning Radiomics Workflow</ins>

Workflow for Radiomics Machine Learning Model shown in notebook
