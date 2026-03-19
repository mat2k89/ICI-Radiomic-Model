# Radiomics predictor for hepatocellular carcinoma immunotherapy treatment 
This is a repository with the code and workflow for the machine learning based radiomic predictor for immunotherapy response in patients with hepatocellular carcinoma published as:<br/><br/>
Vithayathil M, Koku D, Campani C, Nault JC, Sutter O, Ganne-Carrié N, Aboagye EO, Sharma R. Machine learning based radiomic models outperform clinical biomarkers in predicting outcomes after immunotherapy for hepatocellular carcinoma. J Hepatol. 2025 Oct;83(4):959-970. [doi: 10.1016/j.jhep.2025.04.017](https://www.journal-of-hepatology.eu/article/S0168-8278(25)00244-2/fulltext)
. Epub 2025 Apr 17. PMID: 40246150.

## Features
<ins>Image preprocessing</ins><br/>
-dicom to nifti conversion<br/>
-voxel resampling<br/><br/>
<ins>Data preprocessing</ins><br/>
-Feature harmonisation<br/>
-Feature engineering (z-normalisationand one-hot encoding)<br/>
-Imputation<br/><br/>
<ins>Model training</ins><br/>
-Feature selection<br/>
-Supervised machine learning prediction<br/>
-Ensemble learning<br/><br/>
<ins>Model evaluation</ins><br/>
-AUC against clinical benchmarks<br/>
-Survival analysis with Kaplan-Meier, Cox Regression and C-index<br/>
-Calibration curve<br/>
-Decision curve analysis<br/>
-Feature importance<br/>

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
### Dicom-to-nifti conversion
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
python dicom_nifti_conversion.py -dicom_directory path -nifti_directory path --compression=True
```
Output<br/>
Directory with subdirectory for each case with nifti files for each dicom series
