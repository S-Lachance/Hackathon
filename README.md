This repository contains the code for the **RSN Hackathon Challenge: _Synthetic Data for Accessible Learning in Healthcare: Predicting the Hospital One-year Mortality Risk with Admission Data_**.


## Files
- **`homr.py`** — code used to train the model on the synthetic dataset and optimize its hyperparameters.
- **`rf_homr_best_hps_synthetic.json`** — stores the hyperparameters that produced the best results when training on the synthetic dataset.  
- **`rf_homr_best_hps_original.json`** — stores the hyperparameters that produced the best results when training on the original dataset.

## Pretrained models
Pretrained models can be downloaded from [https://drive.google.com/drive/u/4/folders/13r3tEXg5SfyesCLX0w76K5fJ7MHmbJes](https://drive.google.com/drive/u/4/folders/13r3tEXg5SfyesCLX0w76K5fJ7MHmbJes)
- **`random_forest_synthetic.joblib`** — pretrained model generated using the best hyperparameters when training on the synthetic dataset.
- **`random_forest_original.joblib`** — pretrained model generated using the best hyperparameters when training on the original dataset.

## Dataset
The synthetic dataset can be:
- Automatically installed from the corresponding section in `homr.py`, **or**
- Manually downloaded from **[zenodo.org/records/12954673](https://zenodo.org/records/12954673)**.


## Usage
1. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   
2. **Load the pretrained model for post-analyses**
   ```python
   import joblib
   rf_synthetic = joblib.load("random_forest_synthetic.joblib")
   rf_original = joblib.load("random_forest_original.joblib")
