### Dataset Overview:
- **Article:** "Electrocardiogram as a single source for early diagnosis of multi-label cardiac hypertrophy and dilation: a reduced-channel deep-learning approach for portable devices"
- **Training Samples:** Approximately 10,000 samples.
  - **Distribution:**
    - **LAE:** 47.80%
    - **LVH/LVD:** 34.24%
    - **RAE:** 20.40%
    - **RVH/RVD:** 12.28%
    - **Normal:** 29.33%
- **Test Set:** 473 samples.

### Released Data: 
Please download our public data in the 'Releases' file.
- **Training Samples:** 
  - The 10,000 training samples are divided into two ZIP files:
    - `TrainECG_first_5000.zip` contains the first 5,000 training samples.
    - `TrainECG_second_5000.zip` contains the remaining 5,000 training samples.
- **Labels:**
  - All labels for the training samples are stored in the `label.xlsx` file. Each row in the file corresponds to a label for a specific training sample.
- **Test Samples:** 
  - The 473 test samples are compressed into `Test_ECG.zip`.
  - Labels for the test samples are stored in `Test_label.xlsx`.

### Label Information:
- **Label Format:** The label files (`label.xlsx and Test_label.xlsx`) contain binary labels where '1' indicates the presence of a specific condition in the corresponding ECG data.

### Model Information:
- **Model Training and Testing:** The code for model training and testing is provided in `training.py`. The parameters are based on the complete training set. For the publicly available 10,000 training samples, it's recommended to reduce the number of model layers to prevent overfitting.
