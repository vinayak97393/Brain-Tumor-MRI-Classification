# Brain Tumor MRI Image Classification

An educational deep-learning project that classifies brain MRI images into four dataset categories using a Custom Convolutional Neural Network (CNN) and MobileNetV2 transfer learning.

## Project Overview

This project explores image classification with TensorFlow and Keras. It includes two trained models, model evaluation, a Streamlit image-classification interface, and a performance dashboard.

**Important:** This project is for educational and research purposes only. It is not a medical diagnostic tool.

## Dataset

The dataset contains MRI images organized into four classes:

- Glioma
- Meningioma
- No Tumor
- Pituitary

| Dataset Split | Number of Images |
|---|---:|
| Training | 1,695 |
| Validation | 502 |
| Testing | 246 |
| **Total** | **2,443** |

The supplied dataset README identifies the dataset licence as **CC BY 4.0**. Review and follow its attribution requirements when redistributing the dataset or its contents.

## Models

### 1. Custom CNN

A convolutional neural network built using TensorFlow and Keras, with convolutional layers, max pooling, dropout, and dense classification layers.

### 2. MobileNetV2

A transfer-learning model using an ImageNet-pretrained MobileNetV2 base with a custom classification head.

## Model Evaluation

| Metric | Custom CNN | MobileNetV2 |
|---|---:|---:|
| Test Accuracy | 89.43% | 86.18% |
| Macro F1-Score | 0.89 | 0.86 |
| Total Parameters | 11,169,476 | 2,263,108 |
| Correct Test Predictions | 220 / 246 | 212 / 246 |

On this test set, the Custom CNN achieved higher accuracy, while MobileNetV2 used substantially fewer parameters.

These results describe performance on this particular dataset and do not establish clinical reliability.

## Features

- Upload MRI images in JPG, JPEG, or PNG format.
- Generate predictions across four dataset classes.
- Display model output scores for each class.
- Compare model accuracy and parameter counts.
- View the Custom CNN confusion matrix.
- Explore an interactive Streamlit interface and performance dashboard.

## Technologies Used

- Python
- TensorFlow and Keras
- NumPy
- Pandas
- Matplotlib
- Scikit-learn
- Pillow
- Streamlit

## Project Structure

```text
Brain_Tumor_MRI_Classification/
├── app.py
├── dashboard.py
├── check_environment.py
├── dataset_summary.py
├── view_mri_images.py
├── build_cnn.py
├── train_cnn.py
├── evaluate_cnn.py
├── train_mobilenet.py
├── evaluate_mobilenet.py
├── model_comparison.py
├── datasets/
├── models/
│   ├── brain_tumor_cnn.keras
│   └── mobilenetv2_brain_tumor.keras
└── outputs/
    ├── confusion_matrix.png
    └── model_comparison.csv
```

## Installation and Usage

### 1. Clone the repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
cd Brain_Tumor_MRI_Classification
```

### 2. Create and activate a virtual environment

```bash
python -m venv .venv
```

On Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

Install the packages used by the project:

```bash
python -m pip install tensorflow streamlit numpy pandas matplotlib scikit-learn pillow
```

### 4. Run the MRI classifier

```bash
python -m streamlit run app.py
```

### 5. Run the performance dashboard

Open another terminal in the project directory and run:

```bash
python -m streamlit run dashboard.py
```

Ensure the trained model files and evaluation outputs are available at the expected paths before launching the applications.

## Limitations

- Performance depends on the dataset and image characteristics.
- High model confidence does not guarantee a correct prediction.
- Images from different sources may produce unreliable results.
- The models have not been validated for clinical use.

**Never use this project to diagnose a person, rule out a tumor, or make medical decisions.**

## Future Improvements

- Add automated unit tests and input validation.
- Improve performance using validation-based experiments.
- Add reproducible training and evaluation instructions.
- Deploy a demo for portfolio presentation.
- Improve the interface and document model limitations.

## Author

Developed as a machine-learning portfolio project focused on medical-image classification, transfer learning, model evaluation, and Streamlit application development.