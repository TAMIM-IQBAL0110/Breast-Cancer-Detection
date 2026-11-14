# Breast Cancer Detection Using PyTorch

## 1. Abstract
This project implements a binary classification model for breast cancer detection using PyTorch. A manually constructed neural network is trained to distinguish between benign and malignant breast tumors using diagnostic features. The implementation covers the complete training pipeline, including data preprocessing, forward propagation, loss computation, backpropagation, parameter optimization, and model evaluation.

## 2. Dataset
The project uses the Breast Cancer Wisconsin Diagnostic Dataset.

- **Total samples:** 569
- **Input features:** 30 numerical features
- **Target variable:** Diagnosis
- **Benign (B):** 357 samples
- **Malignant (M):** 212 samples

The dataset is loaded from the following source:

[Breast Cancer Detection Dataset](https://github.com/gscdit/Breast-Cancer-Detection)

## 3. Methodology

The implementation follows these steps:

1. **Data preprocessing**
   - Remove unnecessary columns (`id` and `Unnamed: 32`).
   - Separate input features and target labels.
   - Encode categorical diagnosis labels.
   - Split the dataset into training (80%) and testing (20%) sets using stratified sampling.
   - Apply StandardScaler to normalize input features.

2. **Model development**
   - Initialize weights and bias.
   - Implement a forward pass using matrix multiplication.
   - Apply the sigmoid activation function.
   - Calculate binary cross-entropy loss.

3. **Model training**
   - Perform backpropagation using PyTorch autograd.
   - Update model parameters using gradient descent.
   - Train for 25 epochs with a learning rate of 0.01.

4. **Model evaluation**
   - Generate predictions on the test dataset.
   - Apply a classification threshold of 0.65.
   - Evaluate performance using accuracy, precision, recall, and F1-score.

## 4. Model Architecture

The model is a single-layer logistic regression classifier implemented manually using PyTorch tensors and automatic differentiation.

| Component | Configuration |
|---|---|
| Input layer | 30 features |
| Trainable weights | 30 × 1 |
| Bias | 1 |
| Activation | Sigmoid |
| Loss function | Binary Cross-Entropy |
| Optimization | Manual gradient descent |

## 5. Experimental Configuration

| Parameter | Value |
|---|---|
| Training samples | 455 |
| Testing samples | 114 |
| Train-test ratio | 80:20 |
| Random seed | 42 |
| Epochs | 25 |
| Learning rate | 0.01 |
| Classification threshold | 0.65 |
| Computing device | Apple MPS |

## 6. Results

The model achieved the following performance on the held-out test set:

| Metric | Score |
|---|---:|
| Accuracy | 91.23% |
| Precision (Benign) | 0.93 |
| Recall (Benign) | 0.93 |
| F1-score (Benign) | 0.93 |
| Precision (Malignant) | 0.88 |
| Recall (Malignant) | 0.88 |
| F1-score (Malignant) | 0.88 |

## 7. Technologies Used

- Python
- PyTorch
- NumPy
- Pandas
- Scikit-learn
- Jupyter Notebook

## 8. Reproducibility

A random seed of 42 is used to support reproducibility. The implementation automatically selects an available computing device, including CUDA, Apple MPS, or CPU.

## 9. Project Structure

```text
Pytorch/
│
├── pytorch-Training-Pipeline.ipynb
├── README.md
└── .gitignore
```

## 10. Limitations

- The model uses a single linear layer and does not capture nonlinear feature interactions.
- Evaluation is based on a single stratified train-test split.
- The classification threshold is fixed at 0.65.
- Further validation and threshold analysis are required to assess generalization.

## 11. Disclaimer

This project is developed for educational and research purposes. It is not intended for clinical diagnosis or medical decision-making.
