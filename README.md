# Fraud Detection Project

## Project Overview

This project focuses on detecting fraudulent transactions using machine learning algorithms. The dataset consists of raw transaction data that is preprocessed before training different models. The performance of the models is evaluated using confusion matrices and various metrics.

## Repository Structure

```
├── data
│   ├── processed
│   │   ├── processed_data.csv    
│   ├── raw
│   │   ├── Fraud.csv          
│
├── notebooks
│   ├── 01_EDA.ipynb         
│   ├── 02_Model.ipynb      
│
├── results
│   ├── confusion_matrix      
│   │   ├── decision_tree.png
│   │   ├── knn.png
│   │   ├── logistic_regression.png
│   │   ├── random_forest.png
│   │   ├── svm.png
│   ├── metrics               
│   │   ├── decision_tree.txt
│   │   ├── knn.txt
│   │   ├── logistic_regression.txt
│   │   ├── random_forest.txt
│   │   ├── svm.txt
│   ├── roc_curve              
│   │   ├── decision_tree.png
│   │   ├── knn.png
│   │   ├── logistic_regression.png
│   │   ├── random_forest.png
│   │   ├── svm.png
|   ├── best_model_predictions.csv   
|   ├── feature_importance.png
|   ├── models_performance.json # The different results of models (with random hyperparameters, with grid search, and with feature selection)
├── scr
|   ├── evaluation.py # Functions for model evaluation
|   ├── visualization.py # Functions for confusion matrix and ROC curve visualization
├── requirements.txt
```

## Dataset

- **Raw Data:** Located in `data/raw/Fraud.csv`
- **Processed Data:** Located in `data/processed/processed_data.csv`
- **Source:** The dataset was downloaded from Kaggle (https://www.kaggle.com/datasets/vardhansiramdasu/fraudulent-transactions-prediction).

## Preprocessing Steps

1. Handling missing values and inconsistencies.
2. Handling unbalanced data.
3. Encoding categorical variables.
4. Feature scaling and normalization.
## Notebooks

- `01_EDA.ipynb`: Performs Exploratory Data Analysis (EDA) on the dataset, identifying trends and patterns in fraudulent transactions.
- `02_Model.ipynb`: Trains different machine learning models and evaluates their performance.

## Models Implemented

- Decision Tree
- k-Nearest Neighbors (KNN)
- Logistic Regression
- Random Forest
- Support Vector Machine (SVM)

## Model Training Approaches

For each model, three training approaches were used:

1. Training with random hyperparameters.
2. Training with hyperparameter tuning using GridSearch.
3. Training with feature selection (except for Decision Tree and Random Forest, which perform feature selection implicitly).

The best performance was achieved using the **Random Forest model with GridSearch**, which resulted in a **precision of 99%**.

## Results

- **Confusion Matrices:** Located in `results/confusion_matrix/`
- **Performance Metrics:** Located in `results/metrics/`
- **ROC Curves:** Located in `results/roc_curve/`
- **Feature Importance Analysis:** The `feature_importance.png` file visualizes the most important features contributing to fraud detection.
- **Best Model Predictions:** The `best_model_predictions.csv` file contains the predicted results from the top-performing model.

## How to Use

1. Clone the repository.
2. Open `notebooks/01_EDA.ipynb` and run it to understand the dataset.
3. Run `notebooks/02_Model.ipynb` to train and evaluate models.

## Requirements

Install the necessary dependencies using:

```bash
pip install -r requirements.txt
```

## Contact

For any inquiries, feel free to reach out.

