# Deep Learning Challenge - Alphabet Soup Charity Funding Predictor

## Overview
The nonprofit foundation **Alphabet Soup** seeks a tool to help select applicants for funding with the highest chances of success. This project utilizes **machine learning and neural networks** to develop a **binary classifier** that predicts whether an organization will effectively use the funding provided by Alphabet Soup.

The analysis involves **data preprocessing**, **deep learning model creation**, **model optimization**, and **evaluation** to improve prediction accuracy.

## Dataset
The dataset includes more than **34,000** organizations that received funding. Key metadata columns include:

- **Identification**: `EIN`, `NAME`
- **Application Details**: `APPLICATION_TYPE`, `AFFILIATION`, `CLASSIFICATION`
- **Financial Details**: `INCOME_AMT`, `ASK_AMT`
- **Operational Information**: `ORGANIZATION`, `USE_CASE`, `STATUS`, `SPECIAL_CONSIDERATIONS`
- **Target Variable**: `IS_SUCCESSFUL` (binary classification label)


## Steps & Implementation

### 1. Data Preprocessing
- Loaded the dataset using Pandas.
- Dropped non-essential columns (`EIN` and `NAME`).
- Identified **target variable**: `IS_SUCCESSFUL`.
- Identified **feature variables** and applied transformations:
  - Categorical encoding using `pd.get_dummies()`.
  - Combined rare categories in certain columns.
  - Split data into **training (`X_train`, `y_train`)** and **testing (`X_test`, `y_test`)** sets.
  - Standardized feature variables using `StandardScaler()`.

### 2. Model Creation, Training & Evaluation
- Built a **deep learning model** using TensorFlow/Keras:
  - **Input layer**: Number of neurons equal to input features.
  - **Hidden layers**: Implemented multiple hidden layers with activation functions (`ReLU`).
  - **Output layer**: Used `sigmoid` activation for binary classification.
- Trained the model with:
  - **Loss function**: `binary_crossentropy`
  - **Optimizer**: `Adam`
  - **Epochs**: Initially set to 100
- Created a callback to save model weights every **five epochs**.
- Evaluated the model on test data (**loss and accuracy**).

### 3. Model Optimization
To improve accuracy beyond **75%**, multiple optimizations were attempted:
- Adjusted **input data preprocessing**:
  - Dropped more features with low significance.
  - Created better bins for categorical variables.
- **Tuned model architecture**:
  - Increased/decreased **number of neurons per layer**.
  - Added/removed **hidden layers**.
  - Tested different **activation functions** (`tanh`, `LeakyReLU`).
- **Tuned training process**:
  - Adjusted **learning rate**.
  - Increased/decreased **epochs**.


## Summary & Recommendations
The **deep learning model** demonstrated strong predictive power but can be further enhanced:
- Alternative models such as **Random Forest** or **XGBoost** could be tested for improved explainability and performance.
- Feature engineering improvements could be explored (e.g., deriving new features from existing ones).
- Experimenting with **hyperparameter tuning techniques** (`GridSearchCV`, `RandomSearch`) could yield better configurations.

## Technologies Used
- **Python** (`Pandas`, `NumPy`, `TensorFlow`, `Keras`, `Scikit-learn`, `Matplotlib`, `Seaborn`)
- **Google Colab** (for model training and analysis)
- **Git/GitHub** (for version control and collaboration)
