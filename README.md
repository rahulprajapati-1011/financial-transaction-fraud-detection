# Fraud Detection System Using SMOTE + XGBoost

A structured machine learning project for detecting fraudulent financial transactions in a highly imbalanced dataset.  
This project focuses on **feature engineering, imbalance handling, model comparison, and final fraud classification using XGBoost**.  
Deployment is intentionally excluded for now and will be added in a later phase.

---

## 1. Project Overview

Fraud detection is a classic **imbalanced classification** problem where fraudulent transactions are extremely rare compared to legitimate transactions.  
In this project, the goal is to build a model that can identify fraudulent transactions accurately while keeping false alerts under control.

The final solution is based on:

- **Engineered transaction features**
- **SMOTE** for handling class imbalance
- **XGBoost** for robust prediction on tabular data
- **Two model variants** to compare the impact of the engineered feature `balance_ratio`

---

## 2. Business Problem

In real financial systems, missing fraud can lead to direct financial loss, regulatory issues, and customer trust problems.  
At the same time, too many false fraud alerts create unnecessary investigation cost and degrade customer experience.

So the real objective is not just high accuracy. The model must:

- Catch as many fraud cases as possible
- Keep false positives low
- Work well on highly imbalanced transaction data
- Be explainable enough to support fraud analysis decisions

---

## 3. Dataset Description

The dataset contains transaction-level records with the following core fields:

- `step` – time step of the transaction
- `type` – transaction category such as PAYMENT, TRANSFER, CASH_OUT, etc.
- `amount` – transaction amount
- `oldbalanceOrg` – sender balance before the transaction
- `newbalanceOrig` – sender balance after the transaction
- `oldbalanceDest` – receiver balance before the transaction
- `newbalanceDest` – receiver balance after the transaction
- `isFraud` – target label
- `isFlaggedFraud` – system-generated flag

### Target Variable

- `isFraud = 1` → fraudulent transaction
- `isFraud = 0` → legitimate transaction

Because fraud cases are very rare, the dataset is extremely imbalanced.  
That is why special handling is required during modeling.

---

## 4. Why This Dataset Is Difficult

Fraud detection datasets usually create three major challenges:

1. **Extreme class imbalance**  
   Fraud examples are only a tiny fraction of total records.

2. **Hidden fraud patterns**  
   Fraudulent behavior is not always obvious from one feature alone.

3. **Business trade-off**  
   Missing fraud is expensive, but flagging too many normal transactions is also costly.

This makes the problem much harder than standard classification.

---

## 5. Data Preparation

### 5.1 Transaction Type Encoding

The `type` column is categorical, so it must be converted into numeric form before training the model.

In this project, the transaction type is encoded so the model can learn the difference between categories such as:

- PAYMENT
- TRANSFER
- CASH_OUT
- CASH_IN
- DEBIT

Tree-based models like XGBoost do not directly work with raw text values.  
Encoding converts the transaction category into a machine-readable format.

---

### 5.2 Why We Kept These Features

The final feature set used in the project is:

```python
X = df[['type','amount','oldbalanceOrg','newbalanceOrig','oldbalanceDest','newbalanceDest','org_diff','dest_diff','balance_ratio']]
y = df['isFraud']
```

This feature set was chosen because it captures both the raw transaction details and the balance movement behavior.

---

## 6. Feature Engineering

Feature engineering is the most important part of this project.  
Instead of relying only on raw columns, we created additional features that expose transaction behavior more clearly.

### 6.1 `org_diff`

```python
org_diff = oldbalanceOrg - newbalanceOrig
```

#### Why this feature was created
This feature measures how much the sender balance changed after the transaction.

#### Why it helps fraud detection
For normal transactions, the sender balance should decrease in a consistent way.  
Fraud cases often show unusual or inconsistent balance movement.  
`org_diff` gives the model a direct view of this change.

---

### 6.2 `dest_diff`

```python
dest_diff = newbalanceDest - oldbalanceDest
```

#### Why this feature was created
This feature measures how much the receiver balance changed after the transaction.

#### Why it helps fraud detection
In a valid transaction, the receiver balance should usually increase in a reasonable way.  
Fraudulent transfers often show balance movement that is suspicious or inconsistent.  
This feature captures that signal clearly.

---

### 6.3 `balance_ratio`

```python
balance_ratio = amount / (oldbalanceOrg + 1)
```

#### Why this feature was created
This feature compares the transaction amount with the sender’s available balance.

#### Why it helps fraud detection
A transaction may look normal in absolute amount, but become suspicious when compared to the account balance.

Example:
- A transfer of `50,000` from an account with `5,00,000` may be normal
- A transfer of `50,000` from an account with `2,000` is highly suspicious

This feature normalizes the transaction amount and makes suspicious activity easier to detect.

#### Why it was important in this project
Based on the model comparison, `balance_ratio` improved precision and overall fraud classification performance significantly.  
That is why Model 2, which includes `balance_ratio`, became the better version.

---

## 7. Model Variants Compared

To validate whether the engineered ratio feature was useful, two models were compared:

### Model 1
Features used:

- `type`
- `amount`
- `oldbalanceOrg`
- `newbalanceOrig`
- `oldbalanceDest`
- `newbalanceDest`
- `org_diff`
- `dest_diff`

### Model 2
Features used:

- All Model 1 features
- `balance_ratio`

### Result of Comparison
Model 2 performed much better than Model 1, especially in:

- Precision
- Recall
- F1-score
- PR-AUC

This confirmed that `balance_ratio` was a valuable engineered feature for the fraud detection task.

---

## 8. Handling Class Imbalance with SMOTE

### Why imbalance handling is necessary
Fraud detection data is highly imbalanced.  
If a model is trained directly on the raw dataset, it may learn to predict only the majority class and still appear accurate.

### Why SMOTE was used
SMOTE (Synthetic Minority Oversampling Technique) helps by generating synthetic fraud samples in the training set.

Instead of just copying fraud rows, SMOTE creates new synthetic fraud examples based on neighboring minority-class samples.

### Why SMOTE is useful here
- Improves learning of fraud patterns
- Helps the model pay attention to minority class
- Reduces majority-class bias
- Can improve recall for fraud detection

### Important rule followed
SMOTE is applied **only to the training set**, not the full dataset.  
This avoids data leakage and keeps the evaluation realistic.

---

## 9. Why XGBoost Was Chosen

XGBoost is one of the most effective algorithms for structured/tabular data.  
It is especially strong for fraud detection because it can learn complex relationships between transaction variables.

### Reasons XGBoost fits this problem well

- Handles non-linear relationships very well
- Works strongly on tabular data
- Supports regularization to reduce overfitting
- Learns feature interactions automatically
- Performs well on large datasets
- Gives strong results even when classes are imbalanced

### Why XGBoost over simpler models
Simple models such as Logistic Regression may fail to capture complex fraud patterns.  
Random Forest may become overly confident or overfit depending on feature structure.  
XGBoost gives a better balance of learning power and control.

---

## 10. Model Training Strategy

The final training flow used in the notebook is:

1. Select final features  
2. Encode categorical column `type`
3. Split the data into train and test sets
4. Apply SMOTE only on the training data
5. Train XGBoost
6. Predict probabilities
7. Apply threshold to convert probability into class label
8. Evaluate using fraud-focused metrics

---

## 11. Evaluation Metrics

For fraud detection, accuracy alone is not enough.  
The model is evaluated using:

- **Precision** – how many predicted fraud cases were actually fraud
- **Recall** – how many fraud cases the model successfully detected
- **F1-score** – balance between precision and recall
- **ROC-AUC** – overall ranking quality
- **PR-AUC** – more useful than accuracy for imbalanced classification
- **Confusion Matrix** – shows false positives and false negatives

### Why PR-AUC is important
Since fraud is rare, PR-AUC gives a more honest picture of model performance than accuracy.

---

## 12. Final Model Interpretation

The final model showed strong fraud detection capability with:

- Very high recall
- High precision
- Very strong PR-AUC
- Excellent balance between catching fraud and reducing false alarms

This means the model is suitable as a strong fraud screening system.

---

## 13. Key Findings

- Fraud patterns are highly concentrated in transaction behavior and balance movement
- Engineered features improve fraud detection significantly
- `balance_ratio` is a highly informative variable
- SMOTE helps the model learn minority-class fraud patterns
- XGBoost gives strong performance on this tabular financial dataset
- The final feature set is more powerful than using raw columns alone

---

## 14. Why This Project Is Valuable

This project is valuable because it reflects a real-world fraud analytics workflow:

- data exploration
- feature engineering
- imbalance handling
- model comparison
- fraud-centric evaluation
- business interpretation

It is not just a classification task.  
It is a full fraud detection analysis pipeline.

---

## 15. Future Improvements

Planned next steps for the project include:

- Threshold optimization
- SHAP-based explainability
- Feature stability analysis
- FastAPI deployment
- Streamlit or web app interface
- Model monitoring and retraining strategy

---

## 16. Repository Structure

```bash
Fraud-Detection-System/
│
├── fraud_detection.ipynb
├── dataset.csv
├── fraud_xgb_model.pkl
├── requirements.txt
├── README.md
└── app.py   # deployment will be added later
```

---

## 17. Conclusion

This project demonstrates an end-to-end fraud detection workflow using engineered features, SMOTE, and XGBoost.  
The final model is designed to detect fraud effectively in a highly imbalanced financial dataset while keeping false positives under control.

The strongest part of the project is the feature engineering logic, especially the use of `balance_ratio`, which significantly improved the final model performance.

---

## 18. Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Imbalanced-learn (SMOTE)
- XGBoost
- Matplotlib
- Seaborn
