# Comparative Analysis of ML Models using Nested Cross-Validation

Rigorously evaluated and compared four machine learning algorithms — **k-Nearest Neighbors**, **SVM**, **Decision Tree**, and **Random Forest** — on a real-world credit application dataset to identify the most reliable, high-performing model.

The core of this project is the design of a **Nested Cross-Validation** framework to ensure that model evaluation is statistically robust, unbiased, and free from data leakage.

![Python](https://img.shields.io/badge/Python-3.x-blue.svg?logo=python&logoColor=white)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-Model_Validation-F7931E.svg?logo=scikit-learn&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-Data_Preprocessing-150458.svg?logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-Scientific_Computing-4D77CF.svg?logo=numpy&logoColor=white)

---

## 🎯 Project Goal

The primary goal was to move beyond simple train/test splits and implement a best-practice evaluation pipeline to produce a reliable, unbiased estimate of each model's generalization performance on unseen data. This involves two key challenges:
1.  **Preventing Data Leakage:** Ensuring that information from the test set *never* influences the training or hyperparameter tuning process.
2.  **Avoiding "Lucky" Tunings:** Ensuring that a model's high score isn't just the result of hyperparameters that were "overfit" to a specific validation set.

---

## 🔬 Methodology

### 1. Preprocessing Pipeline
A `ColumnTransformer` and `Pipeline` were constructed to handle the mixed-type credit dataset.
* **Categorical Features:** Processed using `OneHotEncoder`.
* **Numerical Features:** Scaled using `MinMaxScaler`.

**Crucially, this entire preprocessing pipeline was included *inside* the inner loop of the nested cross-validation. This is a best practice that prevents data leakage**, as the `MinMaxScaler` is fit *only* on the inner training folds, and the test fold is transformed using that fit (never fit on the test data).

### 2. Nested Cross-Validation Framework

A two-loop system was implemented to separate hyperparameter tuning from model evaluation.

* **Outer Loop (`RepeatedStratifiedKFold`):**
    * **Purpose:** To produce a reliable, final performance score for a *tuned* model.
    * **Process:** Splits the *entire* dataset into 10 folds. It iterates 10 times, holding one fold out as a final, unseen test set. This entire 10-fold process is repeated 5 times with different shuffles to get a stable statistical distribution of scores.

* **Inner Loop (`GridSearchCV`):**
    * **Purpose:** To find the best hyperparameters for a given model.
    * **Process:** For *each* of the 10 outer loops, the inner loop is initialized with the 9 training folds. It then performs its *own* 3-fold cross-validation on this data for every possible hyperparameter combination.
    * The `GridSearchCV` object finds the best-performing hyperparameters from the inner loop and is then retrained on all 9 training folds.

* **Final Evaluation:** The tuned model from the inner loop is finally evaluated *once* on the "unseen" test fold from the outer loop. The result is a list of 50 independent performance scores (10 folds x 5 repeats) for each algorithm, providing a highly reliable mean accuracy and confidence interval.

> **[Image: Diagram of Nested Cross-Validation]**
>
> *(**Developer Note:** A diagram showing the outer loop splitting data for testing and the inner loop (GridSearchCV) running its own CV for tuning would be perfect here.)*

---

## 📊 Results & Model Comparison

After running all four algorithms through the identical nested cross-validation framework, the **Random Forest** was the clear and consistent winner.

| Model | Mean Accuracy |
| :--- | :---: |
| **Random Forest** | **0.76** |
| SVM | 0.74 |
| Decision Tree | 0.70 |
| k-Nearest Neighbors (KNN) | 0.72 |

*The Random Forest consistently outperformed all other models, achieving a mean accuracy of 0.76 and a high F₁ score, confirmed through repeated testing across folds.*

---

## 🧠 Model Interpretation

To understand *why* the models were making their decisions, further analysis was performed:

* **Random Forest (Winner):** A **Feature Importance** analysis was conducted on the tuned Random Forest model to identify which data features (e.g., credit history, account balance) were the most predictive of a positive outcome.
* **SVM:** The **Support Vectors** were inspected to understand which data points were most critical in defining the decision boundary between "approve" and "deny."

> **[Image: Bar chart of Random Forest feature importances]**
>
> *(**Developer Note:** Place the bar chart showing the most important features from your Random Forest model here.)*

---

## 🚀 How to Run

1.  **Clone** the repository.
2.  **Install** required libraries:
    ```bash
    pip install numpy pandas scikit-learn matplotlib
    ```
3.  **Run** the main analysis script from your terminal:
    ```bash
    python cross_validation_analysis.py
    ```
4.  The script will execute the full 5x10 nested cross-validation for all four models and print the final comparative performance metrics.