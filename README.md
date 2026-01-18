# 🚢 Titanic - Machine Learning from Disaster

This repository contains a comprehensive machine learning pipeline to predict passenger survival on the Titanic. The project benchmarks multiple classification algorithms to identify the most effective predictive model.

## 📊 Project Overview
The Titanic shipwreck is one of the most infamous maritime disasters in history. Using data from the passenger manifest, this project builds predictive models to answer the question: “What sorts of people were more likely to survive?” using features like socio-economic status, gender, and age.

## 🛠️ Technical Stack
* **Language:** Python 3.12
* **Libraries:**
  * **Data Analysis:** `Pandas`, `NumPy`
  * **Visualization:** `Seaborn`, `Matplotlib`
  * **Machine Learning:** `Scikit-Learn`, `XGBoost`, `CatBoost`

## 🚀 Key Features

### 1. Data Cleaning & EDA
* Identified missing values: **177** in `Age` and **687** in `Cabin`.
* Generated a **Pearson Correlation Heatmap** using `Seaborn` to visualize the relationship between features and the target variable (`Survived`).

### 2. Feature Engineering
* **Categorical Encoding:** Utilized `LabelEncoder` to transform the `Sex` column into a machine-readable numerical format.
* **Feature Selection:** Systematically removed low-correlation and high-cardinality features: `PassengerId`, `Name`, `Age`, `Ticket`, `Fare`, `Embarked`, `Cabin`, and `Parch`.

### 3. Model Benchmarking
Five distinct classification algorithms were implemented and evaluated using the **F1-Score** to ensure balanced precision and recall:

* **Logistic Regression:** Achieved a baseline F1-score of **0.7536**.
* **Random Forest:** Configured with `n_estimators=50` and `entropy` criterion.
* **Decision Tree:** Implemented with `max_depth=50`.
* **CatBoost:** Leveraged gradient boosting on decision trees.
* **XGBoost:** Utilized for high-performance gradient boosting.

## 📂 File Structure
* `notebook6d17e35d2e.ipynb`: Main analysis and modeling notebook.
* `/kaggle/input/titanic/`: Source dataset files (`train.csv`, `test.csv`).

## 📈 Results Summary
The Logistic Regression model served as a strong baseline with a competitive F1-score, while the ensemble methods (Random Forest, XGBoost, CatBoost) provided a framework for further hyperparameter optimization.

---
*Developed as part of a Kaggle Competition analysis.*
