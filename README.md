🚢 Titanic - Machine Learning from Disaster
This project implements a predictive pipeline to determine passenger survival on the Titanic. It utilizes various machine learning classification algorithms and focuses on feature selection and model benchmarking to achieve robust results.
📊 Project Overview
The goal of this project is to predict whether a passenger survived the Titanic shipwreck based on features such as ticket class and sex. This is a classic binary classification problem solved using the Python data science ecosystem.
🛠️ Tech Stack
Language: Python 3.12

Libraries: * Data Handling: Pandas, NumPy

Visualization: Seaborn, Matplotlib

Machine Learning: Scikit-Learn, XGBoost, CatBoost, RandomForestClassifier, DecisionTreeClassifier

🚀 Workflow
1. Exploratory Data Analysis (EDA)
Analyzed missing values across the dataset (identifying 177 missing Age values and 687 missing Cabin values).

Visualized feature correlations using a Pearson Correlation Heatmap to identify variables with the highest predictive power for survival.

2. Data Preprocessing
Feature Selection: Systematically dropped irrelevant or high-cardinality features including PassengerId, Name, Age, Ticket, Fare, Embarked, Cabin, and Parch.

Categorical Encoding: Converted the Sex feature into numerical format using LabelEncoder to make it compatible with machine learning models.

3. Model Benchmarking
The project evaluates five different classification models using the F1-Score to ensure a balance between precision and recall:

Model,Evaluation Metric
Logistic Regression,F1-Score: ~0.75
Random Forest,Entropy-based ensemble
Decision Tree,Deep tree classification
CatBoost,Gradient boosting on decision trees
XGBoost,Optimized distributed gradient boosting
