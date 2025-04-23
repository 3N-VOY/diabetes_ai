# Diabetes Prediction Analysis

This project analyzes the [Healthcare-Diabetes dataset](https://www.kaggle.com/datasets/nanditapore/healthcare-diabetes) to predict diabetes status b
ased on clinical measurements. The analysis is performed as part of the Data Mining and Foundations of AI course.

## Problem Statement

**Research Question:**  
Can we predict diabetes status based on patients’ clinical measurements?

**Objectives:**  
- Explore potential predictors associated with diabetes  
- Develop and compare machine learning models for diabetes prediction  
- Provide insights for early screening and risk assessment

## Dataset

- **Source:** [Kaggle - Healthcare-Diabetes](https://www.kaggle.com/datasets/nanditapore/healthcare-diabetes)
- **Rows:** 2,768 patients
- **Features:** Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age, Outcome

## Approach

1. **Exploratory Data Analysis (EDA):**
   - Checked for missing and physiologically implausible values (e.g., zeros in Glucose, BMI, etc.)
   - Replaced zero values with the median of non-zero values for each feature
   - Visualized feature distributions and compared diabetic vs. non-diabetic patients
   - Analyzed feature importance and correlations

2. **Model Building:**
   - **Logistic Regression:** Linear baseline model
   - **Decision Tree:** Interpretable model
   - **Random Forest:** Ensemble model for higher accuracy

3. **Evaluation:**
   - Used accuracy, precision, recall, F1-score, and ROC-AUC for model comparison
   - Visualized confusion matrices and ROC curves

## Results

- **Random Forest** achieved the highest accuracy (~98%) and best overall performance.
- **Decision Tree** performed well (~83% accuracy) and provided interpretable decision rules.
- **Logistic Regression** served as a strong baseline (~75% accuracy).
- **Most important predictors:** Glucose, BMI, Age, and Pregnancies.

## Clinical Implications

- **Screening Priority:** Patients with high glucose, BMI, and age should be prioritized for diabetes screening.
- **Risk Assessment:** The Random Forest model can be adapted for clinical risk assessment tools.
- **Interpretability:** The Decision Tree model offers clear decision rules for clinical use.

## Limitations & Future Work

- **Data Quality:** Imputation of zero values may introduce bias.
- **Feature Set:** Additional features (e.g., lifestyle, family history) could improve predictions.
- **Future Directions:** Model deployment, longitudinal analysis, and external validation.

## How to Run

1. Clone the repository:
   ```bash
   git clone https://github.com/3N-VOY/diabetes_ai.git
   ```
2. Open Diabetes_Analysis.ipynb in Jupyter Notebook or Google Colab.
3. Ensure Healthcare-Diabetes.csv is in the same directory.
4. The analysis was performed using Python 3.8+ and the following libraries:
  - pandas
  - numpy
  - matplotlib
  - seaborn
  - scikit-learn
  ```bash
  pip install pandas numpy matplotlib seaborn scikit-learn
  ```
5. Run all cells to reproduce the analysis.

## References
   * [Healthcare-Diabetes Dataset on Kaggle](https://python.langchain.com/docs/introduction/](https://www.kaggle.com/datasets/nanditapore/healthcare-diabetes)
