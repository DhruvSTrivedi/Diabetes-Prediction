# Diabetes Prediction using Stacking Ensemble Model

## Overview
This project implements a stacking ensemble model to predict diabetes using the Pima Indians Diabetes Dataset. The dataset includes various health indicators such as glucose levels, BMI, and insulin levels to determine whether a patient has diabetes.
We explore multiple machine learning models, compare their performance, and build an optimized stacked model to improve accuracy.

## Motivation
The motivation behind this project was to:
* Compare the performance of different machine learning models.
* Implement ensemble learning (stacking) to improve model accuracy.
* Apply real-world feature scaling, hyperparameter tuning, and model evaluation.
* Provide a well-documented workflow for ML practitioners and researchers.

## Dataset
* The dataset is sourced from the Pima Indians Diabetes Database.
* It consists of 768 observations and 8 features.
* The target variable (Outcome) indicates the presence (1) or absence (0) of diabetes.

## Exploratory Data Analysis (EDA)
To gain deeper insights into the dataset, an interactive **Streamlit dashboard** has been created.

### **Explore the EDA Dashboard**:
[Diabetes Dashboard](https://diabetes-dashboard-84b6cpt5nfnstq4fnk3opf.streamlit.app/?embed_options=show_toolbar,light_theme,show_colored_line,disable_scrolling,show_padding,show_footer,dark_theme)

## Models Implemented
We implemented multiple models to evaluate performance:
* Logistic Regression
* Support Vector Classifier (SVC)
* Decision Tree Classifier
* Random Forest Classifier
* K-Means Clustering
* Naive Bayes (GaussianNB)
* XGBoost Classifier
* K-Nearest Neighbors (KNN)
* LightGBM Classifier (used in stacking ensemble)

Finally, we combined the best models using stacking ensemble learning.

## Performance & Findings
1. **Baseline models performed as follows:**
    * Logistic Regression: ~70.13%
    * SVC: ~61.68%
    * Decision Tree: ~69.48%
    * Random Forest: (~75%)
    * XGBoost: ~69.48%
    * LightGBM: Improved to ~73.38%
2. **Stacking Model Results:**
    * Using LightGBM as the meta-learner, the final model achieved ~75.97% accuracy, outperforming all individual models.
    * The ensemble method demonstrated improved generalization and robustness.

## Technical Implementation
### 1. Data Preprocessing
* Standardization using `StandardScaler`.
* Splitting into train (80%) and test (20%) sets.
* Applied Principal Component Analysis (PCA) for dimensionality reduction.

### 2. Model Training & Evaluation
* Each model was trained separately and evaluated using accuracy, confusion matrix, and classification report.
* Stacking ensemble used multiple base learners and a meta-learner for final predictions.

### 3. Hyperparameter Optimization
* LightGBM was tuned using:
    * `num_leaves=31`
    * `learning_rate=0.05`
    * `min_child_samples=20`
* The tuning helped improve stability and prevent overfitting.

## Why Stacking?
* Instead of relying on a single model, stacking combines multiple models to leverage their strengths.
* Unlike bagging (Random Forest) or boosting (XGBoost, LightGBM), stacking learns from the weaknesses of base models, using a meta-learner to further refine predictions.
* The meta-learner in our case (LightGBM) improved performance by capturing hidden patterns across all models.

## Future Improvements
* Implement Automated Hyperparameter Tuning (GridSearchCV, Bayesian Optimization).
* Use Deep Learning (ANNs) for further improvement.
* Perform Feature Engineering to extract more meaningful data points.
* Explore different stacking architectures for enhanced results.

## How to Run the Project
### 1. Clone the Repository
```bash
git clone https://github.com/DhruvSTrivedi/diabetes-stacking-model.git
cd diabetes-stacking-model
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the Model
```bash
python train_stacking_model.py
```

## Conclusion
This project demonstrates how stacking ensembles can enhance model performance by leveraging the strengths of multiple algorithms. The final stacked model outperforms individual classifiers, making it a powerful approach for real-world classification problems.

## Contributions
Feel free to fork this repository, make improvements, and submit a pull request. Contributions from the ML community are always welcome!

## License
This project is open-source and available under the MIT License.

## Contact
For any questions or discussions, feel free to open an issue or reach out to me via email.

