# <div align="center">Diabetes Prediction using Stacking Ensemble Model</div>

## <div align="center">Overview</div>
This project implements a stacking ensemble model to predict diabetes using the Pima Indians Diabetes Dataset. The dataset includes various health indicators such as glucose levels, BMI, insulin levels, blood pressure, and skin thickness to determine whether a patient has diabetes.
We explore multiple machine learning models, compare their performance, and build an optimized stacked model to improve accuracy. The aim is to provide an in-depth analysis of different models and their strengths when combined.

## <div align="center">Motivation</div>
The motivation behind this project was to:
* Compare the performance of different machine learning models.
* Implement ensemble learning (stacking) to improve model accuracy.
* Apply real-world feature scaling, hyperparameter tuning, and model evaluation.
* Provide a well-documented workflow for ML practitioners and researchers.
* Explore the significance of different health parameters in diabetes prediction.
* Demonstrate how feature engineering and data preprocessing impact model performance.

## <div align="center">Dataset</div>
* The dataset is sourced from the Pima Indians Diabetes Database.
* It consists of 768 observations and 8 key features:
  * **Pregnancies** - Number of times pregnant
  * **Glucose** - Plasma glucose concentration
  * **Blood Pressure** - Diastolic blood pressure (mm Hg)
  * **Skin Thickness** - Triceps skinfold thickness (mm)
  * **Insulin** - 2-Hour serum insulin (mu U/ml)
  * **BMI** - Body Mass Index
  * **Diabetes Pedigree Function** - Genetic risk indicator
  * **Age** - Patient's age in years
* The target variable (Outcome) indicates the presence (1) or absence (0) of diabetes.

## <div align="center">Exploratory Data Analysis (EDA)</div>
To gain deeper insights into the dataset, an interactive **Streamlit dashboard** has been created.

### <div align="center">Explore the EDA Dashboard</div>
<div align="center">[Diabetes Dashboard](https://diabetes-dashboard-84b6cpt5nfnstq4fnk3opf.streamlit.app/?embed_options=show_toolbar,light_theme,show_colored_line,disable_scrolling,show_padding,show_footer,dark_theme)</div>

## <div align="center">Models Implemented</div>
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
* Gradient Boosting Machine (GBM) for additional comparison

Finally, we combined the best models using stacking ensemble learning.

## <div align="center">Performance & Findings</div>
1. **Baseline models performed as follows:**
    * Logistic Regression: ~70.13%
    * SVC: ~61.68%
    * Decision Tree: ~69.48%
    * Random Forest: (~75%)
    * XGBoost: ~69.48%
    * LightGBM: Improved to ~73.38%
    * GBM: ~72.5%
2. **Stacking Model Results:**
    * Using LightGBM as the meta-learner, the final model achieved ~75.97% accuracy, outperforming all individual models.
    * The ensemble method demonstrated improved generalization and robustness.
    * The highest improvement was observed when using a combination of tree-based models.

## <div align="center">Technical Implementation</div>
### <div align="center">1. Data Preprocessing</div>
* Standardization using `StandardScaler`.
* Splitting into train (80%) and test (20%) sets.
* Applied Principal Component Analysis (PCA) for dimensionality reduction.
* Handled missing values through imputation.

### <div align="center">2. Model Training & Evaluation</div>
* Each model was trained separately and evaluated using accuracy, confusion matrix, and classification report.
* Stacking ensemble used multiple base learners and a meta-learner for final predictions.
* Cross-validation was used to assess model stability.

### <div align="center">3. Hyperparameter Optimization</div>
* LightGBM was tuned using:
    * `num_leaves=31`
    * `learning_rate=0.05`
    * `min_child_samples=20`
* GridSearchCV and Bayesian Optimization were applied to refine hyperparameters.

## <div align="center">Why Stacking?</div>
* Instead of relying on a single model, stacking combines multiple models to leverage their strengths.
* Unlike bagging (Random Forest) or boosting (XGBoost, LightGBM), stacking learns from the weaknesses of base models, using a meta-learner to further refine predictions.
* The meta-learner in our case (LightGBM) improved performance by capturing hidden patterns across all models.
* It reduces overfitting and improves generalization across different patient profiles.

## <div align="center">Future Improvements</div>
* Implement Automated Hyperparameter Tuning (GridSearchCV, Bayesian Optimization).
* Use Deep Learning (ANNs) for further improvement.
* Perform Feature Engineering to extract more meaningful data points.
* Explore different stacking architectures for enhanced results.
* Utilize external datasets for better model generalization.

## <div align="center">How to Run the Project</div>
### <div align="center">1. Clone the Repository</div>
```bash
git clone https://github.com/yourusername/diabetes-stacking-model.git
cd diabetes-stacking-model
```

### <div align="center">2. Install Dependencies</div>
```bash
pip install -r requirements.txt
```

### <div align="center">3. Run the Model</div>
```bash
python train_stacking_model.py
```

## <div align="center">Conclusion</div>
This project demonstrates how stacking ensembles can enhance model performance by leveraging the strengths of multiple algorithms. The final stacked model outperforms individual classifiers, making it a powerful approach for real-world classification problems.

## <div align="center">Contributions</div>
Feel free to fork this repository, make improvements, and submit a pull request. Contributions from the ML community are always welcome!

## <div align="center">License</div>
This project is open-source and available under the MIT License.

## <div align="center">Contact</div>
For any questions or discussions, feel free to open an issue or reach out to me via GitHub Discussions.

