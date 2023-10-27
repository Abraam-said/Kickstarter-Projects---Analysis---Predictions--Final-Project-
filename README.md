# Kickstarter Projects - Analysis + Predictions (Final Project)

## Kickstarter Project Success Prediction

### **Dataset Link**: [https://kickstarter-project-success-predictor.streamlit.app/](https://www.kaggle.com/datasets/kemical/kickstarter-projects?select=ks-projects-201801.csv)

### Overview
Kickstarter, founded in 2009, has transformed the landscape of crowdfunding by providing a platform for creators to connect with supporters globally. This analysis delves into Kickstarter projects, focusing on predicting their success before launch. This predictive insight empowers creators, ensuring their ideas have a higher likelihood of success from the outset.

## Libraries Used
### EDA (Exploratory Data Analysis) Libraries
- pandas
- numpy
- plotly.express
- plotly.figure_factory
- plotly.subplots
- plotly.graph_objects
- seaborn
- matplotlib.pyplot
- pycountry

### Data Preprocessing Libraries
- datasist.structdata
- sklearn.model_selection
- sklearn.preprocessing
- category_encoders
- datasist.structdata
- sklearn.impute
- imblearn.over_sampling

### Machine Learning (Classification Models) Libraries
- sklearn.linear_model (LogisticRegression)
- sklearn.neighbors (KNeighborsClassifier)
- sklearn.svm (SVC)
- sklearn.naive_bayes (GaussianNB)
- sklearn.tree (DecisionTreeClassifier)
- sklearn.ensemble (RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier)
- sklearn.feature_selection (SequentialFeatureSelector, SelectKBest, f_regression, RFE, SelectFromModel)
- imblearn.pipeline
- sklearn.compose
- sklearn.metrics (confusion_matrix, accuracy_score, f1_score, classification_report, roc_curve, roc_auc_score)
- lightgbm
- xgboost
- sklearn.model_selection (GridSearchCV)

### Model Deployment Libraries
- joblib
- streamlit

## Purpose
The purpose of this project is to assist creators on Kickstarter by predicting the success of their projects before they are launched. By leveraging a variety of machine learning models and data preprocessing techniques, this analysis aims to provide valuable insights that enhance the probability of a project's success.

## How to Use
- Data Exploration: Utilize the EDA libraries to explore the dataset, understanding its features and patterns.
- Data Preprocessing: Use data preprocessing techniques to clean, transform, and prepare the data for machine learning models.
- Model Selection: Experiment with different classification models provided in the analysis to choose the most suitable one for the specific Kickstarter project.
- Model Evaluation: Evaluate the selected model's performance using metrics such as accuracy, F1 score, and ROC AUC score.
- Deployment: Deploy the chosen model using the provided model deployment libraries for real-time predictions.

 ## **App link**: https://kickstarter-project-success-predictor.streamlit.app/
