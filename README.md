# House-Price-Prediction

![image](https://github.com/him100gupta/House-Price-Prediction/assets/29143253/fa21f98a-6163-4f7a-8909-27abf5cef67d)

## Overview
This project leverages advanced machine learning algorithms to develop a robust predictive model for estimating residential property values. The model is trained on a comprehensive dataset obtained from Kaggle, a renowned platform for data science competitions.
The primary objective of this endeavor is to accurately forecast the final sale prices of houses by analyzing a multitude of relevant features encompassed within the dataset. Through rigorous model training and optimization, the developed solution achieves an impressive accuracy of **92%**, translating to a remarkable Kaggle score of **0.50498**. This outstanding performance has secured a prestigious rank of **4432** on the competition's leaderboard, a testament to the model's exceptional predictive capabilities.
By harnessing the power of machine learning techniques and leveraging the rich feature set provided, this project demonstrates the potential of data-driven approaches in the real estate domain, enabling more informed decision-making and enhancing the overall understanding of factors influencing property valuations.

## Libraries Used
+ **For Data Manipulation and Analysis : -** Pandas, NumPy.
+ **Statistics and Modeling: -** Statsmodels, Scipy.
+ **Data Visualization: -** Matplotlib, Seaborn.
+ **Machine Learning: -** Sklearn, Xgboost, LightGBM, Catboost.
+ **Ensemble Learning: -** Mlxtend.
+ **Utility: -** Warnings, Math.

## Comprehensive Approach to predict House price.

+ **Data Preprocessing:** Handled missing values using imputation techniques, encoded categorical variables using various methods such as target encoding, one-hot encoding, label encoding, and manual mapping based on domain knowledge. Performed feature scaling using log transformation and Yeo-Johnson power transformation.
+ **Exploratory Data Analysis:** Conducted data visualization to understand feature distributions and relationships with the target variable using histograms, Spearman's correlation for numerical columns, and box plots and ANOVA tests for categorical columns to select relevant features.
+ **Feature Engineering:** Created new features such as House_Age, Remodel_status, Total_Bathroom, Total_Porch_SF, Have_Garage and HeatingQuality to improve model performance.
+ **Model Training:** Trained various machine learning models including linear regression, decision trees etc.
+ **Hyperparameter Tuning:** Optimized model parameters for better performance using GridSearch.
+ **Stacked Modeling:** Created a final stacked model to enhance predictive performance by combining multiple base models.
+ **Model Evaluation:** Assessed model performance using metrics such as MAE, MSE, RMSE, R^2 and Adjusted_R2 score.

## Results
+ **Model Accuracy: -** 92.1224%
+ **Kaggle Score: -** 0.50498
+ **Leaderboard Rank: -** 4432(On 12/05/2024)


