# Mobile Phone Price Prediction

This machine learning project predicts the price range of mobile phones based on technical specifications such as RAM, battery capacity, camera quality and more. A Random Forest Classifier is used to classify mobile phones into one of four price categories.


## Dataset

mobile_phone_pricing.zip - contains dataset.csv 

Target variable: price_range (0 = Low Cost, 1 = Medium Cost, 2 = High Cost, 3 = Very High Cost)


## Workflow

- Preprocessing: Standardized features using StandardScaler and removed duplicate records.
- Exploratory Data Analysis (EDA): Correlation heatmap visualized top feature relationships.
- Model Training: Splits data into training (80%) and testing (20%) sets. Trains a RandomForestClassifier model on scaled training data.
- Evaluation: Evaluates model accuracy, confusion matrix and classification report on test data. Saves the trained model and scaler as .pkl files
- Feature Importance Visualization: Plots the top 10 features contributing most to the model’s decisions.


## Libraries Used

pandas

numpy

scikit-learn

matplotlib

seaborn

joblib

zipfile


## Results

Model Accuracy: 0.8925

Confusion Matrix:
[[101   4   0   0]
 [  5  79   7   0]
 [  0   6  80   6]
 [  0   0  15  97]]

Classification Report:
              precision    recall  f1-score   support

           0       0.95      0.96      0.96       105
           1       0.89      0.87      0.88        91
           2       0.78      0.87      0.82        92
           3       0.94      0.87      0.90       112

    accuracy                           0.89       400
   macro avg       0.89      0.89      0.89       400
weighted avg       0.90      0.89      0.89       400


<img width="1536" height="754" alt="correlation_heatmap" src="https://github.com/user-attachments/assets/8f428981-82b1-45de-b9bd-6ae06c772ca9" />

<img width="640" height="480" alt="top_10_features" src="https://github.com/user-attachments/assets/96986a7b-5c76-4eb3-a94e-fc631a4d06f4" />

