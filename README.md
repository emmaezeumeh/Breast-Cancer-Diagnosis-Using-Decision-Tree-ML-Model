# Breast-Cancer-Diagnosis-Using-Decision-Tree-ML-Model

![Breast-cancer-awareness-different-women](https://user-images.githubusercontent.com/115907457/218345813-e51582e9-b7f0-4e72-a264-e12a4cb096a9.jpg)

## Task

The goal is to predict breast cancer, specifically the presence of a malignant tumor in a patient, depending on the features/predictors in the dataset. The predictors in the dataset for this project are discrete and the outcome is binary, determining if the tumor is either malignant or benign. 

This is a classification problem and so Decision Tree, a machine learning classification method, was chosen and used to fit a function to predict the class of new data points. To improve the accuracy of my model, I used stratified KFold cross validation and then GridSearchCV to tune the hyperparameters in my ML classifier. 

## Dataset Description
Source: https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+%28Diagnostic%29
The features are computed from a digitized image of a fine needle aspirate (FNA) of a breast mass. They describe characteristics of the cell nuclei present in the image. The number of data points in the dataset are 569, with 30 features.

## Result

### Training Set:

AUC: 1.000

Accuracy: 0.993

Recall: 1.000

Precision: 0.990

Specificity: 0.982
 
### Test Set:

AUC: 0.943

Accuracy: 0.921

Recall: 0.958

Precision: 0.919

Specificity: 0.860


## Dependencies

Python

SKLearn

Numpy

Pandas

![636114372309137984-10 10 16EarlyDetection](https://user-images.githubusercontent.com/115907457/218345831-5e924ded-f157-4bfa-847b-0af24e3ccb7a.jpg)


<!-- ![image](https://user-images.githubusercontent.com/115907457/218345366-8e301628-0d0b-4f52-94d7-2440c99d263b.png) -->
