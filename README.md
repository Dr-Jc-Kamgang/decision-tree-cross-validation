# Decision Tree Classification with Cross-Validation
## Project Overview
This project implements a Decision Tree classification model and evaluates its performance using cross-validation.

The main objective is to study model performance, detect potential overfitting, and select suitable hyperparameters using GridSearchCV.

## Dataset: 
- ** Dataset: ** Breast Cancer Wisconsin
- ** Source:** scikit-learn
- ** Function: ** 'load_breast_cancer()'
- ** Type: ** Binary classification
- ** Objective: ** Predict whether a tumor is malignant or benign

# Methodology
The poject follows these main steps:

1. Split the dataset into training and testing sets.
2. Train a Decision Tree classifier.
3. Evaluate the model on the training and test data.
4. Apply 5-fold cross- validation.
5. Analyze the difference between training and validation scores to assess overfitting.
6. Use GridSearchCV to optimize the model hyperparameters.
7. Evaluate the optimized model on the test set.
## Results
## Initial Decision Tree Model

The initial decision Tree Model achieved:

. Training accuracy: 1.00000
.Test accuracy: 0.9473
. Train-Test gap: 0.0526

The perfect training score combined with a lower test score indicates some degree of overfitting.

## Cross-Validation
A 5-fold cross-validation was performed to evaluate the model's generalization perforrmance.

Validation scores
. Fold 1: 0.9122
. Fold 2: 0.9O35
. Fold 3: 0.9298
. Fold 4: 0.9561
. Fold 5: 0.8849

Mean cross-validation accuracy: 0.9173

The results show that the model maintains relatively stable performance accros the different folds, although some variation can be observed.

### Hyperparameter Optimization with GridSearchCV

GridSearchCV was used to optimize the Decision Tree hyperparameters.

Best parameters:

. max_depth = 4
. min_samples_leaf = 1
.min_samples_split = 10

Best cross-validation score: 0.9384

The optimized model achieved:

. Training accuracy: 0.9934
. Test accuracy: 0.9385
.Train-Test gap: 0.0548


## Fold-by-Fold Analysis of the Optimized model
Fold  Training Score   Validation Score   Gap
1     0.9670           0.9451             0.0220
2     0.9835           0.9121             0.0714
3     0.9890           0.9780             0.0110
4     0.9808           0.9231             0.0577
5     0.9808           0.9341             0.0467

Mean train-validation gap: 0.0418

Overall, the optimized Decision Tree provides a good balance between training and validation performance. The reduced tree depth and minimum sample constraints help control model complexity and limit overfitting. 

## Technologies

. Python
. Numpy
. Pandas
. Scikit-learn
. Decision Tree
. Cross-validation
. GridSearchCV
. Git & Github


## Conclusion

The Decision Tree model achieved strong predictive performance on the Breast Cancer Wisconsin dataset.

Cross-validation and hyperparameter optimization with GridSearchCV helped evaluate the model's generalization ability and control overfitting.

The project demonstrates a complete machine learning workflow, from model training and validation to hyperparameter optimization and performance analysis.


