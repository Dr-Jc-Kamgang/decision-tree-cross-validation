# Decision Tree Classification with Cross-Validation
## Project Overview
This project implements a Decision Tree classification model and evaluates its performance using cross-validation.

The main objective is to study model performance, detect potential overfitting, and select suitable hyperparameters using GridSearchCV.

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
The validation scores obtained for the five folds were approximately:

- Fold 1: 0.9778
- Fold 2: 0.9286
- Fold 3: 0.9890
- Fold 4: 0.9423
- Fold 5: 0.9533

The average validation score was approximately 0.9582.

The difference between the training and validation scores was also analyzed for each fold to assess the risk of overfitting.