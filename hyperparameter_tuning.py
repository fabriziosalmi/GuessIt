from sklearn.model_selection import GridSearchCV
from keras.wrappers.scikit_learn import KerasClassifier
from model_architecture import create_model
from progress_monitor import log_info
import numpy as np

def perform_hyperparameter_tuning(X_train, y_train):
    """
    Perform hyperparameter tuning for the model.
    :param X_train: Training features.
    :param y_train: Training labels.
    :return: Best model after tuning.
    """
    # Define the model for hyperparameter tuning
    model = KerasClassifier(build_fn=create_model, verbose=0)

    # Define the grid search parameters
    param_grid = {
        'batch_size': [10, 20, 50],
        'epochs': [10, 50, 100],
        'input_shape': [X_train.shape[1]],  # Assuming X_train is numpy array
        'num_classes': [np.unique(y_train).shape[0]]
    }

    grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1, cv=3)
    grid_result = grid.fit(X_train, y_train)

    # Summarize results
    log_info("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
    for params, mean_score, scores in grid_result.cv_results_['params'], grid_result.cv_results_['mean_test_score'], grid_result.cv_results_['std_test_score']:
        log_info("%f (%f) with: %r" % (mean_score, scores, params))
    
    return grid_result.best_estimator_

# Example Usage:
# best_model = perform_hyperparameter_tuning(X_train, y_train)
