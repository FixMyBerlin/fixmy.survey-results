from functools import partial
from pathlib import Path
from os.path import join
import argparse
import logging
import json
import mord
from sklearn.model_selection import cross_val_score
# HyperOpt Parameter Tuning
from hyperopt import STATUS_OK, tpe, Trials, hp, fmin, space_eval
import pandas as pd

from src.features.build_features import feature_engineering
from src.data import load_full_data

PROJECT_DIR = Path(__file__).resolve().parents[2]


def parse_args():
    parser = argparse.ArgumentParser(description="Model Training")
    parser.add_argument("--experiment", type=str, default=None,
                        help="Experiment data to load and train the model on")
    return parser.parse_args()


def objective(params, data):
    """Objective function for Logistic Regression Hyperparameter Tuning"""

    # Perform n_fold cross validation with hyperparameters
    # Use early stopping and evaluate based on ROC AUC
    X, y = data

    clf = mord.LogisticAT(**params)
    scores = cross_val_score(clf, X, y, cv=5,
                             scoring='neg_mean_absolute_error')

    # Extract the best score
    mean_score = scores.mean()

    # Loss must be minimized
    loss = - 1 * mean_score
    # Dictionary with information for evaluation
    return {'loss': loss, 'params': params, 'status': STATUS_OK}


def train_model(experiment):
    if experiment is None:
        raise AttributeError("Please choose an experiment for training")
    logger = logging.getLogger(__name__)
    logger.info(('loading data for experiment ' + experiment))
    dataset = load_full_data()
    dataset = dataset[(dataset["Experiment"] == experiment)
                      & (dataset["Kamera"] == "C")]
    dataset = dataset.drop(["Experiment", "Kamera"],
                           axis=1).dropna(axis=1,
                                          how="all")
    logger.info('transform data')
    dataset = feature_engineering(dataset)

    y = dataset["rating"].astype('int32')
    X = dataset.drop(["rating"], axis=1)
    space = {
        'alpha': hp.uniform('alpha', 0.005, 8),
        'max_iter': hp.choice('max_iter', range(5, 2000))
    }

    # Algorithm
    tpe_algorithm = tpe.suggest

    # Trials object to track progress
    bayes_trials = Trials()
    partial_obj = partial(objective, data=(X, y))

    logger.info('hyperparameter optimization')
    # Optimize
    best = fmin(fn=partial_obj,
                space=space,
                algo=tpe_algorithm,
                max_evals=40,
                trials=bayes_trials)

    logger.info('retrain model on best parameters and full dataset')
    # Retrain on best hyperparameters
    best_parameters = space_eval(space, best)
    json.dump(best_parameters, open(join(PROJECT_DIR,
                                         "models",
                                         experiment + "_parameters.json"),
                                    "w"))
    ord_reg = mord.LogisticAT(**best_parameters).fit(X, y)
    logger.info('save feature importances')
    top_parameters = pd.Series(ord_reg.coef_, index=X.columns)
    top_parameters.to_csv(join(PROJECT_DIR,
                               "models",
                               experiment + "_feature_importance.csv"))


if __name__ == "__main__":
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    args = parse_args()  # read cli parameters
    if args.experiment == "all":
        experiments = ["MS", "CP", "SE"]
        for ex in experiments:
            train_model(ex)
    else:
        train_model(args.experiment)
