"""
Copyright 2017 Ronald J. Nowling

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import random

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import chi2
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import log_loss

N_SAMPLES = 100
N_SIMS = 100
CORR_PROBS = [1.0, -1.0, 0.95, -0.95, 0.9, -0.9, 0.85, -0.85, 0.8, -0.8, 0.75, -0.75, 0.7, -0.7, 0.65, -0.65, 0.6, -0.6, 0.5, -0.5]

def generate_binary_data(n_samples, corr_probs):
    """
    Generate labels and binary features for data from two classes.  The
    probabilities given in `corr_probs` determine the probability that a
    feature's value will agree with the sample's label.  A negative
    probability indicates that the feature's value should be the inverse
    of the label.  For uncorrelated features, use a probability of 0.5.

    Returns a vector of labels and matrix of features.
    """
    n_features = len(corr_probs)
    features = np.zeros((n_samples, n_features))
    labels = np.zeros(n_samples)

    for r in range(n_samples):
        labels[r] = random.randint(0, 1)

            
    for i, p in enumerate(corr_probs):
        inverted = p < 0.
        p = np.abs(p)
        if inverted:
            for r in range(n_samples):
                if random.random() < p:
                    features[r, i] = 1 - labels[r]
                else:
                    features[r, i] = labels[r]
        else:
            for r in range(n_samples):
                if random.random() < p:
                    features[r, i] = labels[r]
                else:
                    features[r, i] = 1 - labels[r]

    return labels, features

def needed_sgd_iter(n_samples):
    """
    Return number of the number of SGD iterations (epochs) needed
    based on the number of samples using advice from
    http://scikit-learn.org/stable/modules/sgd.html#tips-on-practical-use
    """
    return max(20,
               int(np.ceil(10**6 / n_samples)))

def likelihood_ratio_test(features_alternate, labels, lr_model, features_null=None):
    """
    Compute the likelihood ratio test for a model trained on the set of features in
    `features_alternate` vs a null model.  If `features_null` is not defined, then
    the null model simply uses the intercept (class probabilities).  Note that
    `features_null` must be a subset of `features_alternative` -- it can not contain
    features that are not in `features_alternate`.

    Returns the p-value, which can be used to accept or reject the null hypothesis.
    """
    labels = np.array(labels)
    features_alternate = np.array(features_alternate)
    
    if features_null:
        features_null = np.array(features_null)
        
        if features_null.shape[1] >= features_alternate.shape[1]:
            raise(ValueError, "Alternate features must have more features than null features")
        
        lr_model.fit(features_null, labels)
        null_prob = lr_model.predict_proba(features_null)[:, 1]
        df = features_alternate.shape[1] - features_null.shape[1]
    else:
        null_prob = sum(labels) / float(labels.shape[0]) * \
                    np.ones(labels.shape)
        df = features_alternate.shape[1]
    
    lr_model.fit(features_alternate, labels)
    alt_prob = lr_model.predict_proba(features_alternate)

    alt_log_likelihood = -log_loss(labels,
                                   alt_prob,
                                   normalize=False)
    print(null_prob)
    null_log_likelihood = -log_loss(labels,
                                    null_prob,
                                    normalize=False)

    G = 2 * (alt_log_likelihood - null_log_likelihood)
    p_value = chi2.sf(G, df)

    return p_value

def plot_pvalues(flname, p_values, title):
    log_p_values = np.log10(p_values)
    plt.clf()
    plt.boxplot(x=log_p_values)
    plt.xlabel("Variable", fontsize=16)
    plt.ylabel("P-Value (log10)", fontsize=16)
    plt.title(title, fontsize=18)
    plt.savefig(flname, DPI=200)
    plt.show()

if __name__ == "__main__":
    # burn in
    for i in range(100):
        random.random()


    model = SGDClassifier(loss="log",
                          penalty="l2",
                          max_iter=needed_sgd_iter(N_SAMPLES))

    print("Feature Details:")
    for i in range(len(CORR_PROBS)):
        inverted = CORR_PROBS[i] < 0.
        print("Feature:", i, "Corr Prob:", np.abs(CORR_PROBS[i]), "Inverted:", inverted)


    feature_log_p_values = np.zeros((N_SIMS, len(CORR_PROBS)))
    for j in range(N_SIMS):
        labels, features = generate_binary_data(N_SAMPLES, CORR_PROBS)

        print("Trial:", (j+1))
        for i in range(len(CORR_PROBS)):
            # force into Nx1 matrix
            column = features[:, i].reshape(-1, 1)
            # print("1",column)
            # print("2",labels)
            p_value = likelihood_ratio_test(column,
                                            labels,
                                            model)
            feature_log_p_values[j, i] = p_value
            #inverted = CORR_PROBS[i] < 0.
            #print "Feature:", i, "Corr Prob:", np.abs(CORR_PROBS[i]), "Inverted:", inverted, "Likelihood Ratio Test p-value:", p_value

    plot_pvalues("p_values_boxplot.png",
                 feature_log_p_values,
                 "")