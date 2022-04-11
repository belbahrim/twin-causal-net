import numpy as np
import pandas as pd
from scipy.stats import norm
import warnings

def generate_data(sample_size, nb_features, rho, sigma, theta, nb_main_effects=10, nb_treat_effects=10,
                  main_effect=0.3, uplift_effect=0.2, interaction_effect=0.2, seed=10):
    """
    :param sample_size: number of samples
    :param nb_features: number of features
    :param rho: correlation between features
    :param sigma: overall noise
    :param theta: probability of treatment allocation
    :param nb_main_effects: number of non-zero main effects (must be less then number of features)
    :param nb_treat_effects: number of non-zero treatment effects (must be less then number of features)
    :param seed: seed for reproductibility
    :param main_effect: magnitude of predictors main effects
    :param uplift_effect: standard deviation for treatment effect gamma
    :param interaction_effect: magnitude of predictors interactions effect
    :return: a synthetic data frame for evaluating uplift models
    """

    # Set a seed for reproductibility
    np.random.seed(seed)

    # Generate nb_predictors from a multivariate Gaussian distribution
    mu = [0] * nb_features
    cov_mat = np.zeros((nb_features, nb_features)) + rho
    np.fill_diagonal(cov_mat, 1)

    features = np.random.multivariate_normal(mu, cov_mat, sample_size)

    # Generate noise from a standard Gaussian distribution
    eps = np.random.normal(0, sigma, sample_size)

    # Generate treatment allocation randomly with probability theta -- This is for Randomized setting
    treat = np.random.binomial(1, theta, sample_size)

    # Main effects
    beta = np.zeros(nb_features)
    for j in range(nb_main_effects):
        beta[j] = main_effect * (-1) ** j

    # Treatment effects
    gamma = np.zeros(nb_features)

    for j in range(nb_treat_effects):
        gamma[j] = np.random.normal(0, uplift_effect)

    # Interaction effects
    beta_int = np.zeros(nb_features - 1)
    for j in range(nb_features - 1):
        beta_int[j] = interaction_effect * (-1) ** j

    features_int = features[:, :(nb_features - 1)] * features[:, 1:]

    # Response variable
    outcome = 1 * (features.dot(beta) + features_int.dot(beta_int) + (features * treat.reshape((sample_size, 1))).dot(
        gamma) + eps > 0)

    # "True" uplift
    true_uplift = norm.cdf((features.dot(beta + gamma) + features_int.dot(beta_int)) / sigma) - norm.cdf(
        (features.dot(beta) + features_int.dot(beta_int)) / sigma)

    # Concatenate in a pandas data frame
    data = np.concatenate((true_uplift.reshape((sample_size, 1)), outcome.reshape((sample_size, 1)),
                           treat.reshape((sample_size, 1)), features), axis=1)

    feature_names = ['ts', 'outcome', 'treat']
    for i in range(nb_features):
        feature_names.append('X' + str(i + 1))

    data = pd.DataFrame(data, columns=feature_names)
    return data


def powers_generator(sample_size, nb_features, rho, sigma, theta, scenario=1, seed=10):
    """
    :param sample_size: number of samples
    :param nb_features: number of features
    :param rho: correlation between features
    :param sigma: overall noise
    :param theta: probability of treatment allocation
    :return: a synthetic data frame for evaluating uplift models
    """

    # Set a seed for reproductibility
    np.random.seed(seed)

    # Generate nb_predictors from a multivariate Gaussian distribution
    mu = [0] * nb_features
    cov_mat = np.zeros((nb_features, nb_features)) + rho
    np.fill_diagonal(cov_mat, 1)

    features = np.random.multivariate_normal(mu, cov_mat, sample_size)
    for j in np.arange(int(nb_features / 2)):
        features[:, 2 * j + 1] = np.random.binomial(1, 0.5, sample_size)

    # Generate noise from a standard Gaussian distribution
    eps = np.random.normal(0, sigma, sample_size)

    # Generate treatment allocation randomly with probability theta -- This is for Randomized setting
    treat = np.random.binomial(1, theta, sample_size)
    # For observational data (no randomization) -- you need to define theta as a function of some covariates

    # Functions from Powers et al. (2018)
    f1_x = np.zeros(sample_size)
    f2_x = 5 * (features[:, 0] > 1) - 5
    f3_x = 2 * features[:, 0] - 4
    f4_x = features[:, 1] * features[:, 3] * features[:, 5] + 2 * features[:, 1] * features[:, 3] * (1 - features[:, 5])
    + 3 * features[:, 1] * (1 - features[:, 3]) * features[:, 5] + 4 * features[:, 1] * (1 - features[:, 3]) * (
            1 - features[:, 5])
    + 5 * (1 - features[:, 1]) * features[:, 3] * features[:, 5] + 6 * (1 - features[:, 1]) * features[:, 3] * (
            1 - features[:, 5])
    + 7 * (1 - features[:, 1]) * (1 - features[:, 3]) * features[:, 5] + 8 * (1 - features[:, 1]) * (
            1 - features[:, 3]) * (1 - features[:, 5])
    f5_x = features[:, 0] + features[:, 2] + features[:, 4] + features[:, 6] + features[:, 7] + features[:, 8] - 2
    f6_x = 4 * (features[:, 0] > 1) * (features[:, 2] > 0) + 4 * (features[:, 4] > 1) * (
            features[:, 6] > 0) + 2 * features[:, 7] * features[:, 8]
    f7_x = 0.5 * (features[:, 0] ** 2 + features[:, 1] + features[:, 2] ** 2 + features[:, 3]
                  + features[:, 4] ** 2 + features[:, 5] + features[:, 6] ** 2 + features[:, 7] + features[:,
                                                                                                  8] ** 2 - 11)
    f8_x = (1 / np.sqrt(2)) * (f4_x + f5_x)

    # Scenarios
    scenario_dict = {#   1:[f8_x, f1_x],
                     #   2:[f5_x, f2_x],
                     #   3:[f4_x,f3_x],
                        1:[f7_x, f4_x],
                        2:[f3_x, f5_x],
                     #   6:[f1_x,f6_x],
                        3:[f2_x, f7_x],
                        4:[f6_x, f8_x]}
    scenario_value = scenario_dict[scenario]
    mu_x = scenario_value[0]
    tau_x = scenario_value[1]

    # Response variable
    # outcome = 1 * (mu_x + (treat - 0.5) * tau_x + eps > 0)
    outcome = 1 * (mu_x + treat * tau_x + eps > 0)

    # "True" uplift
    # true_uplift = norm.cdf((mu_x + 0.5 * tau_x) / sigma) - norm.cdf((mu_x - 0.5 * tau_x) / sigma)
    true_uplift = norm.cdf((mu_x + tau_x) / sigma) - norm.cdf(mu_x / sigma)

    # Concatenate in a pandas data frame
    data = np.concatenate((true_uplift.reshape((sample_size, 1)), outcome.reshape((sample_size, 1)),
                           treat.reshape((sample_size, 1)), features), axis=1)

    feature_names = ['ts', 'outcome', 'treat']
    for i in range(nb_features):
        feature_names.append('X' + str(i + 1))

    data = pd.DataFrame(data, columns=feature_names)

    return data
