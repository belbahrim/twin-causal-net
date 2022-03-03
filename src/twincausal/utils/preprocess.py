import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from imblearn.over_sampling import RandomOverSampler


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def class_transform(outcome, treat):
    """
    :param outcome: y variable
    :param treat: treatment variable
    :return: transformed outcome variable
    """
    prop_treat = np.sum(treat) / len(treat)
    outcome_transform = np.zeros(len(outcome))
    ## try to broadcast w/o for loop
    for i in range(len(outcome)):
        if treat[i] == 1:
            outcome_transform[i] = 0.5 * (outcome[i] / prop_treat) + 0.5 * (- (1 - outcome[i]) / prop_treat)
        else:
            outcome_transform[i] = 0.5 * (-outcome[i] / (1 - prop_treat)) + 0.5 * ((1 - outcome[i]) / (1 - prop_treat))
    return outcome_transform


def bound_variable(data, variable, threshold=1):
    """
    :param data: data to bound
    :param variable: variable to bound
    :param threshold: percentage threshold
    """
    lower_bound = np.percentile(data[variable], threshold)
    upper_bound = np.percentile(data[variable], 100 - threshold)
    data.loc[data[variable] >= upper_bound, variable] = upper_bound
    data.loc[data[variable] <= lower_bound, variable] = lower_bound


def bound_all_variables(data, threshold=1):
    """
    :param data: data to bound
    :param threshold: percentage threshold
    """
    for column in data.columns:
        bound_variable(data, column, threshold)


class UpliftDataset(Dataset):
    """
    Dataset for uplift
    """

    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


def read_and_split_data(data, test_size, outcome_var, treatment_var, seed, standardize=False):
    """
    :param data: data to split
    :param test_size: proportion of the data to sample for the test set
    :param outcome_var: y variable
    :param treatment_var: treatment variable
    :param seed: seed
    :param standardize: if the data should be standardized or not
    :return: train-test split data
    """
    df = data.copy()
    bound_all_variables(df)

    ros_train_all, ros_test_all = train_test_split(df, test_size=test_size, stratify=df[[outcome_var, treatment_var]],
                                   random_state=seed)
    
    temp = class_transform(ros_train_all[outcome_var].copy().values,
                                                   ros_train_all[treatment_var].copy().values)
    
    ros_train_all.insert(len(ros_train_all.columns), 'class_transform', temp)

    temp1  = class_transform(ros_test_all[outcome_var].copy().values,
                                                      ros_test_all[treatment_var].copy().values)
    ros_test_all.insert(len(ros_test_all.columns), 'class_transform', temp1)
    
    # Generating the target variables
    y_train = ros_train_all[outcome_var].values
    y_test = ros_test_all[outcome_var].values

    y_up_train = ros_train_all["class_transform"].values
    y_up_test = ros_test_all["class_transform"].values

    # List of the columns of the dataframe in order to remove treatment and outcome from the normalization step
    list_columns = list(ros_train_all.columns)
    list_columns.remove(outcome_var)
    list_columns.remove(treatment_var)
    list_columns.remove("class_transform")

    X_train_treatment = ros_train_all[treatment_var].copy().values
    X_train = ros_train_all.drop([outcome_var], axis=1)
    X_train = X_train.drop(["class_transform"], axis=1)
    X_train_zero = X_train.copy()

    X_test_treatment = ros_test_all[treatment_var].copy().values
    X_test = ros_test_all.drop([outcome_var], axis=1)
    X_test = X_test.drop(["class_transform"], axis=1)
    X_test_zero = X_test.copy()

    # Assingning the value 1 to treatment for both the test and train set
    X_train[treatment_var] = 1
    X_test[treatment_var] = 1

    X_train_zero[treatment_var] = 0
    X_test_zero[treatment_var] = 0

    if standardize:
        # Normalisation Train
        X_train[list_columns] = (X_train[list_columns] - X_train[list_columns].min()) / (
                X_train[list_columns].max() - X_train[list_columns].min())
        X_train_zero[list_columns] = (X_train_zero[list_columns] - X_train_zero[list_columns].min()) / (
                X_train_zero[list_columns].max() - X_train_zero[list_columns].min())

        # Normalisation Test
        X_test[list_columns] = (X_test[list_columns] - X_test[list_columns].min()) / (
                X_test[list_columns].max() - X_test[list_columns].min())
        X_test_zero[list_columns] = (X_test_zero[list_columns] - X_test_zero[list_columns].min()) / (
                X_test_zero[list_columns].max() - X_test_zero[list_columns].min())

    # retaining variables that we want and putting the treatment variable at the end
    X_train = X_train[list_columns + [treatment_var]].values
    X_train_zero = X_train_zero[list_columns + [treatment_var]].values
    X_test = X_test[list_columns + [treatment_var]].values
    X_test_zero = X_test_zero[list_columns + [treatment_var]].values
    #X_train is X_train_one
    return X_train, X_train_zero, X_train_treatment, y_train, y_up_train, X_test, X_test_zero, X_test_treatment, y_test, y_up_test


def concat_create_dataset(X_one, X_zero, X_treatment, y, y_up, input_with_interaction=False):
    # The true input matrix is X
    X = np.concatenate([X_one[:, :-1], X_treatment.reshape((X_treatment.shape[0], 1))], axis=1)

    # If we work without any hidden layer, we need to create the interactions X times T in the model
    X_interaction_treat = X[:, :-1] * 1
    X_interaction_control = X[:, :-1] * 0

    # Now, we can concatenate the inputs and outputs for train and test sets (we keep X_train_one and X_train_zero
    # at the end
    # to be able to go get the artificial treatment variable forced to 1 and 0 respectively in training and inference
    if input_with_interaction:  # need to add explicit interaction terms
        X_concat = np.concatenate(
            [X_interaction_treat, X_one, X_interaction_control, X_zero,
             X_treatment.reshape((X_treatment.shape[0], 1))],
            axis=1)
    else:  # should not work for LogisticSMITE
        X_concat = np.concatenate(
            [X_one, X_zero,
             X_treatment.reshape((X_treatment.shape[0], 1))],
            axis=1)

    # Response variable and DataLoader
    y_concat = np.concatenate([y.reshape((len(y), 1)), y_up.reshape((len(y_up), 1))], axis=1)
    dataset = UpliftDataset(X_concat, y_concat)

    return dataset
