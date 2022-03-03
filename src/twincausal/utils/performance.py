import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import rankdata, kendalltau


def qini_curve(treat, outcome, uplift_prob_, p_precision=2, plotit=True, name=""):
    """
    Returns a qini curve table using final predicted uplifts from the model
    and the associated qini coefficient

    Parameters:
    :param treat - Traetment group for each sample (Whether the sample is in Test/Conrol group)
    :param outcome -  Outcome for the given sample (Sale happened or not)
    :param uplift_prob_ - Predicted uplift probabilities by the model for sorting the customers/samples
    :param p_precision - Precision for ranking quantiles (if 1 it rounds the quantiles to the first decimal and
                  generates Qini for deciles, if 2 it rounds the quantiles to the second decimal and
                  generates Qini for percentiles, and so on...)
    :param plotit - Plot the Qini curve
    """

    # Create a dataframe for the input variables
    outcome = np.array(outcome).reshape(-1,1)
    treat = np.array(treat).reshape(-1,1)
    uplift_prob_ = np.array(uplift_prob_).reshape(-1,1)
    d = np.concatenate((outcome, treat, uplift_prob_), axis=1)
    df = pd.DataFrame(d, columns=['outcome', 'treat', 'pred_uplift'])
    df = df.sort_values(by='pred_uplift', ascending=False).reset_index(drop=True)

    # Sorting and ranking the samples/customers using the predicted uplifts
    df['rank'] = np.round(rankdata(-df['pred_uplift'], method='max') / df.values.shape[0], p_precision)

    # Building cummulative sales and actual uplifts for input data for each of the groups
    no_of_groups = np.unique(df['rank'])
    no_of_groups = no_of_groups[no_of_groups > 0]
    qini_table = np.zeros((len(no_of_groups), 7))
    for i in range(len(no_of_groups)):
        subset = df.loc[df['rank'] <= no_of_groups[i]]
        qini_table[i, 0] = no_of_groups[i]
        qini_table[i, 1] = subset.loc[subset['treat'] == 1].loc[subset['outcome'] == 1].shape[0]  # sales_in_Test
        qini_table[i, 2] = subset.loc[subset['treat'] == 1].shape[0]
        qini_table[i, 3] = subset.loc[subset['treat'] == 0].loc[subset['outcome'] == 1].shape[0]  # sales_in_control
        qini_table[i, 4] = subset.loc[subset['treat'] == 0].shape[0]

        # Actual Uplift using the Normalized sales values for control & test groups
        qini_table[i, 5] = qini_table[i, 1] - qini_table[i, 3] * qini_table[i, 2] / qini_table[i, 4]

    # Cummulative acutal uplifts calculated
    qini_table[:, 6] = qini_table[:, 5] / qini_table[len(no_of_groups) - 1, 2] * 100

    # Uplift values from the model for plotting Qini
    x_axis = np.concatenate((np.zeros(1), qini_table[:, 0]))
    y_axis = np.concatenate((np.zeros(1), qini_table[:, 6]))

    # Uplift values by random targeting
    x_rand = np.array((0, 1))
    y_rand = np.array((0, qini_table[len(no_of_groups) - 1, 6]))

    # Build dataframe for the Decile table
    qini_table_to_return = pd.DataFrame(qini_table)
    qini_table_to_return.columns = ['%_of_list', 'Sales_in_Test', 'Obs_in_Test', 'Sales_in_Control', 'Obs_in_Control',
                                    'Norm_Cum_Uplift', 'Cum_Uplift(%)']

    # Compute the Qini coefficient
    nb = len(qini_table_to_return)
    qini_coeff = qini_table_to_return.values[0, 6] / 2 * qini_table_to_return.values[0, 2] / \
                 qini_table_to_return.values[nb - 1, 2]
    for i in range(1, nb):
        qini_coeff += (qini_table_to_return.values[i, 6] + qini_table_to_return.values[i - 1, 6]) / 2 * (
                qini_table_to_return.values[i, 2] / qini_table_to_return.values[nb - 1, 2] -
                qini_table_to_return.values[i - 1, 2] / qini_table_to_return.values[nb - 1, 2])

    qini_coeff -= qini_table_to_return.values[nb - 1, 6] / 2

    if plotit:
        # plot the qini curve
        plt.title('Qini Curve')
        plt.xlabel('Proportion of Treated Observations')
        plt.ylabel('Incremental Positive Outcomes (%)')
        plt.plot(x_axis, y_axis, '--r', label='Uplift Model')
        plt.plot(x_rand, y_rand, '--b', label='Random')
        plt.legend(loc='lower center', bbox_to_anchor=(1.4, 0.0), shadow=True, ncol=1)
        if name != "":
            plt.savefig(name)
        plt.show(block=True)

    return qini_table_to_return, qini_coeff


def qini_barplot(treat, outcome, uplift_prob_, p_precision=1, plotit=True):
    """
    Returns a barplot table using final predicted uplifts from the model
    and the associated uplift kendall's correlation

    Parameters:
    :param treat - Traetment group for each sample (Whether the sample is in Test/Conrol group)
    :param outcome -  Outcome for the given sample (Sale happened or not)
    :param uplift_prob_ - Predicted uplift probabilities by the model for sorting the customers/samples
    :param p_precision - Precision for ranking quantiles (if 1 it rounds the quantiles to the first decimal and
                  generates barplot for deciles, if 2 it rounds the quantiles to the second decimal and
                  generates barplot for percentiles, and so on...)
    """

    # Create a dataframe for the input variables
    outcome = np.array(outcome).reshape(-1, 1)
    treat = np.array(treat).reshape(-1, 1)
    uplift_prob_ = np.array(uplift_prob_).reshape(-1, 1)
    d = np.concatenate((outcome, treat, uplift_prob_), axis=1)
    df = pd.DataFrame(d, columns=['outcome', 'treat', 'pred_uplift'])
    df = df.sort_values(by='pred_uplift', ascending=False).reset_index(drop=True)

    # Sorting and ranking the samples/customers using the predicted uplifts
    df['rank'] = np.round(rankdata(-df['pred_uplift'], method='max') / df.values.shape[0], p_precision)

    # Build BarPlot
    no_of_groups = np.unique(df['rank'])
    no_of_groups = no_of_groups[no_of_groups > 0]
    table = np.zeros((len(no_of_groups), 12))
    for i in range(len(no_of_groups)):
        if i == 0:
            subset = df.loc[df['rank'] <= no_of_groups[i]]
        else:
            subset = df.loc[(no_of_groups[i - 1] < df['rank']) & (df['rank'] <= no_of_groups[i])]
        table[i, 0] = no_of_groups[i] * 100
        table[i, 1] = subset.loc[subset['treat'] == 1].loc[subset['outcome'] == 1].shape[0]
        table[i, 2] = subset.loc[subset['treat'] == 1].shape[0]
        table[i, 3] = subset.loc[subset['treat'] == 0].loc[subset['outcome'] == 1].shape[0]
        table[i, 4] = subset.loc[subset['treat'] == 0].shape[0]
        table[i, 5] = "{0:.4f}".format(subset['pred_uplift'].mean())
        table[i, 6] = table[i, 1] / table[i, 2] * 100
        table[i, 7] = table[i, 3] / table[i, 4] * 100
        table[i, 8] = table[i, 6] - table[i, 7]
        table[i, 9] = "{0:.4f}".format(subset['pred_uplift'].min())  # min uplift in that particular bin
        table[i, 10] = "{0:.4f}".format(subset['pred_uplift'].max())  # max uplift in that particular bin
        table[i, 11] = (table[i, 1] + table[i, 3]) * 100 / df.outcome.sum()  # %sales captured in that bin

    # Build dataframe for the Decile table
    table_to_return = pd.DataFrame(table)
    table_to_return.columns = ['%_of_list', 'Sales_in_Test', 'Obs_in_Test', 'Sales_in_Control', 'Obs_in_Control',
                               'avg_pred_uplift', 'sale%_in_Test', 'sale%_in_Control', 'observed_uplift(%)',
                               'min_pred_uplift', 'max_pred_uplift', '%sales_captured']

    # Compute Kendall's rank correlation
    observed_uplift_rank = rankdata(-table_to_return['observed_uplift(%)'], method='average')
    predicted_uplift_rank = np.array(range(1, len(table_to_return) + 1))
    uplift_kendalltau = np.round(kendalltau(observed_uplift_rank, predicted_uplift_rank)[0], 2)
    uplift_risk = np.mean((observed_uplift_rank - predicted_uplift_rank)**2)

    if plotit:
        # Create bars
        plt.bar(np.arange(len(table_to_return)), table_to_return['observed_uplift(%)'])
        # Create names on the x-axis
        plt.xticks(np.arange(len(table_to_return)), table_to_return['%_of_list'])
        plt.title('Uplift Barplot')
        plt.xlabel('Proportion or Targeted Individuals')
        plt.ylabel('Policy Sales (%)')
        # Show graphic
        plt.show(block=True)

    return uplift_risk, uplift_kendalltau
