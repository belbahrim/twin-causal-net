from twincausal.utils.generator import powers_generator
from twincausal.utils.performance import qini_barplot, qini_curve


def generator(scenario=1, random_state=1234):
    rho = 0
    theta = 0.5
    drop_cols = ['ts']

    scenario_dict = {1: [1 / 2, 10000, 200],
                     2: [1, 20000, 100],
                     3: [2, 20000, 100],
                     4: [4, 20000, 100]}

    if scenario_dict.get(scenario) == None:
        print("Value Not Found, enter a scenario value between 1 and 4")

    scenario_value = scenario_dict[scenario]
    scenario_sigma = scenario_value[0]
    scenario_n = scenario_value[1]
    scenario_p = scenario_value[2]

    # Generate synthetic data
    df = powers_generator(scenario_n, scenario_p, rho, scenario_sigma, theta, scenario, random_state)
    # _, true_qini = qini_curve(df['treat'], df['outcome'], df['ts'], 1, False)
    # _, true_tau = qini_barplot(df['treat'], df['outcome'], df['ts'], 1, False)
    # print('True Qini coefficient:', true_qini)
    # print('True Uplift correlation:', true_tau)

    df = df.drop(drop_cols, axis=1)
    T = df["treat"].values.reshape((-1, 1))
    Y = df["outcome"].values.reshape((-1, 1))
    X = df.iloc[:, 2:].values

    return X, T, Y
