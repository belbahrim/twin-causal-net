from twincausal.utils.generator import powers_generator
from twincausal.utils.performance import qini_barplot, qini_curve
import numpy as np

def generator(scenario):
        seed = 33
        rho = 0
        theta = 0.5
        drop_cols = ['ts']

        scenario_dict = {4:[1/2,10000,200],5:[1,20000,100],6:[1,20000,100],7:[2,20000,100],8:[4,20000,100]}
        if scenario_dict.get(scenario) == None: print("Value Not Found, enter a scenario value between 4 and 8")
        scenario_value = scenario_dict[scenario]
        scenario_sigma = scenario_value[0]
        scenario_n = scenario_value[1]
        scenario_p = scenario_value[2]

        # Generate synthetic data
        df = powers_generator(scenario_n, scenario_p, rho, scenario_sigma, theta, scenario, seed)
        _, true_qini = qini_curve(df['treat'], df['outcome'], df['ts'], 1, False)
        true_risk, true_tau = qini_barplot(df['treat'], df['outcome'], df['ts'], 1, False)
        print('True adjusted Qini:', np.max((0, true_qini)) * true_tau)
        print('True Risk:', true_risk)
        df = df.drop(drop_cols, axis=1)
        T = df["treat"].values.reshape((-1,1))
        Y = df["outcome"].values.reshape((-1,1))
        X = df.iloc[:,2:].values
        return X,T,Y