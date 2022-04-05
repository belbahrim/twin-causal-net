### Importing the modules and library

# Importing the modules of the twincausal library
from twincausal.utils.performance import qini_barplot, qini_curve
from twincausal.core.test import device_agno
from twincausal.proximal.proximal import LinearProximal
from twincausal.utils.logger import Logger
from twincausal.utils.preprocess import read_and_split_data, concat_create_dataset
from twincausal.losses.losses import indirect_loss, uplift_loss, likelihood_version_loss
from twincausal.proximal.proximal import PGD
from twincausal.models.models import LinearProximal
from twincausal.core.test import test_data
from twincausal.utils.generator import powers_generator

# Importing Pytorch and tensorboard
import torch
from torch import nn
import torch.nn.functional as functional
from torch.utils.data import DataLoader
from torch.nn import Linear
from torch.autograd import Variable
from tensorboardX import SummaryWriter

# Importing general libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Importing sys libraries
import warnings
from os import path
from datetime import datetime
import time, sys


def variable_name(vname):
    vnames = [name for name in globals() if globals()[name] is vname]
    return vnames[0]


def hyperparameter_value(hyperparameters):
    for i in hyperparameters:
        hyperparameter_name = i[0]
        hyperparameter_value = i[1]
        # Printing the int based hyperparameters/ configs used
        if type(hyperparameter_value) == int:
            print('%-25s%-12i' % (hyperparameter_name, hyperparameter_value))
        # Printing the float based hyperameters/ configs used
        if type(hyperparameter_value) == float:
            print('%-25s%-12i' % (hyperparameter_name, hyperparameter_value))
        # Printing the bool based hyperameters/ configs used
        if type(hyperparameter_value) == bool:
            print('%-25s%-12i' % (hyperparameter_name, hyperparameter_value))

def update_progress(progress):
    """
    Shows the progress in the terminal
      """
    barLength = 20  # Modify this to change the length of the progress bar
    status = ""
    if isinstance(progress, int):
        progress = float(progress)
    if not isinstance(progress, float):
        progress = 0
        status = "error: progress var must be float\r\n"
    if progress < 0:
        progress = 0
        status = "Halt...\r\n"
    if progress >= 1:
        progress = 1
        status = "Done...\r\n"
    block = int(round(barLength * progress))
    text = "\rPercent: [{0}] {1}% {2}".format("#" * block + "-" * (barLength - block), progress * 100, status)
    sys.stdout.write(text)
    sys.stdout.flush()


def call(model, inputs_0, inputs_1, treatment):
    return model(inputs_0, inputs_1, treatment)


class twin_causal(nn.Module):
    def __init__(self, input_size, hlayers=1, nb_neurons=256, lrelu_slope=0, batch_size=256, shuffle=True, max_iter=100,
                 learningRate=0.005, reg_type=1, l_reg_constant=0.001, prune=True, gpl_reg_constant=0.0001, loss="uplift_loss",
                 learningCurves=True, save_model=False, verbose=False, logs=False, random_state=1234):
        super().__init__()
        self.input_size = input_size
        self.hlayers = hlayers
        self.lrelu_slope = lrelu_slope
        self.nb_neurons = nb_neurons
        self.prune = prune
        self.reg_type = reg_type
        self.epochs = max_iter
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.learningRate = learningRate
        self.seed = random_state
        self.verbose = verbose
        self.logs = logs
        self.loss = loss
        self.l_reg_constant = l_reg_constant  # regularization on the weights
        self.gpl_reg_constant = gpl_reg_constant
        self.save_model = save_model
        self.logger = Logger("twincausal", "{}".format(nb_neurons))
        self.learningCurves = learningCurves
        self.list_qini_test = []

        linear_cls = nn.Linear if not prune else LinearProximal

        self.fc_layer = linear_cls(input_size, nb_neurons)
        if prune:
            self.scaling_a = nn.Parameter(torch.randn(nb_neurons).abs_())
            self.scaling_b = nn.Parameter(torch.randn(nb_neurons).abs_())
        self.fc_output = linear_cls(nb_neurons, 1)

        if self.verbose:
            print("hlayers", self.hlayers)
        if hlayers > 1:
            self.prune = False
            self.nb_neurons_per_layer = nb_neurons
            self.fc_layer1 = nn.Linear(input_size, self.nb_neurons_per_layer)
            self.fc_layer2 = nn.Linear(self.nb_neurons_per_layer, self.nb_neurons_per_layer)
            self.fc_output = nn.Linear(self.nb_neurons_per_layer, 1)

    def forward(self, x_treat, x_control, treat):
        if self.hlayers == 1:
            x1 = self.fc_layer(x_treat)
            if self.prune:
                x1 = x1.mul(self.scaling_a - self.scaling_b)
            x1 = functional.leaky_relu(x1, self.lrelu_slope)
            x1 = self.fc_output(x1)
            m11 = torch.sigmoid(x1)

            x0 = self.fc_layer(x_control)
            if self.prune:
                x0 = x0.mul(self.scaling_a - self.scaling_b)
            x0 = functional.leaky_relu(x0, self.lrelu_slope)
            x0 = self.fc_output(x0)
            m10 = torch.sigmoid(x0)

            m1t = m11 * treat + m10 * (1.0 - treat)

            return m1t, m11, m10
        elif self.hlayers == 2:
            x1 = self.fc_layer1(x_treat)
            x1 = functional.leaky_relu(x1, self.lrelu_slope)
            x1 = self.fc_layer2(x1)
            x1 = functional.leaky_relu(x1, self.lrelu_slope)
            x1 = self.fc_output(x1)

            x0 = self.fc_layer1(x_control)
            x0 = functional.leaky_relu(x0, self.lrelu_slope)
            x0 = self.fc_layer2(x0)
            x0 = functional.leaky_relu(x0, self.lrelu_slope)
            x0 = self.fc_output(x0)

            x_return = x1 * treat + x0 * (1.0 - treat)
            x_return = torch.sigmoid(x_return)

            p1 = torch.sigmoid(x1)
            p0 = torch.sigmoid(x0)

            return x_return, p1, p0
        else:
            x1 = self.fc_layer1(x_treat)
            x1 = functional.leaky_relu(x1, self.lrelu_slope)
            x1 = self.fc_layer2(x1)
            x1 = functional.leaky_relu(x1, self.lrelu_slope)
            for x in range(self.hlayers - 2):
                x1 = self.fc_layer2(x1)
                x1 = functional.leaky_relu(x1, self.lrelu_slope)
            x1 = self.fc_output(x1)

            x0 = self.fc_layer1(x_control)
            x0 = functional.leaky_relu(x0, self.lrelu_slope)
            x0 = self.fc_layer2(x0)
            x0 = functional.leaky_relu(x0, self.lrelu_slope)
            for x in range(self.hlayers - 2):
                x0 = self.fc_layer2(x0)
                x0 = functional.leaky_relu(x0, self.lrelu_slope)
            x0 = self.fc_output(x0)

            x_return = x1 * treat + x0 * (1.0 - treat)
            x_return = torch.sigmoid(x_return)

            p1 = torch.sigmoid(x1)
            p0 = torch.sigmoid(x0)

            return x_return, p1, p0

    def fit(self, X, treat, Y, test_size=0.3):

        if X.shape[1] != self.input_size - 1:
            raise ValueError('The number of inputs provided is different from the input size of the network')
        list_variable_instance = list(self.__dict__.items())
        list_variable_instance = list_variable_instance[9:]
        hyperparameter_value(list_variable_instance)

        seed = self.seed
        reg_type = self.reg_type
        device = device_agno()
        standardize = False
        con = np.concatenate((Y, treat, X), axis=1)
        df = pd.DataFrame(con)
        if self.verbose:
            print(df.columns.to_list())
        outcome_var = "outcome"
        treatment_var = "treat"
        df = df.rename(columns={0: outcome_var, 1: treatment_var})
        if self.verbose:
            print(df.columns.to_list())
        train_test, out_of_sample = train_test_split(df, test_size=test_size, stratify=df[[outcome_var, treatment_var]],
                                                     random_state=seed)
        X_train_one, X_train_zero, X_train_treatment, y_train, y_up_train, X_test_one, X_test_zero, X_test_treatment, y_test, y_up_test = read_and_split_data(
            data=train_test,
            test_size=test_size,
            outcome_var=outcome_var,
            treatment_var=treatment_var,
            seed=seed,
            standardize=standardize)

        train_dataset = concat_create_dataset(X_train_one, X_train_zero, X_train_treatment, y_train, y_up_train,
                                              False)
        test_dataset = concat_create_dataset(X_test_one, X_test_zero, X_test_treatment, y_test, y_up_test,
                                             False)
        shuffle = self.shuffle
        batch_size = self.batch_size
        train_dataloader = DataLoader(train_dataset, batch_size, shuffle)
        test_dataloader = DataLoader(test_dataset, batch_size, shuffle)

        input_size = self.input_size
        nb_samples = df.shape[0]  # for reporting only

        torch.manual_seed(seed)

        learningRate = self.learningRate
        prune = self.prune
        criterion = torch.nn.BCELoss()
        criterion_uplift = indirect_loss
        if self.loss == "uplift_loss":
            criterion_new = uplift_loss
        elif self.loss == "likelihood_loss":
            criterion_new = likelihood_version_loss
        else:
            print("custom loss")
            criterion_new = self.loss

        if prune:
            optimizer = PGD(self.parameters(), lr=learningRate)
            optim_used = 'PGD'
        else:
            optimizer = torch.optim.SGD(self.parameters(), lr=learningRate)
            optim_used = 'SGD'

        logs = self.logs
        if logs == True:
            scenario = 8
            data_name = 'scen_' + str(scenario)
            log_name = "testing_loggers"
            reporting_path = "./runs/Test11111111111/"
            reporting_filename = "0001_Test.csv"

            if not path.exists(reporting_path + reporting_filename):
                first_row = ["Data", "Nb_Samples", "Input_Size", "Input_Int", "OOS_Size", "Seed", "Model", "Alpha",
                             "Lambda_Weights", "Lambda_Nodes", "H_Layer", "Nb_H_Units", "Rank", "LReLU_NSlopes",
                             "Batch_Size", "lr", "Nb_Epochs",
                             "Best_Epoch", "CV_Metric", "aQini_Train", "aQini_Test", "aQini_OOS", "Risk_Train",
                             "Risk_Test",
                             "Risk_OOS", "Optim", "Nb_H_Units_Pruning"]
                # print("----------==--------=-=-=-=-=-=-=-able to access")
            # If folder does not exist, this line will check and create it
            # logger._make_dir(reporting_path)
            # Append the first row of the reporting file
            # logger.append_list_as_row(reporting_path + reporting_filename, first_row)

        _, _, _, _, _, training_err_list, qini_list_epoch = self.training1(train_dataloader, test_dataloader, criterion,
                                                                           criterion_uplift, criterion_new, optimizer,
                                                                           prune, reg_type)
        self.list_qini_test = qini_list_epoch
        return training_err_list, qini_list_epoch

    def training1(self, train_dataloader, test_dataloader, criterion, criterion_uplift, criterion_new, optimizer, prune,
                  reg_type=1):
        logs = self.logs
        model_type = 'IE'
        l_reg_constant = self.l_reg_constant  # regularization on the weights
        gpl_reg_constant = self.gpl_reg_constant  # regularization on the scaling factors
        reg_constant = 0
        epochs = self.epochs
        device = device_agno()
        input_size = self.input_size
        logdir = "testing_writer1/logs/scalars/" + datetime.now().strftime("%Y%m%d-%H%M%S")
        writer = SummaryWriter(logdir)
        # Train the model
        train_qini_res_per_epoch = []
        train_risk_res_per_epoch = []

        log_interval = 50
        nb_non_zero_neurons_per_epoch = []

        # Create a logger to save the runs for Tensorboard
        # if logs:
        train_err_list = []
        test_err_list = []
        train_err_list_check = []
        qini_list = []
        val_err_list = []
        val_qini_list = []
        stopping_flag = 0
        for epoch in range(epochs):
            if self.verbose:
                print(f"---------- Start epoch {epoch} ----------------")
            epoch_train_loss_propensity = 0
            epoch_train_loss_uplift = 0
            epoch_total_batch = 0
            for batch_idx, (X_train_batch, y_train_batch) in enumerate(train_dataloader):
                # Converting inputs and labels to Variable

                if model_type == 'TO':
                    inputs_0 = Variable(X_train_batch[:, :input_size].to(device))
                    inputs_1 = Variable(X_train_batch[:, input_size:2 * input_size].to(device))
                    inputs_2 = Variable(
                        X_train_batch[:, 2 * input_size].to(device).reshape(X_train_batch[:, 2 * input_size].shape[0],
                                                                            1))
                    labels = Variable(y_train_batch.to(device))
                    labels_propensity = labels[:, 0]
                    labels_uplift = labels[:, 1]
                elif model_type == 'IE':
                    inputs_0 = Variable(X_train_batch[:, :input_size].to(device))
                    inputs_1 = Variable(X_train_batch[:, input_size:2 * input_size].to(device))
                    treatment = Variable(
                        X_train_batch[:, 2 * input_size].to(device).reshape(X_train_batch[:, 2 * input_size].shape[0],
                                                                            1))
                    labels = Variable(y_train_batch.to(device))
                    labels_propensity = labels[:, 0]
                if stopping_flag == 1:
                    break
                # Train the model
                self.train()
                optimizer.zero_grad()

                if model_type == 'TO':
                    # get output from the model, given the inputs
                    outputs_prop, p1, p0 = self(inputs_0.float(), inputs_1.float(), inputs_2.float())

                    loss_uplift = criterion_uplift(p1 - p0,
                                                   labels_uplift.float().reshape((labels_uplift.shape[0], 1)))
                elif model_type == 'IE':
                    # get output from the model, given the inputs
                    outputs_prop, p1, p0 = self(inputs_0.float(), inputs_1.float(), treatment.float())

                    loss_uplift = criterion_uplift(p1, p0, treatment.float(),
                                                   labels_propensity.float().reshape((labels_propensity.shape[0], 1)))
                # criterion_new
                # full_loss = uplift_loss (p1, p0, treatment.float(),labels_propensity.float().reshape((labels_propensity.shape[0], 1)))
                full_loss = criterion_new(p1, p0, treatment.float(),
                                          labels_propensity.float().reshape((labels_propensity.shape[0], 1)))

                # get loss for the predicted output
                loss_propensity = criterion(outputs_prop,
                                            labels_propensity.float().reshape((labels_propensity.shape[0], 1)))

                epoch_total_batch += 1
                epoch_train_loss_propensity += loss_propensity.item()
                epoch_train_loss_uplift += loss_uplift.item()
                l2_reg = 0
                l1_reg = 0
                if prune:
                    group_lasso_reg = (self.scaling_a - self.scaling_b).abs().sum()
                for m in self.modules():
                    if isinstance(m, LinearProximal):
                        if reg_type == 2:
                            l1_reg += (m.weight_u - m.weight_v).square().sum()
                        else:
                            l1_reg += (m.weight_u - m.weight_v).abs().sum()
                    elif isinstance(m, Linear):
                        if reg_type == 2:
                            l1_reg += m.weight.square().sum()
                        else:
                            l1_reg += m.weight.abs().sum()

                if prune:
                    if model_type == 'TO':
                        loss = (
                                       1 - reg_constant) * loss_propensity + reg_constant * loss_uplift + l_reg_constant * l1_reg + gpl_reg_constant * group_lasso_reg
                    elif model_type == 'IE':
                        loss = full_loss + l_reg_constant * l1_reg + gpl_reg_constant * group_lasso_reg
                else:
                    if model_type == 'TO':
                        loss = (
                                       1 - reg_constant) * loss_propensity + reg_constant * loss_uplift + l_reg_constant * l1_reg
                    elif model_type == 'IE':
                        loss = full_loss + l_reg_constant * l1_reg

                loss.backward()

                # update parameters
                optimizer.step()

                # prune nodes if norm of the vector is smaller than a threshold (for group lasso part)
                if prune:
                    effective_nb_nodes = (self.scaling_a - self.scaling_b).norm(0)

                if prune:
                    if batch_idx % log_interval == 0:
                        if self.verbose:
                            print(
                                f"epoch = {epoch}, batch_idx = {batch_idx}, loss = {loss.item()},loss_propensity = {loss_propensity.item()},loss_uplift  == {loss_uplift.item()}, nb_nodes =={effective_nb_nodes.item()}")
                else:
                    if batch_idx % log_interval == 0:
                        if self.verbose:
                            print(
                                f"epoch = {epoch}, batch_idx = {batch_idx}, loss = {loss.item()},loss_propensity = {loss_propensity.item()},loss_uplift  == {loss_uplift.item()}")

                            # Printing the progress of training in the terminal.
            update_progress(epoch / epochs)

            if self.verbose:
                print(f"--------------- End epoch {epoch} ---------------")

            if prune:
                nb_non_zero_neurons = effective_nb_nodes.item()

            # Inference part to get the performance metrics
            if self.verbose:
                print(f"--------------- Training Performance {epoch} ------------------")

            train_up, train_resp, train_treat, train_err_prop, train_err_up, train_full_loss = test_data(self,
                                                                                                         train_dataloader,
                                                                                                         input_size,
                                                                                                         criterion,
                                                                                                         criterion_uplift,
                                                                                                         criterion_new,
                                                                                                         prune)
            test_up, test_resp, test_treat, test_err_prop, test_err_up, test_full_loss = test_data(self,
                                                                                                   test_dataloader,
                                                                                                   input_size,
                                                                                                   criterion,
                                                                                                   criterion_uplift,
                                                                                                   criterion_new, prune)
            # qini only for the best epoch.
            _, train_qini_res = qini_curve(train_treat, train_resp, train_up, 1, name="qini_curve.pdf", plotit=False)
            _, train_tau_res = qini_barplot(train_treat, train_resp, train_up, 1, plotit=False)

            _, test_qini_res = qini_curve(test_treat, test_resp, test_up, 1, name="qini_curve_val.pdf", plotit=False)
            _, test_tau_res = qini_barplot(test_treat, test_resp, test_up, 1, plotit=False)

            # Log the training and validation loss & uplift
            self.logger.log(train_full_loss, train_qini_res, test_full_loss, test_qini_res, epoch, epoch, epoch)
            # self.logger.log(loss.item(),train_qini_res,epoch)

            train_err_list.append(train_full_loss)
            test_err_list.append(test_full_loss)

            train_err_list_check.append(train_full_loss)

            qini_list.append(train_qini_res)

            val_qini_list.append(test_qini_res)

            self.criterion = criterion
            self.criterion_uplift = criterion_uplift

            if self.save_model:
                self.logger.save_model(self, epoch, best=True)

            if self.verbose:
                print(f"_____ Test Performance {epoch} _________ ")

        if self.learningCurves:
            # y_train,y_val,plot_no,title,y_label,x_label="Epochs",color_train='#54B848',color_val='red')
            self.logger.plotter(train_err_list, test_err_list, 1, "Learning Curves", "Uplift Loss")
            # self.logger.plotter(test_err_list, 1, "Learning Curves", "Validation", "Loss", color='red')
            self.logger.plotter(qini_list, val_qini_list, 2, "Qini Curves", "Qini Coefficient")
            # self.logger.plotter(val_qini_list, 2, "Qini Curves", "Validation", "Qini", color='red')

        return train_up, train_resp, train_treat, train_err_prop, train_err_up, train_err_list, qini_list

    def predict(self, X):
        self.eval()
        with torch.no_grad():
            n_sample = X.shape[0]
            ones = np.ones((n_sample, 1))
            zeros = np.zeros((n_sample, 1))
            X_ones = np.concatenate((X, ones), axis=1)
            X_zeros = np.concatenate((X, zeros), axis=1)
            X_ones_t = torch.from_numpy(X_ones)
            X_zeros_t = torch.from_numpy(X_zeros)
            treatment = np.ones(n_sample)
            treatment_t = torch.from_numpy(treatment)

            _, p1, p0 = call(self, X_ones_t.float(), X_zeros_t.float(), treatment_t.float())
            outputs_up = p1 - p0

        return outputs_up

    def generator(self, scenario):
        seed = self.seed
        rho = 0
        theta = 0.5
        drop_cols = ['ts']

        scenario_dict = {4: [1 / 2, 10000, 200], 5: [1, 20000, 100], 6: [1, 20000, 100], 7: [2, 20000, 100],
                         8: [4, 20000, 100]}
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
        T = df["treat"].values.reshape((-1, 1))
        Y = df["outcome"].values.reshape((-1, 1))
        X = df.iloc[:, 2:].values
        return X, T, Y