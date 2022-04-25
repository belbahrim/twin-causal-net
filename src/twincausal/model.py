### Importing the modules and library

# Importing the modules of the twincausal library
from twincausal.utils.performance import qini_barplot, qini_curve
from twincausal.core.test import device_agno
from twincausal.utils.logger import Logger
from twincausal.utils.preprocess import read_and_split_data, concat_create_dataset
from twincausal.losses.losses import uplift_loss, likelihood_version_loss, logistic_loss
from twincausal.proximal.proximal import PGD, LinearProximal
from twincausal.core.test import test_data

# Importing Pytorch and tensorboard
import torch
import torch.nn.functional as functional
from torch import nn
from torch.utils.data import DataLoader
from torch.nn import Linear
from torch.autograd import Variable
from tensorboardX import SummaryWriter

# Importing general libraries
import numpy as np
import pandas as pd

# Importing sys libraries
import sys
from datetime import datetime


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
            print('%-25s%-12f' % (hyperparameter_name, hyperparameter_value))
        # Printing the bool based hyperameters/ configs used
        if type(hyperparameter_value) == bool:
            print('%-25s%-12s' % (hyperparameter_name, hyperparameter_value))


def update_progress(progress):
    """
    Shows the progress in the terminal
    """
    barLength = 20  # Modify to change the length of the progress bar
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
    def __init__(self, nb_features, nb_hlayers=1, nb_neurons=256, lrelu_slope=0, batch_size=256, shuffle=True,
                 max_iter=100, learningRate=0.005, reg_type=1, l_reg_constant=0.001, prune=True,
                 gpl_reg_constant=0.0001, loss="uplift_loss", learningCurves=True, save_model=False, verbose=False,
                 logs=True, random_state=1234):
        super().__init__()
        self.input_size = nb_features + 1
        self.nb_hlayers = nb_hlayers
        self.nb_neurons = nb_neurons
        self.lrelu_slope = lrelu_slope
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.epochs = max_iter
        self.learningRate = learningRate
        self.reg_type = reg_type
        self.l_reg_constant = l_reg_constant
        self.prune = prune
        self.gpl_reg_constant = gpl_reg_constant
        self.loss = loss
        self.seed = random_state

        self.learningCurves = learningCurves
        self.save_model = save_model
        self.verbose = verbose
        self.logs = logs

        self.active_nb_nodes = nb_neurons
        self.optim_used = "SGD" if not self.prune else "PGD"
        linear_cls = nn.Linear if not self.prune else LinearProximal

        if nb_hlayers == 0:
            self.input_size = 2 * self.input_size - 1
            self.prune = False
            self.nb_neurons = 0
            self.active_nb_nodes = 0
            self.gpl_reg_constant = 0
            if reg_type == 2:
                self.optim_used = "SGD"
                self.fc_output = nn.Linear(self.input_size, 1)
            elif reg_type == 1:
                self.fc_output = linear_cls(self.input_size, 1)

        elif nb_hlayers == 1:
            if self.prune:
                self.scaling_a = nn.Parameter(torch.randn(nb_neurons).abs_())
                self.scaling_b = nn.Parameter(torch.randn(nb_neurons).abs_())

            self.fc_layer = linear_cls(self.input_size, nb_neurons)
            self.fc_output = linear_cls(nb_neurons, 1)

        elif nb_hlayers > 1:
            self.prune = False
            self.nb_neurons_per_layer = nb_neurons
            self.fc_layer1 = linear_cls(self.input_size, self.nb_neurons_per_layer)
            self.fc_layer2 = linear_cls(self.nb_neurons_per_layer, self.nb_neurons_per_layer)
            self.fc_output = linear_cls(self.nb_neurons_per_layer, 1)

        self.logger = Logger("_twincausal", "{}_{}".format(self.nb_hlayers, self.nb_neurons))

    def forward(self, x_treat, x_control, treat):
        if self.nb_hlayers == 0:
            x1 = self.fc_output(x_treat)
            x0 = self.fc_output(x_control)

            m1t = x1 * treat + x0 * (1.0 - treat)
            m1t = torch.sigmoid(m1t)

            m11 = torch.sigmoid(x1)
            m10 = torch.sigmoid(x0)

            return m1t, m11, m10

        elif self.nb_hlayers == 1:
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

        elif self.nb_hlayers == 2:
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

            m1t = x1 * treat + x0 * (1.0 - treat)
            m1t = torch.sigmoid(m1t)

            m11 = torch.sigmoid(x1)
            m10 = torch.sigmoid(x0)

            return m1t, m11, m10
        else:
            x1 = self.fc_layer1(x_treat)
            x1 = functional.leaky_relu(x1, self.lrelu_slope)
            x1 = self.fc_layer2(x1)
            x1 = functional.leaky_relu(x1, self.lrelu_slope)
            for x in range(self.nb_hlayers - 2):
                x1 = self.fc_layer2(x1)
                x1 = functional.leaky_relu(x1, self.lrelu_slope)
            x1 = self.fc_output(x1)

            x0 = self.fc_layer1(x_control)
            x0 = functional.leaky_relu(x0, self.lrelu_slope)
            x0 = self.fc_layer2(x0)
            x0 = functional.leaky_relu(x0, self.lrelu_slope)
            for x in range(self.nb_hlayers - 2):
                x0 = self.fc_layer2(x0)
                x0 = functional.leaky_relu(x0, self.lrelu_slope)
            x0 = self.fc_output(x0)

            m1t = x1 * treat + x0 * (1.0 - treat)
            m1t = torch.sigmoid(m1t)

            m11 = torch.sigmoid(x1)
            m10 = torch.sigmoid(x0)

            return m1t, m11, m10

    def fit(self, X, treat, Y, val_size=0.3):

        if self.nb_hlayers > 0:
            if X.shape[1] != self.input_size - 1:
                raise ValueError('The number of inputs provided is different from the input size of the network')
        else:
            if self.input_size != 2 * X.shape[1] + 1:
                raise ValueError('The number of inputs provided is different from the input size of the network')
        list_variable_instance = list(self.__dict__.items())
        list_variable_instance = list_variable_instance[9:]
        hyperparameter_value(list_variable_instance)

        seed = self.seed
        reg_type = self.reg_type
        device = device_agno()

        con = np.concatenate((Y, treat, X), axis=1)
        df = pd.DataFrame(con)
        outcome_var = "outcome"
        treatment_var = "treat"
        df = df.rename(columns={0: outcome_var, 1: treatment_var})

        X_train_one, X_train_zero, X_train_treatment, y_train, y_up_train, X_test_one, X_test_zero, X_test_treatment, y_test, y_up_test = read_and_split_data(
            data=df,
            test_size=val_size,
            outcome_var=outcome_var,
            treatment_var=treatment_var,
            seed=seed,
            standardize=False)

        if self.nb_hlayers == 0:
            train_dataset = concat_create_dataset(X_train_one, X_train_zero, X_train_treatment, y_train, y_up_train,
                                                  True)
            test_dataset = concat_create_dataset(X_test_one, X_test_zero, X_test_treatment, y_test, y_up_test, True)
        else:
            train_dataset = concat_create_dataset(X_train_one, X_train_zero, X_train_treatment, y_train, y_up_train,
                                                  False)
            test_dataset = concat_create_dataset(X_test_one, X_test_zero, X_test_treatment, y_test, y_up_test, False)

        shuffle = self.shuffle
        batch_size = self.batch_size
        train_dataloader = DataLoader(train_dataset, batch_size, shuffle)
        test_dataloader = DataLoader(test_dataset, batch_size, shuffle)

        torch.manual_seed(seed)
        learningRate = self.learningRate
        prune = self.prune

        if self.loss == "uplift_loss":
            criterion = uplift_loss
        elif self.loss == "likelihood_loss":
            criterion = likelihood_version_loss
        elif self.loss == "logistic_loss":
            criterion = logistic_loss
        else:
            print("custom loss")
            criterion = self.loss

        if self.optim_used == 'PGD':
            optimizer = PGD(self.parameters(), lr=learningRate)
        else:
            optimizer = torch.optim.SGD(self.parameters(), lr=learningRate)

        _, _, _, training_err_list, training_qini_list = self.training1(train_dataloader, test_dataloader,
                                                                        criterion, optimizer,
                                                                        prune, reg_type)

        # return training_err_list, training_qini_list
        return None

    def training1(self, train_dataloader, test_dataloader, criterion, optimizer, prune, reg_type=1):

        model_type = 'IE'
        log_interval = 50
        l_reg_constant = self.l_reg_constant  # regularization on the weights
        gpl_reg_constant = self.gpl_reg_constant  # regularization on the scaling factors
        epochs = self.epochs
        device = device_agno()
        input_size = self.input_size

        # Train the model
        train_err_list = []
        val_err_list = []
        train_qini_list = []
        val_qini_list = []
        stopping_flag = 0
        for epoch in range(epochs):
            if self.verbose:
                print(f"---------- Start epoch {epoch} ----------------")
            epoch_total_batch = 0
            for batch_idx, (X_train_batch, y_train_batch) in enumerate(train_dataloader):
                # Converting inputs and labels to Variable
                if model_type == 'TO':
                    inputs_0 = Variable(X_train_batch[:, :input_size].to(device))
                    inputs_1 = Variable(X_train_batch[:, input_size:2 * input_size].to(device))
                    inputs_2 = Variable(X_train_batch[:, 2 * input_size].to(device).reshape(X_train_batch[:, 2 * input_size].shape[0], 1))
                    labels = Variable(y_train_batch.to(device))
                    labels_propensity = labels[:, 0]
                    labels_uplift = labels[:, 1]
                elif model_type == 'IE':
                    inputs_0 = Variable(X_train_batch[:, :input_size].to(device))
                    inputs_1 = Variable(X_train_batch[:, input_size:2 * input_size].to(device))
                    treatment = Variable(X_train_batch[:, 2 * input_size].to(device).reshape(X_train_batch[:, 2 * input_size].shape[0], 1))
                    labels = Variable(y_train_batch.to(device))
                    labels_propensity = labels[:, 0]
                if stopping_flag == 1:
                    break
                # Train the model
                self.train()
                optimizer.zero_grad()

                outputs_prop, p1, p0 = self(inputs_0.float(), inputs_1.float(), treatment.float())

                full_loss = criterion(p1, p0, treatment.float(),
                                      labels_propensity.float().reshape((labels_propensity.shape[0], 1)))

                epoch_total_batch += 1
                l_reg = 0
                if prune:
                    gpl_reg = (self.scaling_a - self.scaling_b).abs().sum()
                for m in self.modules():
                    if isinstance(m, LinearProximal):
                        if reg_type == 2:
                            l_reg += (m.weight_u - m.weight_v).square().sum()
                        elif reg_type == 1:
                            l_reg += (m.weight_u - m.weight_v).abs().sum()
                    elif isinstance(m, Linear):
                        if reg_type == 2:
                            l_reg += m.weight.square().sum()
                        elif reg_type == 1:
                            l_reg += m.weight.abs().sum()

                if prune:
                    loss = full_loss + l_reg_constant * l_reg + gpl_reg_constant * gpl_reg
                else:
                    loss = full_loss + l_reg_constant * l_reg

                loss.backward()

                # update parameters
                optimizer.step()

                # prune nodes if norm of the vector is smaller than a threshold (for group lasso part)
                if prune:
                    effective_nb_nodes = (self.scaling_a - self.scaling_b).norm(0)
                    if batch_idx % log_interval == 0:
                        if self.verbose:
                            print(f"epoch = {epoch}, batch_idx = {batch_idx}, loss = {loss.item()}, nb_nodes =={effective_nb_nodes.item()}")
                else:
                    if batch_idx % log_interval == 0:
                        if self.verbose:
                            print(f"epoch = {epoch}, batch_idx = {batch_idx}, loss = {loss.item()}")

            # Printing the progress of training in the terminal.
            update_progress(epoch / epochs)

            if prune:
                self.active_nb_nodes = effective_nb_nodes.item()

            # Inference part to get the performance metrics
            train_up, train_resp, train_treat, train_full_loss = test_data(self, train_dataloader, input_size,
                                                                           criterion)
            test_up, test_resp, test_treat, test_full_loss = test_data(self, test_dataloader, input_size, criterion)

            # qini only for the best epoch.
            _, train_qini_res = qini_curve(train_treat, train_resp, train_up, 1, name="", plotit=False)
            _, train_tau_res = qini_barplot(train_treat, train_resp, train_up, 1, plotit=False)

            _, test_qini_res = qini_curve(test_treat, test_resp, test_up, 1, name="", plotit=False)
            _, test_tau_res = qini_barplot(test_treat, test_resp, test_up, 1, plotit=False)

            # Log the training and validation loss & uplift
            self.logger.log(train_full_loss, train_qini_res, test_full_loss, test_qini_res, train_up, test_up, epoch)

            train_err_list.append(train_full_loss)
            val_err_list.append(test_full_loss)

            train_qini_list.append(train_qini_res)
            val_qini_list.append(test_qini_res)

            if self.save_model:
                self.logger.save_model(self, epoch, best=True)
            if self.verbose:
                print(f"--------------- End epoch {epoch} ---------------")

        if self.learningCurves:
            self.logger.plotter(train_err_list, val_err_list, 1, "Learning Curves", "Loss")
            self.logger.plotter(train_qini_list, val_qini_list, 2, "Qini Curves", "Qini Coefficient")

        return train_up, train_resp, train_treat, train_err_list, train_qini_list

    def predict(self, X):
        self.eval()
        with torch.no_grad():
            n_sample = X.shape[0]
            ones = np.ones((n_sample, 1))
            zeros = np.zeros((n_sample, 1))

            if self.nb_hlayers == 0:
                n_features = X.shape[1]
                X_times_zeros = np.zeros((n_sample, n_features))  # Interaction terms
                X_ones = np.concatenate((X, X, ones), axis=1)
                X_zeros = np.concatenate((X_times_zeros, X, zeros), axis=1)
            else:
                X_ones = np.concatenate((X, ones), axis=1)
                X_zeros = np.concatenate((X, zeros), axis=1)

            X_ones_t = torch.from_numpy(X_ones)
            X_zeros_t = torch.from_numpy(X_zeros)
            treatment = np.ones(n_sample)
            treatment_t = torch.from_numpy(treatment)

            _, m11, m10 = call(self, X_ones_t.float(), X_zeros_t.float(), treatment_t.float())
            pred_uplift = m11 - m10

        return pred_uplift