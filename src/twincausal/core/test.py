import numpy as np
import torch
from torch.autograd import Variable


def device_agno():
    """
    Returns the available device that can be used for training torch 
    """
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    return device


def test_data(model, data, input_size, criterion, verbose=False):
    device = device_agno()
    model.eval()

    full_loss = 0
    nb_batch = 0
    accumulated_vector_uplift = np.array([])
    accumulated_vector_response = np.array([])
    accumulated_vector_ctrl_treat = np.array([])
    with torch.no_grad():
        for X_batch, y_batch in data:
            inputs_0 = Variable(X_batch[:, :input_size].to(device))
            inputs_1 = Variable(X_batch[:, input_size:2 * input_size].to(device))
            treatment = Variable(X_batch[:, 2 * input_size].to(device).reshape(X_batch[:, 2 * input_size].shape[0], 1))
            labels = Variable(y_batch.to(device))
            labels_propensity = labels[:, 0]
            labels_uplift = labels[:, 1]

            outputs_prop, p1, p0 = model(inputs_0.float(), inputs_1.float(), treatment.float())
            outputs_up = p1 - p0
            buffer_accumulated_vector_uplift = outputs_up.cpu().detach().numpy().reshape(-1)
            buffer_accumulated_vector_response = labels_propensity.cpu().detach().numpy().reshape(-1)
            buffer_accumulated_vector_ctrl_treat = treatment.cpu().detach().numpy().reshape(-1)

            full_loss += criterion(p1, p0, treatment.float(), labels_propensity.float().reshape((labels_propensity.shape[0], 1)))
            nb_batch += 1

            accumulated_vector_uplift = np.concatenate([accumulated_vector_uplift, buffer_accumulated_vector_uplift])
            accumulated_vector_response = np.concatenate([accumulated_vector_response, buffer_accumulated_vector_response])
            accumulated_vector_ctrl_treat = np.concatenate([accumulated_vector_ctrl_treat, buffer_accumulated_vector_ctrl_treat])
            if verbose:
                print(f"Average uplift-loss: {full_loss / nb_batch}")
    return accumulated_vector_uplift, accumulated_vector_response, accumulated_vector_ctrl_treat, full_loss / nb_batch

