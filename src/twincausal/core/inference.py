import numpy as np
import torch
from torch.autograd import Variable
from twincausal.core.training import device_agno

def test(model, data, input_size, criterion, criterion_uplift, model_type='TO'):
    
    """
    :param model:
    :param data:
    :param input_size:
    :param criterion:
    :param criterion_uplift:
    :param model_type:
    :return:
    """
    device = device_agno()
    model.eval()
    loss_propensity = 0
    loss_uplift = 0
    nb_batch = 0
    accumulated_vector_uplift = np.array([])
    accumulated_vector_response = np.array([])
    accumulated_vector_ctrl_treat = np.array([])

    with torch.no_grad():
        for X_batch, y_batch in data:

            if model_type == 'TO':
                # if torch.cuda.is_available():
                inputs_0 = Variable(X_batch[:, :input_size].to(device))
                inputs_1 = Variable(X_batch[:, input_size:2 * input_size].to(device))
                inputs_2 = Variable(X_batch[:, 2 * input_size].to(device).reshape(X_batch[:, 2 * input_size].shape[0], 1))
                labels = Variable(y_batch.to(device))
                labels_propensity = labels[:, 0]
                labels_uplift = labels[:, 1]
                # else:
                #     inputs_0 = Variable(X_batch[:, :input_size])
                #     inputs_1 = Variable(X_batch[:, input_size:2 * input_size])
                #     inputs_2 = Variable(X_batch[:, 2 * input_size].reshape(X_batch[:, 2 * input_size].shape[0], 1))
                #     labels = Variable((y_batch))
                #     labels_propensity = labels[:, 0]
                #     labels_uplift = labels[:, 1]

                outputs_prop, p1, p0 = model(inputs_0.float(), inputs_1.float(), inputs_2.float())
                outputs_up = p1 - p0
                buffer_accumulated_vector_uplift = outputs_up.cpu().numpy().reshape(-1)
                buffer_accumulated_vector_response = labels_propensity.cpu().numpy().reshape(-1)
                buffer_accumulated_vector_ctrl_treat = inputs_2.cpu().numpy().reshape(-1)

                loss_propensity += criterion(outputs_prop,
                                             labels_propensity.float().reshape((labels_propensity.shape[0], 1)))
                loss_uplift += criterion_uplift(outputs_up, labels_uplift.float().reshape((labels_uplift.shape[0], 1)))

            elif model_type == 'IE':
                if torch.cuda.is_available():
                    inputs_0 = Variable(X_batch[:, :input_size].cuda())
                    inputs_1 = Variable(X_batch[:, input_size:2 * input_size].cuda())
                    treatment = Variable(
                        X_batch[:, 2 * input_size].cuda().reshape(X_batch[:, 2 * input_size].shape[0], 1))
                    labels = Variable(y_batch.cuda())
                    labels_propensity = labels[:, 0]
                    labels_uplift = labels[:, 1]
                else:
                    inputs_0 = Variable(X_batch[:, :input_size])
                    inputs_1 = Variable(X_batch[:, input_size:2 * input_size])
                    treatment = Variable(X_batch[:, 2 * input_size].reshape(X_batch[:, 2 * input_size].shape[0], 1))
                    # labels = Variable(torch.from_numpy(y_batch))
                    labels = Variable(torch.tensor(y_batch).float())
                    labels_propensity = labels[:, 0]
                    labels_uplift = labels[:, 1]

                outputs_prop, p1, p0 = model(inputs_0.float(), inputs_1.float(), treatment.float())
                outputs_up = p1 - p0
                buffer_accumulated_vector_uplift = outputs_up.cpu().numpy().reshape(-1)
                buffer_accumulated_vector_response = labels_propensity.cpu().numpy().reshape(-1)
                buffer_accumulated_vector_ctrl_treat = treatment.cpu().numpy().reshape(-1)

                loss_propensity += criterion(outputs_prop,
                                             labels_propensity.float().reshape((labels_propensity.shape[0], 1)))
                loss_uplift += criterion_uplift(p1, p0, treatment.float(),
                                                labels_propensity.float().reshape((labels_propensity.shape[0], 1)))

            nb_batch += 1

            accumulated_vector_uplift = np.concatenate(
                [accumulated_vector_uplift, buffer_accumulated_vector_uplift])
            accumulated_vector_response = np.concatenate(
                [accumulated_vector_response, buffer_accumulated_vector_response])
            accumulated_vector_ctrl_treat = np.concatenate(
                [accumulated_vector_ctrl_treat, buffer_accumulated_vector_ctrl_treat])

        print(f"Average loss propensity: {loss_propensity / nb_batch}")
        print(f"Average loss uplift: {loss_uplift / nb_batch}")
    return accumulated_vector_uplift, accumulated_vector_response, accumulated_vector_ctrl_treat, loss_propensity / nb_batch, loss_uplift / nb_batch


def testMLP(model, data, input_size, criterion, criterion_uplift, model_type='TO'):
    """
    :param model:
    :param data:
    :param input_size:
    :param criterion:
    :param criterion_uplift:
    :param model_type:
    :return:
    """
    device = device_agno()
    model.eval()
    loss_propensity = 0
    loss_uplift = 0
    nb_batch = 0
    accumulated_vector_uplift = np.array([])
    accumulated_vector_response = np.array([])
    accumulated_vector_ctrl_treat = np.array([])

    with torch.no_grad():
        for X_batch, y_batch in data:

            # if torch.cuda.is_available():
            inputs_true = Variable(X_batch[:, :input_size].to(device))
            inputs_one = Variable(X_batch[:, input_size:2 * input_size].to(device))
            inputs_zero = Variable(X_batch[:, 2 * input_size:3 * input_size].to(device))
            inputs_treat_var = Variable(
                X_batch[:, 3 * input_size].to(device).reshape(X_batch[:, 3 * input_size].shape[0], 1))
            labels = Variable(y_batch.to(device))
            labels_propensity = labels[:, 0]
            labels_uplift = labels[:, 1]
            # else:
            #     inputs_true = Variable(X_batch[:, :input_size])
            #     inputs_one = Variable(X_batch[:, input_size:2 * input_size])
            #     inputs_zero = Variable(X_batch[:, 2 * input_size:3 * input_size])
            #     inputs_treat_var = Variable(
            #         X_batch[:, 3 * input_size].cuda().reshape(X_batch[:, 3 * input_size].shape[0], 1))
            #     labels = Variable(torch.from_numpy(y_batch))
            #     labels_propensity = labels[:, 0]
            #     labels_uplift = labels[:, 1]

            outputs_prop = model(inputs_true.float())
            p1 = model(inputs_one.float())
            p0 = model(inputs_zero.float())
            outputs_up = p1 - p0
            buffer_accumulated_vector_uplift = outputs_up.cpu().numpy().reshape(-1)
            buffer_accumulated_vector_response = labels_propensity.cpu().numpy().reshape(-1)
            buffer_accumulated_vector_ctrl_treat = inputs_treat_var.cpu().numpy().reshape(-1)

            loss_propensity += criterion(outputs_prop,
                                             labels_propensity.float().reshape((labels_propensity.shape[0], 1)))
            loss_uplift += criterion_uplift(p1, p0, inputs_treat_var.float(),
                                                labels_propensity.float().reshape((labels_propensity.shape[0], 1)))

            nb_batch += 1

            accumulated_vector_uplift = np.concatenate(
                [accumulated_vector_uplift, buffer_accumulated_vector_uplift])
            accumulated_vector_response = np.concatenate(
                [accumulated_vector_response, buffer_accumulated_vector_response])
            accumulated_vector_ctrl_treat = np.concatenate(
                [accumulated_vector_ctrl_treat, buffer_accumulated_vector_ctrl_treat])

        print(f"Average loss propensity: {loss_propensity / nb_batch}")
        print(f"Average loss uplift: {loss_uplift / nb_batch}")
    return accumulated_vector_uplift, accumulated_vector_response, accumulated_vector_ctrl_treat, loss_propensity / nb_batch, loss_uplift / nb_batch
