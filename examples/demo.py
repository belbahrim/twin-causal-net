# Import the libraries and generate synthetic uplift data
import torch
import matplotlib.pyplot as plt
import twincausal.utils.data as twindata
from twincausal.model import twin_causal
from sklearn.model_selection import train_test_split
from twincausal.utils.performance import qini_curve, qini_barplot

X, T, Y = twindata.generator(5)  # Generate fake uplift data
X_train, X_test, T_train, T_test, Y_train, Y_test = train_test_split(X, T, Y, test_size=0.3, random_state=1234)


# Custom loss function (used uplift_loss here for demo)
def custom_loss(m11, m10, t, y):  # TODO: Important to keep the same parameters for the custom loss
    m11 = torch.clamp(m11, 0.01, 0.99)
    m10 = torch.clamp(m10, 0.01, 0.99)
    m1t = t * m11 + (1 - t) * m10
    py1 = y * m11 / (m11 + m10) + (1.0 - y) * (1.0 - m11) / ((1.0 - m11) + (1.0 - m10))
    loss = t * torch.log(py1) + (1 - t) * torch.log(1 - py1) + y * torch.log(m1t) + (1 - y) * torch.log(1 - m1t)
    return (-1) * torch.mean(loss)


new_loss = custom_loss  # The custom loss to pass to the twin_causal CLASS
input_size = X.shape[1] + 1  # Number of features + 1 (for treatment indicator)

# Initialize the model
uplift = twin_causal(input_size=input_size, hlayers=1, nb_neurons=256, lrelu_slope=0, batch_size=256,
                     shuffle=True, max_iter=100, learningRate=0.005, reg_type=1, l_reg_constant=0.001,
                     prune=True, gpl_reg_constant=0.0001, loss=new_loss, learningCurves=True,
                     save_model=False, verbose=False, logs=False, random_state=1234)

# Fitting model
_, _ = uplift.fit(X_train, T_train, Y_train)

# Prediction and visualization
pred = uplift.predict(X_test)

_, q = qini_curve(T_test, Y_test, pred, p_precision=1, plotit=True)
print('The Qini coefficient is:', q)

_, tau = qini_barplot(T_test, Y_test, pred, p_precision=1, plotit=True)
print('The Uplift correlation is:', tau)
