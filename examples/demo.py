from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import torch

#Parameters
scenario = 7
prune = True
hlayer = 1
input_size =101
nb_samples = 20000

p = 100 # for scenario 8
nb_neurons, reg_constant = 2*p+1, 0
negative_slope = 0
struc_prune = 1
epochs = 10
batch_size = 256
shuffle = True
learningRate = 0.2
seed = 33
verbose = False
logs = True
l_reg_constant = 0.001  # regularization on the weights
gpl_reg_constant = 0.001 #include both of them
save_model = True
plotit = True
# loss = "likelihood_loss"
# loss = "uplift_loss"

# Custom loss function (used uplift_loss here for demo)
def custom_loss(m11,m10,t,y):
    m11 = torch.clamp(m11, 0.01, 0.99)
    m10 = torch.clamp(m10, 0.01, 0.99)
    m1t = t * m11 + (1-t) * m10
    py1 = y * m11 / (m11 + m10) + (1.0 - y) * (1.0 - m11) / ((1.0 - m11) + (1.0 - m10))
    loss = t*torch.log(py1) + (1-t)*torch.log(1-py1) + y*torch.log(m1t) + (1-y)*torch.log(1-m1t)
    
    return (-1)*torch.mean(loss)

loss = custom_loss

from twincausal.model import twin_causal
import twincausal.utils.data as twindata

#initialize the model
models = twin_causal(input_size,hlayer,negative_slope,nb_neurons,seed, epochs,learningRate,l_reg_constant,gpl_reg_constant,shuffle,save_model,batch_size, struc_prune,loss,prune, verbose,plotit,logs)

#generate the data
X,T,Y = twindata.generator(scenario)

#splitting data in train and test
X_train, X_test, T_train, T_test, Y_train, Y_test = train_test_split(X,T,Y,test_size=0.33, random_state=42)

#fitting model
train_err_epoch, qini_list_epoch = models.fit(X_train,T_train,Y_train)    

pred_uplift = models.predict(X_test)
