import torch
import math
from tensorboardX import SummaryWriter
from datetime import datetime
from twincausal.utils.logger import Logger

def call(model1, x):
    return model1(x)

class Polynomial3(torch.nn.Module):
    def __init__(self):
        """
        In the constructor we instantiate four parameters and assign them as
        member parameters.
        """
        super().__init__()
        self.a = torch.nn.Parameter(torch.randn(()))
        self.b = torch.nn.Parameter(torch.randn(()))
        self.c = torch.nn.Parameter(torch.randn(()))
        self.d = torch.nn.Parameter(torch.randn(()))

        self.log = Logger("wed222","thurs222")
    
    def forward(self, x):
        """
        In the forward function we accept a Tensor of input data and we must return
        a Tensor of output data. We can use Modules defined in the constructor as
        well as arbitrary operators on Tensors.
        """
        return self.a + self.b * x + self.c * x ** 2 + self.d * x ** 3

    def string(self):
        """
        Just like any class in Python, you can also define custom method on PyTorch modules
        """
        return f'y = {self.a.item()} + {self.b.item()} x + {self.c.item()} x^2 + {self.d.item()} x^3'



    def fit(self,x,y):
        self.training1(x,y)

    def training1(self,x,y):
        logdir = "testing_writer1/logs/scalars/" + datetime.now().strftime("%Y%m%d-%H%M%S")
        # writer = SummaryWriter(logdir)
        criterion = torch.nn.MSELoss(reduction='sum')
        optimizer = torch.optim.SGD(self.parameters(), lr=1e-6)
        # log = Logger("monday","tuesday")
        for t in range(30):
            # Forward pass: Compute predicted y by passing x to the model
            y_pred = call(self,x)

            # Compute and print loss
            loss = criterion(y_pred, y)
            if t % 100 == 99:
                print(t, loss.item())
            # writer.add_text('This is loss', 'This is an lstm', 0)
            # writer.add_scalar('loss/train', loss.item(), t)
            # writer.add_scalar('uplift/train', loss.item(), t)
            self.log.lognew(t,t,t)
            # Zero gradients, perform a backward pass, and update the weights.
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    
    def predict(self,x):
        call(self,x)
    
# Create Tensors to hold input and outputs.
x = torch.linspace(-math.pi, math.pi, 2000)
y = torch.sin(x)

# Construct our model by instantiating the class defined above
model = Polynomial3()
model.fit(x,y)
ypred = model.predict(x)
# writer.close()


# tensorboard --logdir="testing_writer1"