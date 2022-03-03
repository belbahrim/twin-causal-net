import os
import errno
import torch
from tensorboardX import SummaryWriter
from csv import writer
import matplotlib.pyplot as plt

class Logger:
    """
    Logs the experiment, training & validation metrics
    and helps visualize the progress of the experiment.
    Also save the model

    param: model_name
    param: data_name
    return: tensorboard logs, saved model 
    """
    def __init__(self, model_name, data_name, suffix = ""):
        self.model_name = model_name
        self.data_name = data_name
        self.list_qini_test = []
        self.list_qini_train = []
        self.comment = '{}_{}'.format(model_name, data_name)
        self.user_comments = ""
        self.data_subdir = '{}/{}'.format(model_name, data_name)
        self.filename_suffix  = suffix
        self.writer = SummaryWriter(comment=self.comment)

    def log(self, loss_uplift, qini_train, val_loss,qini_test, pred_uplift,
            pred_uplift_test, epoch):

        self.list_qini_train.append(qini_train)

        self.writer.add_scalar(
            'Train/{}/Loss_Uplift_Train'.format(self.comment), loss_uplift, epoch)
        self.writer.add_scalar(
            'Train/{}/Adj_Qini_Train'.format(self.comment), qini_train, epoch)
        self.writer.add_scalar(
            'Test/{}/Test_Loss'.format(self.comment), val_loss, epoch)
        self.writer.add_scalar(
            'Test/{}/Adj_Qini_Test'.format(self.comment), qini_test, epoch)
        self.writer.add_histogram(
            '{}/Pred_Uplift_Train'.format(self.comment), pred_uplift, epoch)
        self.writer.add_histogram(
            '{}/Pred_Uplift_Test'.format(self.comment), pred_uplift_test, epoch)    
        
        self.list_qini_test.append(qini_test.item())

    def plotter(self,y_train,y_val,plot_no,title,y_label,x_label="Epochs",color_train='#54B848',color_val='red'):
        plt.figure(plot_no)
        plt.plot(y_train,color=color_train,label='Training')
        plt.plot(y_val,color=color_val,label='Validation')
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.title(title)
        plt.legend()
        plt.show()

    def save_model(self, model, epoch, path ='', best=True):
        if best:
            if self.list_qini_test[epoch] == max(self.list_qini_test):  # TODO: check if it saves the best model
                if path == '':
                    out_dir = './runs/Models/{}'.format(self.data_subdir)
                else:
                    out_dir = path                
                # print("out_dir ", out_dir)
                Logger._make_dir(out_dir)
                torch.save(model.state_dict(), '{}/Model_epoch_{}.pth'.format(out_dir, epoch))
        else:
            if path =="":
                out_dir = './runs/Models/{}'.format(self.data_subdir)
            else:
                out_dir = path
            Logger._make_dir(out_dir)
            torch.save(model.state_dict(), '{}/Model_epoch_{}.pth'.format(out_dir, epoch))

    def is_best(self, epoch):
        return self.list_qini_test[epoch] == max(self.list_qini_test)

    def close(self):
        self.writer.close()

    # Private Functionality

    @staticmethod
    def _make_dir(directory):
        try:
            os.makedirs(directory)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise

    @staticmethod
    def append_list_as_row(file_name, list_of_elem):
        # Open file in append mode
        with open(file_name, 'a+', newline='') as write_obj:
            # Create a writer object from csv module
            csv_writer = writer(write_obj)
            # Add contents of list as last row in the csv file
            csv_writer.writerow(list_of_elem)
