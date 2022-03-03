<!-- <style>
r { color: Red }
o { color: Orange }
g { color: Green }
</style>
 -->
# <g> Twin Causal Model
![Twin Causal](./images/twin_causal_icon.png)
---------------------------------------------------

<!-- <img src="./images/twin_causal_icon.png" alt="logo" width="200"/>
<img src="./images/twin_causal_icon.png" alt="drawing" style="width:200px;"/> -->
#### _Authors: Belbahri, M.<sup>*</sup>, Sharoff. P <sup>*</sup>, Gandouet, O., 

*Equally contributing authors



### Development Instruction

#### Install the package in editable developer mode
Use the following command provided in the snippet in the directory containing setup.py file.    
This avoids creating a copy in the library directory of the environment and runs locally

```
pip install -e . 
(or) pip install --editable . 
```
Once the package is installed, this can be imported as a normal package

For eg, to import train from core, use the following command for the version of the package less than 0.0.3

```
import core.train #deprecated
```
The above command are changed to

```
import twincausal #recommended
```
from version 0.0.3


To avoid the warning messages while trying to install other dependencies from pip package manager, use the following command

```
pip install --trusted-host=pypi.org --trusted-host=files.pythonhosted.org {name of package}
```

Additionally, include the flag, to tell pip to install the package inside your home directory, rather than the system python library location. In that case, also add the home directory to the sys.path for importing the package modules

```
--user
```

To install additional development dependencies, use the command

```
pip install twincausal[dev]
```

To install additional test dependencies, use the command

```
pip install twincausal[test]
```

To check the successfull instalation and other meta information, use the command

```
pip show twincausal
```

**1. Requirements**

To Setup the Conda environment:
```
conda env create --file twin-causal-model.yml
conda activate twin-causal-model
```

To update installation from the file:

```
conda env update --file twin-causal-model.yml  --prune
```


**2. Generate Synthetic Data**

Ue the script in _generate_and_save_data.py_ in order to change the parameters and create/save a _.csv_ file in local. 
This will allow to quickly run some simulations.


## Example Command Interface

```
>>> from twincausal.model import twin_causal

>>> X = [[0., 0.], [1., 1.]] # features
>>> y = [0, 1] #outcome variable
>>> T = [0,1] #treatment variable

>>> twin_model = twin_causal(input_size,h_layer,negative_slope,nb_neurons,prune)

>>> twin_model.fit(X)


******************** Start epoch 1 **********************
.
.
.

>>> twin_model.predict([[2., 2.], [-1., -2.],[1,0]], )
array([1, 0])

```
## Test

**1.1 Testing Requirements**

```Pytest```


**1.2 Running instructions**

Simply run command,

```pytest``` 

in the test directory from the terminal. It automatically runs all the test modules from the test directory.

**Other information**

Here is the following, parameter information used to generate the test

*[Add table with the parameter information]*

#### NOTE: For now, the non-prunning eff. no. of neurons stats are specifically set to fake values


</br> 
</br> 
</br> 

## Twincausal.model.twin_causal

<!-- ```
class Twincausal.core.train_sk.genmod(hidden_layer_sizes=(100), activation='relu', *, solver='adam', alpha=0.0001, batch_size='auto', learning_rate='constant', learning_rate_init=0.001, power_t=0.5, max_iter=200, shuffle=True, random_state=None, tol=0.0001, verbose=False, warm_start=False, momentum=0.9, nesterovs_momentum=True, early_stopping=False, validation_fraction=0.1, beta_1=0.9, beta_2=0.999, epsilon=1e-08, n_iter_no_change=10, max_fun=15000)

``` -->




```
class twincausal.model.twin_causal(input_size,
hidden_layer = 1,negative_slope=0.05,nb_neurons = 1024,
random_seed, max_iter=200,learningRate=0.1, shuffle=True,batch_size=256, struc_prune=1,prune=True, verbose = False,logs=True)

```
[[Source]](https://code.td.com/projects/ADV_PROJ/repos/twin-causal-model/browse?at=refs%2Fheads%2Frefine)

\
\
Twin Networks for modelling Uplifts.

This model optimizes the uplift log-loss function using SGD or PGD.


|   	        |   	|   
|:---	        |:---	|
|**Parameter:**   	|  Hyperparameters|
| |  	| 
|   	|  **hidden_layer: int, default=(2)** </br> The number of hidden layer in the twin networks architecture.|
| |  <ul> <li>‘(1)’ Twin Network architecture with 1 hidden layer </li> <li>‘(2)’, Smite2, Smite architecture with 2 hidden layers</li> <li>'(0)' Smite 0 - Smite architecture with no hidden layers </li> </ul>	| 
| |   **activation{‘logistic’, 'leaky_relu'}, default=’leaky_relu’**, </br>  </br> Activation function for the hidden layer.  </br> </br> <ul> <li>‘logistic’, the logistic sigmoid function, returns $f(x) = \frac{1}{(1 + \exp(-x))}$.</li>  <li>‘relu’, the rectified linear unit function, returns $f(x) = \max(0, x)$</li> </ul>	|     
| |   **negative_slope: float, default= 0.05**, </br>  </br>Negative Slope 	|
| |   **prune: bool, default= True**, </br>  </br>default= True, Prunning for regularization, reduces the effective number of nodes in the network. Depending upon the value of this value, the following optimizer works </br> </br> <ul> <li>‘True (or) 1’,  PGD.</li> <li>‘False (or) 0’, SGD </br>  	|
| |   **struc_prune: int, default= 1**, </br>  </br>default= 1, Structured Prunning for regularization, adds l1 or l2 norm as a regularizer in the loss function for optimizing the network parameters. </br> </br> <ul> <li>‘1’, represents L1 regularizer. </li> <li>‘2’, represents L2 regularizer. </br>  	|
| |   **l1_reg_constant: int, default= 0**, </br>  </br> Regularizer variable which manages the weight on l1 norm as a regularizer in the loss function for optimizing the network parameters. </br> </br>  Values Supported l1_reg_constant $\in$ [0,1]  	|
| |   **nb_neuron: int, default = 1024**, </br>  </br>default= 0,  Number of Neurons in a layer	|  
| |   **input_size: int** </br>  </br>Input size of the data	|
| |   **max_iter: int, default = 200** </br>  </br>Maximum number of iterations. The solver iterates until convergence (determined by ‘tol’) or this number of iterations. For stochastic solvers (‘sgd’, ‘adam’), note that this determines the number of epochs (how many times each data point will be used), not the number of gradient steps.	|   
| |   **Optimizer{'SGD', 'PGD'}**, </br>  </br>default='PGD’ Activation function for the hidden layer. ‘identity’, no-op activation, useful to implement linear bottleneck, returns f(x) = x </br> </br> <ul> <li>‘SGD’,  refers to stochastic gradient descent.</li> <li>‘PGD’, the hyperbolic tan function, returns $f(x) = \tanh(x)$.</li> <li>‘relu’, the rectified linear unit function, returns $f(x) = \max(0, x)$ </br> </br> Note: The default optimizer ‘PGD’ works pretty well on relatively large datasets (with thousands of training samples or more) in terms of both training time and validation score. For small datasets, however, ‘SGD’ can converge faster. </li> </ul>	|   
| |   **shuffle: bool, default = Ture**, </br>  </br>default= True,  Whether to shuffle samples in each iteration. Only used when solver=’sgd’ or ‘adam’.	|  
| |   **batch_size: int, default=256**, </br>  </br>Size of minibatches for stochastic optimizers. If the solver is ‘lbfgs’, the classifier will not use minibatch.	|
| |   **learningRate: int, default=0.1**, </br>  </br>Learning rate schedule for weight updates. Used alongside the solver SGD and PGD	|
| |   **random_state: int, RandomState instance,**, </br>  </br>Determines random number generation for weights and bias initialization and batch sampling in the solvers. Pass an int for reproducible results across multiple function calls.	|
|**Attributes**| <dl>  <dt>**classes_ndarray or list of ndarray of shape (n_classes,)**</dt> <dd>Class labels for each output</dd> </dl>| 
| |   **activation{‘logistic’, ‘relu’}**, </br>  </br>default=`Leaky relu’ Activation function for the hidden layer.  </br> </br> <ul> <li>‘logistic’, the logistic sigmoid function, returns $f(x) = \frac{1}{(1 + \exp(-x))}$.</li> <li>‘relu’, the rectified linear unit function, returns $f(x) = \max(0, x)$</li> </ul>	|   

<!-- Include the best model later -->
|   	        |   	|   
|:---	        |:---	|
|**Parameter:**   	|  Options|
| |   **save_model: bool, default = False,** </br>  </br>Saves the model for inference, this will enable the flexibility for restoring the model later. The models are saved in .pth file format.|
| |   **plotit: bool, default = False,** </br>  </br>Generates a matplotlib figures for the training and validation error and uplifts at the end of the training.|        
| |   **verbose: bool, default = False,** </br>  </br>Whether to print progress messages to stdout.	|
| |   **logs: bool, default = True,** </br>  </br>Log into tensorboard for monitoring the progress of learning otherwise return a matplotlib object with training curve.	|    



## Methods

|   	        |   	|   
|:---	        |:---	|
|**fit**(X,T,Y)   	|  Fit the model to features X, Treatment T and outcome variable Y. |
|**predict**(X,T)   	|  Predict the outcome using the model fitted. |
|**generator**(scenario)   	|  Generates data based on Powers et.al . |

```
fit(X,T,Y) 
```
[[Source]](https://code.td.com/projects/ADV_PROJ/repos/twin-causal-model/browse?at=refs%2Fheads%2Frefine)

Fit the model to data matrix X, treatment T and target(s) y.



|   	        |   	|   
|:---	        |:---	|
|**Parameters:**   	|  **X**: Xndarray or sparse matrix of shape (n_samples, n_features) </br> The input data. | 
|   	|  **Xndarray or sparse matrix of shape (n_samples, 1)** </br> The input treatment. | 
|   	|  **Xndarray or sparse matrix of shape (n_samples, 1)** </br> The Outcome variable. | 
|   	|  **test_split: float, default=0.3** </br> Data Split for test/train | 
|**Returns:**   	|  None |  | 


```
generator(scenario) 
```
[[Source]](https://code.td.com/projects/ADV_PROJ/repos/twin-causal-model/browse?at=refs%2Fheads%2Frefine)

Generates the data according to different scenarios list

Refer: for more details: Scott Powers, Junyang Qian, Kenneth Jung, Alejandro Schuler, Nigam H Shah, Trevor Hastie, and Robert
Tibshirani. Some methods for heterogeneous treatment effect estimation in high dimensions. Statistics in
Medicine, 37(11):1767–1787, 2018.

|   	        |   	|   
|:---	        |:---	|
|**Parameters:**   	|  **Scenarios** </br> The input scenarios|  
|**Returns:**   	|  **Xndarray or sparse matrix of shape (n_samples, n_features)** </br> X - Features. |  | 
|   	|  **Xndarray or sparse matrix of shape (n_samples, n_features)** </br> T - Treatement Variable. |  | 
|   	|  **Xndarray or sparse matrix of shape (n_samples, n_features)** </br> Y - Outcome Varaiable. |  | 



### **Notes**

Twin Causal library helps to trains a twin causal networks iteratively, where in each step of the training it proceeds to update the model parameters in the direction of the gradients to minimize the uplift loss function.  

To avoid the problem of overfitting, both structured prunning and unstructured prunning can be used to regularize the loss function to shrink the model parameters.

Model supports numpy arrays int or floats for data injestion.   


### **Reference**
**A Twin Neural Model for Uplift**  </br>
$\hspace{1cm}$ Mouloud Belbahri, Olivier Gandouet, Alejandro Murua, Vahid Partovi Nia

### **Citing**
If you use twin-causal in a scientific publication, we would appreciate citations to the following paper:



JOSS Paper: Twin-Causal: Deep Learning based Twin model for Uplift .

```
@article{twin-causal,
 title={Scikit-learn: Machine Learning in {P}ython},
 author={Pedregosa, F. and Varoquaux, G. and Gramfort, A. and Michel, V.
         and Thirion, B. and Grisel, O. and Blondel, M. and Prettenhofer, P.
         and Weiss, R. and Dubourg, V. and Vanderplas, J. and Passos, A. and
         Cournapeau, D. and Brucher, M. and Perrot, M. and Duchesnay, E.},
 journal={Journal of Machine Learning Research},
 volume={12},
 pages={2825--2830},
 year={2011}
}
```
* Note: This is a placeholder bib information, please use the weblink in the meanwhile to cite our software.

&copy; 2021 Twin-Causal Developers (MIT License)



<!-- # Working Area

For data generators
https://scikit-learn.org/stable/datasets/toy_dataset.html

<span style="color:blue">some *blue* text</span>.

"And here's to you, <span style="background-color:green">Testing green</span>, 
 -->

<!-- ### Legends TODOs:

- <r>TODO:</r> Important thing to do
- <o>TODO:</o> Less important thing to do
- <g>DONE:</g> Go 
 -->
