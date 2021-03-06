---
title: 'Twin Causal: A Python package for twin Uplift neural networks'
tags:
  - Python
  - Machine Learning
  - Causal Inference
  - Deep Learning
authors:
  - name: [Placeholder(ph)]Belbahri, M., # note this makes a footnote saying 'co-first author'
    orcid: 0000-0003-0872-7098
    affiliation: "1, 2" # (Multiple affiliations must be quoted)
  - name: [Ph] Gandouet, O.^[co-first author] # note this makes a footnote saying 'co-first author'
    affiliation: "2"
  - name: [Ph] Sharoff. P^[corresponding author]
    affiliation: "3,2"
affiliations:
 - name: Department of Mathematics and Statistics, University of Montreal
   index: 1
 - name: Analytics & Modeling Advanced Projects, TD Insurance
   index: 2
 - name: [Ph] University of Victoria
   index: 3
date: 15 October 2021
bibliography: paper.bib
# csl: asm.csl
# Optional fields if submitting to a AAS journal too, see this blog post:
# https://blog.joss.theoj.org/2018/12/a-new-collaboration-with-aas-publishing
aas-doi: 10.3847/xxxxx <- update this with the DOI from AAS once you know it.
aas-journal: Astrophysical Journal <- The name of the AAS journal.
---

# Summary

Hindsight is a powerful perception to reflect on any event, this is because it allows us to fully observe both the effect and the causation, the action that caused the effect. Causal Inference has been widely used to understand the concept of hindsight & utilize it in a more formal statistical frameworks. The  most popular and widely used statistical framework for causal inference is the counterfactual framework aka *potential outcome* paradigm developed by [@NeymanJ]

Not all the events behave similarly nor all factors have the same effect, i.e., give the same effect to the cause. So, it is important to understand how different factors contribute to the outcome. The term treatment effect can be viewed as a measure of the causal effect on the  outcome of interest. Though Average Treatment Effect (ATE) gives the effect of a treatment on the outcome for the entire population, oftentimes researchers/ analysts are interested in the Conditional Average Treatment Effect (CATE) or Uplift. Uplift modeling is studied widely in several fields including marketing [@radcliffe1999differential], Hansotia and Rukstales [@hansotia2002incremental], [@radcliffe2007using], clinical trials [@jaskowski2012uplift]
[@lamont2018identification] etc.
The treatment effect models deal with cause-and-effect inference for a specific factor, such as a marketing intervention or a medical treatment. In practice, these models are built on individual data from randomized clinical trials where the goal is to partition the participants into heterogeneous groups depending on the uplift. Most existing approaches are adaptations of random forests for the uplift case. Several split criteria have been proposed in the literature, all relying on maximizing heterogeneity. However, in practice, these approaches are prone to overfitting. In this work, we bring a new vision to uplift modeling. We propose a new loss function defined by leveraging a connection with the Bayesian interpretation of the relative risk. Our solution is developed for a specific twin neural network architecture allowing to jointly optimize the marginal probabilities of success for treated and control individuals. We show that this model is a generalization of the uplift logistic interaction model. We also modify the stochastic gradient descent algorithm to allow for structured sparse solutions. This helps training our uplift models to a great extent. Our proposed method is competitive with the state-of-the-art in simulation setting and on real data from large scale randomized experiments and details of the simulation and the results can be found in [@belbahri2021twin].

![Twin neural network. \label{fig:twin_nn}](./images/twin_architecture.PNG)
<!-- {#fig:description}
{width=10%} -->



 
<!-- In the clinical trials, this framework allows for prediction of an individual patient's response to a medical treatment. These models are intended for randomized experiments, which randomly assigns the participants in treatement and control group with both the treatment and outcome as binary random variable. -->

<!-- 
In the case of parametric modeling approaches, the simplest model that can be used to estimate the uplift
is logistic regression, because the response variable is binary. Thus, the underlying optimization
problem becomes the maximization of the Binomial likelihood. In this case, the approach does not provide a direct uplift search, but rather the probabilities of positive responses for treated and non-treated are modeled separately. Then, their difference is used as an estimate of the uplift. However, the solution is not optimized for searching for heterogeneous groups depending on the uplift. Hence, maximizing the likelihood is not necessarily the right way to estimate the uplift. Therefore, changes are required in the optimization problem in order to appropriately estimate the uplift.  -->

<!-- There are several ways to model and estimate the uplift. Of which, the logistic regression is the simplest parametric modeling approach for binary response variable. The underlying optimzation problem becomes the maximization of Binomial likelihood. This approach does not provide a direct uplift search, but it models the probabilities (because of logit) of positive response for treated and non-treated separately and the difference between this probabilities gives the estimate of the uplift. However the solution is not optimized for searching for heterogenous groups depending on the uplift so maximizing the likelihood is not necessarily the right way to estimate the uplift. Therefore changes has to be made in the optimization problem in order to appropriately estimate the uplift. -->

There are several ways to model and estimate the uplift. Of which, the logistic regression is the simplest parametric modeling approach for binary response variable which has an underlying optimzation problem of maximizing Binomial likelihood. But, it only provides the probilities of positive response for the treat and non-treated separately and their difference is used to estimate the uplift. However, the solution is not optimized for searching for heterogeneous groups depending on the uplift. Hence, maximizing the likelihood is not necessarily the right way to estimate the uplift. Therefore, changes are required in the optimization problem in order to appropriately estimate the uplift.

<!-- This approach does not provide a direct uplift search, but it models the probabilities (because of logit) of positive response for treated and non-treated separately and the difference between this probabilities gives the estimate of the uplift. However the solution is not optimized for searching for heterogenous groups depending on the uplift so maximizing the likelihood is not necessarily the right way to estimate the uplift. Therefore changes has to be made in the optimization problem in order to appropriately estimate the uplift.  -->

We introduce a novel loss function called uplift loss function defined by leveraging a connection with the Bayesian interpretation of the relative risk, another treatment effect measure specific to the binary case. Defining an appropriate loss function for uplift also allows to use simple models or any off the shelf method for the related optimization problem, including complex models such as neural networks. When prediction becomes more important than estimation, neural networks become more attractive than classical statistical models. There are several reasons why neural networks are suitable tools for uplift:

1. they are flexible models and easy to train
with current GPU hardware 
2. with a single hidden layer modeling covariates interactions is straightforward
[@tsang2018neural]
3. they are guaranteed to approximate a large class of functions [@cybenko1989approximation], [@hornik1991approximation], [@pinkus1999approximation];
4. neural networks perform very well on predictive tasks which is the main objective of the uplift
5. a simple neural network architecture ensures model interpretability.


# Statement of need

We present a new deep learning library to combine the state of the art deep learning technique to estimate. `TwinCausal` library was designed to be used by both researchers and industry practitioners on uplift modelling. It is a deep learning based Causal Inference Python package for estimating the uplift/ conditional average treatment effect. It is written in the Pytorch deep learning framework [@NEURIPS2019_9015]. 
Since, it is based on deep learning methods, different configuration of hyperparameter provides different models when sufficient data is provided. Additionally, the package is prepacked with different synthetic data generators [@powers2018some] to help understand the working of the twin neural network based uplift modelling.

The gains obtained on the twin networks are undeniable in comparison with some of the existing techniques in the contemporary space as shown in [@belbahri2021twin]. A practitioner may therefore be rquired to learn, test and implement the methodology from scratch, this poses severe friction in the adaptability of this technique. we aim to reduce the friction and maximize spending time on the actual technique with this package.

<!-- + Regularization -->
<!-- + Loss function
+ Synthetic Data -->


The Current packages for uplift modelling causalml [@chen2020causalml] 
focuses on estimating the conditional average treatment effect with deeplearning algorithms like DragonNet but does not support twin interaction neural networks. With this package, we aim to reach a wider audience with a purpose to use them with ease.

To accomplish this task, we have made a consistently API to run the twin causal model as easy as a high-level wrapper but providing all the flexibility of a low-level native language. We believe that this will reduce the complex nature of this problem and benefit a larger audience. 
<!-- 
[M] The first version of this package implements


 -->


Therefore an open-sourced deep learning based causal-inference package is required.


# Technology
**PyTorch** The core technical component of the package is PyTorch, a deep learning framework developed by Facebook. We have implemented the twin causal networks, smite architecture etc., for estimating the uplift from the data. All the deep learning components like computational graph, losses and training were carried out in Pytorch.


**Numpy** The datastructure for data and model parameters (optional). The inputs to fitting a data has to be given in Numpy array format. 

**Sklearn** The package takes advantage of some of the existing modules of sklearn for effiency and to avoid redundancy.









# Mathematics
In this section, we have included two specific contribution from [@belbahri2021twin] that can be used in our package with ease. A New Uplift Loss function to help fit models for uplift prediction and a suitable optimizer Proximal Gradient Descent.


**Uplift Loss**

We defined a new loss function in [@belbahri2021twin] called the Uplift loss to help fit the twin network architecture. The main goal is to regularize the conditional means in order for improved uplift prediction. 
$$
\begin{align*}
l(\cdot) = l_1(\cdot) + l_2(\cdot)
\end{align*}
$$
The first term $l_1(\cdot)$ on the right is the negative log-likelihood loss.

$$
\begin{align*}
l_1(\cdot) = - \frac{1}{n} \sum_{i=1}^{n} \left( y_i \log \{m_{1,t_i}(x_i)  \} + (1-y_i) \log \{ 1- m_{1,t_i}(x_i) \}  \right)  
\end{align*}
$$
where $y$ is the response variable and the probabilities of positive response $m_{1t}(x)$, the prediction, $n$ is the total number of samples.

The second term $l_2(\cdot)$ of our loss function is the BCE loss, which models the posterior propensity scores as a function of the conditional means for positive and negative outcomes.

$$
l_2(t,y |x) = - \frac{1}{n} \sum_{i=1}^{n}  \left(t_i \log \{p_{y_i, 1} (x)    \} + (1 - t_i) \log \{1 - p_{y_i, 1} (x_i)  \}  \right)

$$

where, $t$ is the response variable and $p_{y1}(x)$ is the prediction. 

**PGD**

The proximal gradient descent has two steps which are iterated until convergence. Let $u_j^{(s)}$ and $v_j^{(s)}$ be the optimization parameters at epoch $s$, $\eta$ be the learning rate, $\Delta$ be the gradiant computation and $\lambda \in R$ be the regularization constant. 
For given $u_j^{(0)}, v_j{(0)}$,

1. Gradient step: define intermediate points $\tilde{u}_j^{(q)}$, 
$\tilde{v}_j^{(q)}$
by taking a gradient step such as

$$
\begin{align*}
\tilde{u}_j^{(q)} = {u}_j^{(q)} - \eta \{ \lambda + \Delta l_{i_q}(\theta_j^{(q)})   \} \\

\tilde{v}_j^{(q)} = {v}_j^{(q)} - \eta \{ \lambda + \Delta l_{i_q}(\theta_j^{(q)})   \}
\end{align*}
$$

2. Projection step: evaluate the proximal operator at the intermediate points 
$$
\begin{align*}
\tilde{u}_j^{(q+1)} \leftarrow  RELU (\tilde{u}_j^{(q)}) \\

\tilde{v}_j^{(q+1)} \leftarrow  RELU (\tilde{v}_j^{(q)}) \\

{\theta}_j^{(q+1)} \leftarrow  {u}_j^{(q+1)} - {v}_j^{(q+1)} 

\end{align*}
$$

Exact zero weight updates appear when $u_j < 0$ and $v_j < 0 $ to enable unstructured sparsity.

# Design
The twin causal models are specified and initialized as objects, an interface well adapted by many similar to the sklearn and not by inheritance. We use consistent interface for intializing a model with an object using *twincausal( )* providing all the necessary components of the network and training configurations as parameters. Next, *.fit( )* helps us to fit the neural network using the data provided which as described is given in a Numpy array format. Also, there is an option to specify the split in the data. To use the fitted model and predict the uplift, the method *.predict( )* serves the purpose. Further to understand and to explore around this space of using and developing deep learning model for causal inference, we have also provided a data generating method, generate (). 

## Usage

Examples and documentations for using twincausal are available.


* Example to showcase the use of twincausal [here](https://code.td.com/projects/ADV_PROJ/repos/twin-causal-model/browse/examples/twin_causal_examples.ipynb?at=refs%2Fheads%2Fclean)
* Complete documentation on twincausal modules and functions [here](https://code.td.com/projects/ADV_PROJ/repos/twin-causal-model/browse/Readme.md?at=refs%2Fheads%2Fclean)


In the following we give a simple example using twincausal to estimate the uplift. These code snippets solve a practical usecase of estimating average treatment effect and conditional average treatment effect which as described earlier in the text is used in many real world applications. The following example illustrates data generation, model initialization, training and evaluation methods.

### Model Loading & Data Spliting
Using any Machine Learning technique involves two things, we need a suitable model and enough data to traine a model. So, let us first import both the model. For validate the fit of the model, we need data which is assumed to be drawn from the same distribution so we need to split the available data into test data and train data so, let us also import the data spliting modules. For the sake of illustration, we have used sklearn train_test_split for data splitting and we can use any available module from available libraries or custom function for this purpose. 

```
from twincausal.model import twin_causal
from sklearn.model_selection import train_test_split
```


### Model Initialization & Hyperparameter configuration

Once all the suitable modules are loaded, we can define the hyperparameter configuration and initialize the model. In the following listing, we can see a sample of the hyperparameter configuration that accounts for the model configuration and training process which is provided as parameters while initializing the model. For the details on the full list of hyperparameters supported please refer to the original documentation [here].

```
#hyperparameters configuration

hlayer = 2
prune = True
epochs = 10
batch_size = 256
shuffle = True
learningRate = 0.2

....

```

Once, we have the right hyperparameter configuration, we can define the model by calling the class twin_causal to create an instance. The following listing illustrates initialization of a model. 
```
#initializing the model

model = twin_causal(input_size,hlayer,negative_slope,nb_neurons,seed, epochs,learningRate,l_reg_constant,gpl_reg_constant, ....)

```

### Data Generation

The package also offers a ready to use collection of built-in datasets to study and understand the effectiveness of the model by using a known cause effect function. There are 8 different scenarios similar to [@powers2018some] data generation which allows it to generate data according to the datasize required without actually downloading.

```
scenario = 7
X,T,Y = models.generator(scenario)
```
The complete information on the function behind different scenarios in the generator can be found at [@powers2018some] 


### Fitting Model

Once we have the data, we can split it into train and test data to train the model and validate and fine tune the hyperparameters to avoid overfitting. This step can be performed with standard modules from libraries like sklearn. Once the data is split, the initialized model can be fit with the data to estimate the treatment effect using the following command.

```
models.fit(X_train,T_train,Y_train)    
```

### Estimating the uplift

The fitted model can then be used to estimate the uplift using a method 'predict' with the following command.
```
pred_uplift = models.predict(X_test)
```

Thus with this series of steps we can estimate the uplift eliminating the need to write the custom codes from scratch.
## Availability

The software is available as a pip installable package in PyPi. To install it use the command 

```
pip install twincausal
```
Also, it is available in Github (or Bitbucket) at [www.github.com/twin_causal]
<!-- # Citations -->


<!-- ## Conclusion -->





<!-- Citations to entries in paper.bib should be in
[rMarkdown](http://rmarkdown.rstudio.com/authoring_bibliographies_and_citations.html)
format.

If you want to cite a software repository URL (e.g. something on GitHub without a preferred
citation) then you can do it with the example BibTeX entry below for @fidgit.

For a quick reference, the following citation commands can be used:
- `@author:2001`  ->  "Author et al. (2001)"
- `[@author:2001]` -> "(Author et al., 2001)"
- `[@author1:2001; @author2:2001]` -> "(Author1 et al., 2001; Author2 et al., 2002)" -->

<!-- # Figures


jk The Figure \ref{fig:twin_nn} \@ref(fig:twin_nn))

Figures can be included like this:
![Caption for example figure.\label{fig:example}](figure.png)
and referenced from text using \autoref{fig:example}.

Figure sizes can be customized by adding an optional second parameter:
![Caption for example figure.](figure.png){ width=20% } -->

This version
Just include the existing save model in the class.
Title

<!-- Next version  -->
<!-- ### Grid Search
### early stopping -->


# Acknowledgements
We would like to thank td insurance for their [ph]


<!-- ## include td insurance -->


<!-- Mouloud Belbahri and Alejandro Murua were partially funded by The Natural Sciences and Engineering Research Council of Canada grant 2019-05444. Vahid Partovi Nia was supported by the Natural Sciences and Engineering Research Council of Canada grant 418034-2012. Mouloud Belbahri wants to acknowledge Python implementation discussions with Ghaith Kazma and Eyy??b Sari. Mouloud Belbahri and Olivier Gandouet
thank Julie Hussin for proof reading an early version of the manuscript and for interesting discussions
about potential applications in Bioinformatics. -->
<!-- @belbahri2021twin -->
# References




<!-- ## MD usage instructions
Single dollars ($) are required for inline mathematics e.g. $f(x) = e^{\pi/x}$

Double dollars make self-standing equations:

$$\Theta(x) = \left\{\begin{array}{l}
0\textrm{ if } x < 0\cr
1\textrm{ else}
\end{array}\right.$$

You can also use plain \LaTeX for equations
\begin{equation}\label{eq:fourier}
\hat f(\omega) = \int_{-\infty}^{\infty} f(x) e^{i\omega x} dx
\end{equation}
and refer to \autoref{eq:fourier} from text. -->