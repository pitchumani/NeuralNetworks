# Building Neural Networks with PyTorch & PyTorch Lightning
In this directory

* [TrainingRegressionModelWithPyTorch.ipynb](TrainingRegressionModelWithPyTorch.ipynb)
  > Notebook to demonstrate the simple regression model using pytorch
* [TrainingRegressionModelWithPyTorch-complex-nn.ipynb](TrainingRegressionModelWithPyTorch-complex-nn.ipynb)
  > Same as the previous simple regression model, but with more layer in the model

* [TrainingRegressionModelUsingPyTorchLightning.ipynb](TrainingRegressionModelUsingPyTorchLightning.ipynb)
  > Same regression model as previous, but built using Pytorch Lightning.

* [ClassificationModel.ipynb](ClassificationModel.ipynb)
  > Implements the classification model using PyTorch Lightning to predict the Bank Customer will Churn or not. The churned data is available in [Churn_Modelling.csv](Churn_Modelling.csv)

## Setup
- install python3
- setup virtual environment
```bash
python -m venv env_name
```
This will create a directory with the environment name specified.
- activate the environment
```bash
source env_name/bin/activate
```
- To work with Jupyter notebook, need package ipykernel
```bash
pip install ipykernel
```
I had to install jupyter as well (pip install jupyter)
-- might be because of using latest python 3.13, same with 3.12.9

- Check python kernels available
```
jupyter kernelspec list
```
- Install virtual env kernel in the jupyter kernels
```bash
python -m ipykernel install --user --name=py13_pytorch_env
```
Now the kernels list will be:
```bash
jupyter kernelspec list
Available kernels:
  py13_pytorch_env    /Users/pitchumani/Library/Jupyter/kernels/py13_pytorch_env
  python3             /Users/pitchumani/code/Ex_Files_AI_Workshop_Neural_Network_PyTorch/py13_pytorch_env/share/jupyter/kernels/python3
  ```
The virtual env that we created is one of the kernels now. We will use this to run jupyter notebooks.

- Run jupyter notebook server
```
$ jupyter notebook
...
    To access the server, open this file in a browser:
        file:///Users/pitchumani/Library/Jupyter/runtime/jpserver-7018-open.html
    Or copy and paste one of these URLs:
        http://localhost:8888/tree?token=c4831688e5faf3c09ddbdb4626bb933250b50e2269b08199
        http://127.0.0.1:8888/tree?token=c4831688e5faf3c09ddbdb4626bb933250b50e2269b08199
...
```

## Training regression model with pytorch

### Loading required packages
(localhost:8888/... link)
In the jupyter notebook, select new -> venv_name kernel we created

- install scikit-learn, torch, torchmetrics, pandas, seaborn using pip
```
!pip install scikit-learn
  ...
```

- import packages
```
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder
```

### Load and explore the regression data from csv file
- read csv using pandas
```
insurance_data = read_csv("filename.csv")
insurance_data.head() # display few of top rows
```

- check the number of records using shape
```
insurance_data.shape
```
```
(1336, 7)
```

- check types of data - to see if they are correct
```
insurance_data.info()
```
```
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 1338 entries, 0 to 1337
Data columns (total 7 columns):
 #   Column    Non-Null Count  Dtype  
---  ------    --------------  -----  
 0   age       1338 non-null   int64  
 1   sex       1338 non-null   object 
 2   bmi       1338 non-null   float64
 3   children  1338 non-null   int64  
 4   smoker    1338 non-null   object 
 5   region    1338 non-null   object 
 6   charges   1338 non-null   float64
dtypes: float64(2), int64(2), object(3)
memory usage: 73.3+ KB
```

- plot a graph for data "charges"
```
sns.histplot(insurance_data["charges"])
```
> histogram for column data "charges"

- boxplot with another data "smoker"
```
sns.boxplot(y = insurance_data["charges"], x = insurance_data["smoker"])
```
> box plot, x axis has yes and no
            y axis has charges
            the boxes of charges in the yes and no column of x.

- scatter plot with another data "age"
```
sns.scatterplot(y = insurance_data["charges"], x = insurance_data["age"])
```

### Split the data into training and validation sets

> With this info we have some understanding of data. Now we have to split the
available data for training and validation. e.g. 80% for training and 20% for validation.

```
x = insurance_data.drop(columns = ["charges"])
y = insurance_data["charges"]
x_train, x_val, y_train, y_val = train_test_split(x, y, test_size = 0.2, random_state = 123)

x_train.shape, x_val.shape, y_train.shape, y_val.shape
```
```
((1070, 6), (268, 6), (1070,), (268,))
```
- 1070 records for training
- 268 records for validation

### Preprocess the data for training
The neural networks or any machine learning models understands only the numerical values.
We can't feed other type of input (string etc). So, it is necessary to preprocess our
data so that it can be fed into models. The numerical encoding of data can be done using
several methods. One Hot encoding is one such method.

In the insurance data, three string fields are there. These
categorical columns need to be transformed using OneHotEncoder.
```
categorical_features = ["sex", "smoker", "region"]
categorical_transformer = OneHotEncoder(
  handle_unknown = "ignore", drop = "first", sparse_output = False
)

preprocessor = ColumnTransformer(
    transformers = [("cat_tr", categorical_transformer, categorical_features)]
    remainder = "passthrough"
)
```
> numerical columns passthrouh, no transformation

Transformer is instanciated. Now preprocess the data.

```
x_train = preprocessor.fit_transform(X_train)
x_val = preprocessor.transform(x_val)

x_train.shape, x_val.shape
```
> Now the number of features become 8 (from 6 earlier)

Print training data, which is NumPy array
print(x_train)
[[1. 0. 1. ... 40. 26.315 1. ]
...
]

To understand the columns better, create temporary dataframe
and check.
```
pd.DataFrame(x_train, columns = preprocessor.get_feature_names_out()).T 
```
```
...
```
Convert into numpy format
```
y_train = y_train.to_numpy()
y_val = y_val.to_numpy()
y_train[:10]
```
if they are already in numpy format, will get attribute error for calling to_numpy()

preprocessing is not done yet
standarize the scale for numberical values
```
stdscaler = StandardScaler()
x_train = stdscaler.fit_transform(x_train)
x_val = stdscaler.transform(x_val)
print(x_train)
```
Standard scaling converts numerical values to z scores, that is expressing every feature in terms of number of standard deviation from the mean.
Standard scaling ensures the input data do not lie in wildly different range. It preserves the information and ensures the values centered around 0.
> training data (x_train) is transformed, the calculated mean and std-deviation is used for validation data (x_val).

Reshape the y to multi dimensional array required by models.
It has just one column, that is represented in multi dimensional array.
```
y_train.reshape(-1, 1)
```
The insurance charges (feature that is to be predicted by model) is from 0 to 50K or 60K. To make model more robust, the values should be small, to scale value in range 0 to 1, lets convert the charges column to a min max scaler.
```
min_max_scaler = MinMaxScaler()
y_train = min_max_scaler.fit_transform(y_train.reshape(-1,1))
y_val = min_max_scaler.transform(y_val.reshape(-1,1))
```

Now we have our input features and labels to train our neural network. They are in numpy format. Now we need to convert them into torch tensors.
```
train_inputs = torch.from_numpy(x_train).float()
train_targets = torch.from_numpy(y_train.reshape(-1,1)).float()

train_inputs.shape, train_targets.shape
```
```
(torch.Size([1070, 8]), torch.Size([1070, 1]))
```

### Creating simple neural network
Simple neural network, one neuron and no activation function.
```
class SimpleNeuralNet(nn.Module):
    # initialize the layers
    def __init__(self, num_features):
        super().__init__()
        self.linear1 = nn.Linear(num_features, 1)
        
    # perform the computation
    def forward(self, x):
        x = self.linear1(x)
        return x
```
> linear1 is the only layer with one neuron

> override forward function. It takes in the record as input argument that is a training data, invokes the linear layer on the training data. Transformed data is returned.

> nn.Module gives other functionalities, such as moving all parameters of the neural network to cpu or gpu device so that it can be trained on that device.

Now instantiate the neural network.
```
model = SimpleNeuralNet(num_features=8)

print(model)
```
Before a neural network is trained, weights and biases are initilized.
Lets print the values in each layer (they were initialized by randomn values now).
```
for layer in model.children():
    if isinstance(layer, nn.Linear):
        print(layer.state_dict()["weight"])
        print(layer.state_dict()["bias"])
```
Got 8 weights for our 8 input features, one for bias.
```
tensor([[-0.3421, -0.1348, -0.3250, -0.2178, -0.1552,  0.0056,  0.0236, -0.1156]])
tensor([0.2285])
```
The weights and biases are model parameters that are found during the training process of the model. These weights and biases will converge to some values, allowing the model to make predictions.

Count the number of parameters that a model is going to train.
```
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

count_parameters(model)
```
```
9
```
> p.requires_grad is checked to count only the trainable parameters. 8 weights and one bias for the example.

Training involves a loss function. This loss determines how good the neural network/model is at any point in time.
loss - how far way the predictions of the neural network are from the actual target values.
For this regression model, loss function shoud be the mean squared, it is available as mse_loss
```
import torch.nn.functional as F

loss_fn = F.mse_loss
```
```
loss = loss_fn(model(train_inputs), train_targets)

print(loss)
```
```
tensor(0.3137, grad_fn=<MseLossBackward0>)
```
> this loss is from the **untrained model**. This is not meaningful now.

lets check the predictions from untrained model.
```
preds = model(train_inputs)

preds
```
```
tensor([[-0.3902],
        [ 0.3602],
        [ 0.0893],
        ...,
        [ 0.6737],
        [ 0.4313],
        [-0.1858]], grad_fn=<AddmmBackward0>)
```
these are random predictions with random values.

When a model has been trained it can be evaluated by R square score and by computing the mean squared error.
```
from torchmetrics.regression import R2Score
from torchmetrics.regression import MeanSquaredError

MSE = MeanSquaredError()
r2score = R2Score()

print("Mean Squared Error: ", MSE(preds, train_targets).item())
print("R^2 : ", r2score(preds, train_targets).item())
```
```
Mean Squared Error:  0.3136635422706604
R^2 :  -7.178620338439941
```

### Setting up dataset and dataloader
In order to feed our training data into our pytorch neural network in batches, we are using a tensor dataset and tensor dataloader. They are provided by torch.utils.data module.
```
from torch.utils.data import TensorDataset, DataLoader
train_ds = TensorDataset(train_inputs, train_targets)
train_ds[:5]
```
```
(tensor([[ 0.9888, -0.5000,  1.7213, -0.6213, -0.5478,  0.0621, -0.7196, -0.0674],
         [-1.0113,  2.0000, -0.5809,  1.6095, -0.5478, -0.1506,  1.2870, -0.8865],
         [ 0.9888, -0.5000, -0.5809, -0.6213,  1.8254,  0.7709, -0.6722, -0.8865],
         [ 0.9888, -0.5000, -0.5809, -0.6213, -0.5478,  0.4874, -0.9722,  1.5709],
         [-1.0113, -0.5000, -0.5809, -0.6213,  1.8254, -1.2847, -2.2011, -0.0674]]),
 tensor([[0.0857],
         [0.6393],
         [0.1191],
         [0.1363],
         [0.0238]]))
```
> we are using builtin dataset, it is possible to define custom dataset by deriving from TensorDataset base class.

Data loading in batches
```
batch_size = 8
# shuffle - flag to shuffle the data from input instead of any predictable manner
train_dloader = DataLoader(train_ds, batch_size, shuffle = True)
# data loader is iterable
# print the next of of first batch
next(iter(train_dloader))
```
```
[tensor([[-1.0113, -0.5000, -0.5809, -0.6213,  1.8254,  0.3456, -0.8052, -0.0674],
         [-1.0113, -0.5000, -0.5809, -0.6213, -0.5478,  1.2671,  0.6065,  0.7517],
         [ 0.9888, -0.5000, -0.5809, -0.6213, -0.5478,  1.4797,  1.0485, -0.8865],
         [-1.0113, -0.5000, -0.5809, -0.6213, -0.5478, -0.0797, -1.7773,  0.7517],
         [ 0.9888, -0.5000, -0.5809, -0.6213, -0.5478,  0.4874, -0.9722,  1.5709],
         [ 0.9888, -0.5000, -0.5809, -0.6213, -0.5478,  1.1253,  0.7644, -0.0674],
         [ 0.9888, -0.5000, -0.5809, -0.6213,  1.8254,  0.0621, -0.1239,  0.7517],
         [-1.0113, -0.5000, -0.5809,  1.6095, -0.5478,  1.6924,  1.0311, -0.8865]]),
 tensor([[0.1058],
         [0.1969],
         [0.1890],
         [0.0978],
         [0.1363],
         [0.1671],
         [0.0891],
         [0.2077]])]
```
> It is first batch of training data, 8 rows are there as our batch size is 8.

Lets do the same for validation data.
```python
# first change validation data into tensor format
val_inputs = torch.from_numpy(x_val).float()
val_targets = torch.from_numpy(y_val.reshape(-1, 1)).float()
# instantiate the dataset for val data now
val_ds = TensorDataset(val_inputs, val_targets)

val_ds[:5]
```
```python
# instantiate the data loader
# no need to specify the shuffle = True, the validation data are used
# for evaluating the model, doesn't need shuffling
val_dloader = DataLoader(val_ds, batch_size)
# first batch of records from validation data
next(iter(val_dloader))
```

### Training the neural network
```python
# create a dictionary for recording the training and validation
# loss for each epoch (one pass through the entire training data)
loss_stats = {
    "train": [],
    "val": []
}
# number of epochs
num_epochs = 100
```

Setup device on which we are goiing to run this training.
```python
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")
print("Using %s device" %(device))
```

Instantiate our model created (SimpleNeuralNet).
note that PyTorch doesn't have abstractions for training code.

moving parameters to the device using "to" boiler plate code.
```python
# instantiate the model
model = SimpleNeuralNet(num_features=8).to(device)

print(model)
```
```python
# user torch's optimizer to update the model parameters using
# gradient values. e.g. stochastic gradient descent optimizer (SGD)
# SGD is one of the many optimizers available in pytorch
# Pass model parameters and learning rate (10^-2) to optimizer
# Learning rate determines the step size for how the model parameters
# converge to their optimal values.
# Too large the step size, the model may not converge
# Too small the learning rate/ step size, the model may take too long to converge
# 1e-2 == 0.01 - convenient for this model
optimizer = torch.optim.SGD(model.parameters(), lr = 1e-2)
```
```python
# set up training loop for the model
# most of these boilerplate codes will not be required when
# using pytorch-lightning

# iterate through the number of epochs
for epoch in range(num_epochs):
    # set training loss to 0 for every epoch
    train_epoch_loss = 0
    # set the model to training mode
    # in the training mode, gradients are computed so that the model
    # parameters can be updated using those gradients.
    # gradient - partial derivates of the loss function with respect to individual model parameters
    # these partial derivates are used to determine how model parameters can be tweaked to minimize the loss function
    model.train()

    # iterate through each batch of the training data
    for x_train_batch, y_train_batch in train_dloader:
        optimizer.zero_grad()

        # move data to the same device as the model
        x_train_batch, y_train_batch = \
        x_train_batch.to(device), y_train_batch.to(device)

        # generate predictions and compare the loss - forward pass
        preds = model(x_train_batch)

        train_loss = loss_fn(preds, y_train_batch)

        # perform gradient descent - gradients are computed
        train_loss.backward()
        # the computed gradients are used to tweak the model parameter values
        optimizer.step()

        # add the training loss to the epoch loss
        train_epoch_loss+= train_loss.item()

    # VALIDATION
    with torch.no_grad():

        val_epoch_loss = 0

        model.eval()

        for x_val_batch, y_val_batch in val_dloader:
            x_val_batch, y_val_batch = x_val_batch.to(device), y_val_batch.to(device)

            y_val_pred = model(x_val_batch)

            val_loss = loss_fn(y_val_pred, y_val_batch)
            
            val_epoch_loss += val_loss.item()

        train_loss = train_epoch_loss / len(train_dloader)
        val_loss = val_epoch_loss / len(val_dloader)

        loss_stats["train"].append(train_loss)
        loss_stats["val"].append(val_loss)

        print(f"Epoch {epoch+0.01}: | Train Loss: {train_loss:.3f} | Val Loss: {val_loss:.3f}")
```

### Visualizing the loss and evaluating the models
```python
train_val_loss_df = pd.DataFrame.from_dict(loss_stats). \
    reset_index().melt(id_vars = ["index"]). \
    rename(columns = {"index": "epochs"})

train_val_loss_df.head()
```
> only top 5 records of training loss data in dataframe

loss data in seaborn plot
