# BS6207 Assignment 1
## Question 1
Given a fully connected Neural Network as follows:  
a. Input (x1,x2,…,xd): d-nodes  
b. K-hidden fully connected layers with bias of 2d+1 nodes  
c. Output (predict): 1 node  
d. Use Relu activation function for all layers  

#### 1. Implement this neural network in pytorch
Here we set `k=10, d=10`( 10 hidden layers, each layer has 21 nodes)  
<img src="https://i.imgur.com/9TEHE0K.png" width="500" height="400"><br/>
```python
import torch
import torch.nn as nn
import numpy as np
from torch.autograd import Variable
from torchsummary import summary
# ======================================================
class NN1(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        global z,z_relu
        z = []  #save output for each node
        z_relu = [] #save output after activation function for each node
        #input_layer
        super(NN1, self).__init__()
        self.layer = nn.Sequential()
        self.layer.add_module("h_1", nn.Linear(input_size, hidden_size))
        z.append(self.layer(X))
        self.layer.add_module("h_1_relu",nn.ReLU())
        z_relu.append(self.layer(X))
        
        #hidden_layers
        for i in range(K-1):
             index = str(i+2)
             self.layer.add_module("h_"+index, nn.Linear(hidden_size, hidden_size))
             z.append(self.layer(X))
                
             self.layer.add_module("h_"+index+"_relu"+index,nn.ReLU())
             z_relu.append(self.layer(X))
        #output_layer
        self.layer.add_module("h_last", nn.Linear(hidden_size, num_classes))
        z.append(self.layer(X))
        #self.layer.add_module("h_1ast_relu"+index,nn.ReLU())
        z_relu.append(self.layer(X))
        #print(len(self.layer))
  
    def forward(self, X):
        y = self.layer(X)
        return y    
```
>Here use the `nn.module` class to define the network NN1.
#### 2. Generate the input data (x1,x2,..xd) \in [0,1] drawn from a uniform random distribution
```python
#d input nodes k hidden layer
K = 10
d = 10
batch_size = 1

X = Variable(torch.rand(batch_size,d),requires_grad=True)
```
#### 3. Generate the labels y = (x1*x1+x2*x2+…+xd*xd)/d
```python
y = torch.tensor([[torch.sum(X**2)/d]])
```
#### 4. Implement a loss function L = (predict-y)^2
```python
loss = (y_pred - y)**2
```
#### 5. Use batch size of 1, that means feed data one point at a time into network and compute the loss. Do one time forward propagation with one data point.
```python
#create NN1 object and do one time forward propagation
model = NN1(d,2*d+1,1)
y_pred = model(X)
summary(model,(1,10))
```
The summary of the model:  
<img src="https://i.imgur.com/6qyhhuK.png" width="400" height="400"><br/>
#### 6. Compute the gradients using pytorch autograd:
a. dL/dw, dL/db
```python
loss.item()
loss.backward()
#========save autograd_w and autograd_b=======
auto_w =[]
auto_b =[]
i = 0

for param in model.parameters():
    if (i%2==0):
        #auto_w.append(param.grad)
        auto_w.append(np.round(np.array(param.grad).astype(np.double),5))
    if (i%2==1):
        #auto_b.append(param.grad)
         auto_b.append(np.round(np.array(param.grad).astype(np.double),5))
    i+=1
```
b. Print these values into a text file: torch_autograd.dat
```python
with open('torch_autograd.dat', 'w') as f:
    f.write('autograd_w' + '\n')
    i = 1
    for w in auto_w:
        f.write('weight_' +str(i)+ '\n')
        f.write(str(w)+'\n')
        i+=1
    f.write('autograd_b' + '\n')
    i = 1
    for b in auto_b:
        f.write('bias_' +str(i)+ '\n')
        f.write(str(b)+'\r\n')
        i+=1
```
#### 7.Implement the forward propagation and backpropagation algorithm from scratch, without using pytorch autograd, compute the gradients using your implementation
a. dL/dw, dL/db
#### Implement the forward propagation from scratch:
```python
#manual forward propagation shares the same weight and bias with auto_forward
parameters = list(model.parameters())
#manual forward
def feedforward(X,parameters):
    global z_manual,z_relu_manual
    z_manual=[]
    z_relu_manual=[]
    z_manual.append(torch.from_numpy(np.dot(X.detach().numpy(),(parameters[0].detach().numpy().T))+parameters[1].detach().numpy()))
    z_relu_manual.append(ReLu(z_manual[0]))
    
    for i in range(0,10):
        z_manual.append(torch.from_numpy(np.dot(z_relu_manual[i],(parameters[(i+1)*2].detach().numpy().T))+parameters[(i+1)*2+1].detach().numpy()))
        z_relu_manual.append(ReLu(z_manual[i+1]))
    z_relu_manual[-1]=z_manual[-1]
    return z_relu_manual[-1]
y_pred = feedforward(X,parameters)
```
#### Implement the backward propagation from scratch:
```python
# dL/dv5
d_z = []
d_w = []
grad_y_pred = 2.0 * (y_pred - y) #[1,1]
grad_z_y = grad_y_pred.detach().numpy()#*ReLu_d(y_pred)#dL/dz5
grad_w_y = grad_z_y*z_relu_manual[-2])
d_z.append(grad_z_y)
d_w.append(grad_w_y)
#from the last hidden layer
for i in range (9):#i=1,2,3,4,5.... p:-4,-6,-8
    grad_z_i = np.dot(d_z[i],ReLu_d(z_manual[-i-2])*parameters[-(i+1)*2].detach().numpy())
    d_z.append(grad_z_i)
    grad_w_i = np.dot(grad_z_i.T,z_relu_manual[-3-i])
    d_w.append(grad_w_i)
    
#first layer
grad_z_1 = np.dot(d_z[9],ReLu_d(z_manual[-11])*parameters[-20].detach().numpy())
d_z.append(grad_z_1)
grad_w_1 = np.dot(grad_z_1.T,X.detach().numpy())
d_w.append(grad_w_1)
```
#### ReLu function and the derivative of ReLu:
```python
def ReLu_d(x):
    return np.where(x < 0, 0, 1)
def ReLu(x):
    return np.where(x < 0, 0, x)
```
b. Print these values into a text file: my_autograd.dat
```python
for i in range(len(d_w)):
    d_w[i]= np.round(np.array(d_w[i]).astype(np.double),5)
    d_z[i]= np.round(np.array(d_z[i]).astype(np.double),5)
d_w.reverse()
d_z.reverse()
with open('my_autograd.dat', 'w') as f:
    f.write('my_w' + '\n')
    i = 1
    for w in d_w:
        f.write('weight_' +str(i)+ '\n')
        f.write(str(w)+'\n')
        i+=1
    f.write('my_b' + '\n')
    i = 1
    for b in d_z:
        f.write('bias_' +str(i)+ '\n')
        f.write(str(b)+'\r\n')
        i+=1
```
#### 8. Compare the two files torch_autograd.dat and my_autograd.dat and show that they give the same values up to 5 significant numbers
(1)Check the difference between autograd_weight and my_weight, autograd_bias and my_bias or not:
<img src="https://i.imgur.com/JaDBLja.png" width="400" height="200"><br/> 
(2)Compare the two files:  
<img src="https://i.imgur.com/dHTjkUC.png" width="400" height="400">
<img src="https://i.imgur.com/6iww5v1.png" width="400" height="400"><br/>   
Therefore, most of those values are the same, autograd and my_autograd can get the same results in all.

## Question 2
Run the following code, generate the computational graph, label and explain all nodes (all nodes means not just the leave nodes, all intermediate nodes should be explained):
```python
import torch
import torch.nn as nn
from torchviz import make_dot
# ======================================================
def print_compute_tree(name,node):
 dot = make_dot(node) 
 #print(dot)
 dot.render(name)
# ======================================================
if __name__=='__main__':
 torch.manual_seed(2317)#random seed
#10 input nodes
 x = torch.randn([1,1,10],requires_grad=True)
#conv layer,in_channels=out_channels=1,kernel_size=3
 cn1 = nn.Conv1d(1,1,3,padding=1)
#full-connected layers
 fc1 = nn.Linear(10,10)
 fc2 = nn.Linear(10,1)
 y = torch.sum(x)
#create objects
 c = cn1(x)
 x = torch.flatten(x)+torch.flatten(c)
 x = fc1(x)
 x = fc2(x)
 loss = torch.sum((x-y)*(x-y))
 print_compute_tree('./tree_ex' ,loss)
```
>For different types of tensor variables, the autograd backward functions for calculating gradients would be different:  
torch.sum() ---> grad_fn=<`SumBackward`>  
nn.Conv1d(X) ---> grad_fn=<`SqueezeBackward`>  
torch.flatten() ---> grad_fn=<`ViewBackward`>  
a+b ---> grad_fn=<`AddBackward`>  
a-b ---> grad_fn=<`SubBackward`>  
a*b ---> grad_fn=<`MulBackward`>  
nn.Linear(x) ---> grad_fn=<`AddBackward`>  
Those information would be helpful to understand the computational graph.
### Computational Graph
This is the computational graph for the above scripts, and there are `explanations for nodes in the graph`. 
<img src="https://i.imgur.com/lEmh2p4.jpg" width="800" height="1200"><br/> 
>For other nodes that are not marked in the graph:
'`AccumulateGrad`' represents leaf nodes, also as known as the end point of the computational graph for BP. It accumulates all backward gradient information for the leaf nodes.
'`UnsqueezeBackward`','`SqueezeBackward`' are working with tensor variable's dimension and shape, to add a dimension or remove a dimension for a variable.


