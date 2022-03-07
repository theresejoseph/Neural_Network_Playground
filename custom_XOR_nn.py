import numpy as np
from functools import partial

class Neuron:
    def __init__(self, weights, bias, activation):
        self.weights = weights
        self.bias = bias 
        self.activation = activation
        self.neuron_list=[]
        

    def evaluate(self,inputs):
        return self.activation(np.sum(self.weights * inputs + self.bias))


    def link(self,next_neurons):
       self.neuron_list.extend(next_neurons)
       
relu = lambda x : max(0,x)
linear =lambda x: x
step = lambda val,x: 1 if x > val else 0

#initialise XOR
input1 = Neuron(np.array([1,1]),0,linear)
input2 = Neuron(np.array([1,1]),0,linear)
input1.idx = 0
input2.idx = 1

hidden1=Neuron(np.array([1,1]),0,partial(step, 0.5))
hidden2=Neuron(np.array([-1,-1]),0,partial(step,-1.5))

output=Neuron(np.array([1,1]),0,partial(step,1.5))

#connections 
output.link([hidden1,hidden2])
hidden1.link([input1,input2])
hidden2.link([input1,input2])


def recurse(current_neuron,inputs):
    if len(current_neuron.neuron_list) == 0:
        return inputs[current_neuron.idx]
    
    y=[recurse(i,inputs) for i in current_neuron.neuron_list]
    return current_neuron.evaluate(y)
    
print(recurse(output,np.array([1,0])))

##############################################################################
from pyvis.network import Network

net = Network()

net.add_node(1, label='input1')
net.add_node(2, label='input2')
net.add_node(3, label='hidden1')
net.add_node(4, label='hidden2')
net.add_node(5, label='output')

# weights -1.5 = 1, -1 =1.5, -0.5 = 2, 0 = 2.5, 0.5 = 3, 1 = 3.5, 1.5=4
net.add_edge(1, 3,titile="1")
net.add_edge(1, 4,title="-1")
net.add_edge(2, 3,title="1")
net.add_edge(2, 4,title="-1")
net.add_edge(3, 5,title="1")
net.add_edge(4, 5,title="1")


net.show('nodes.html')