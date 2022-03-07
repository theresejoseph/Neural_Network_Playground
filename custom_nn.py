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

