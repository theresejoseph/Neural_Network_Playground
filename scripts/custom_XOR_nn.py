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

# Expected: 1,1 = 0; 0,0= 0; 1,0 = 1, 0,1=1  
if __name__ == '__main__':
    assert recurse(output,np.array([1,0])) == 1
    assert recurse(output,np.array([0,1])) == 1
    assert recurse(output,np.array([1,1]))== 0
    assert recurse(output,np.array([0,0])) == 0
    print("Tests passed")   


##############################################################################
# from pyvis.network import Network

# net = Network()

# net.add_node(1, label='input1')
# net.add_node(2, label='input2')
# net.add_node(3, label='0.5')
# net.add_node(4, label='-0.5')
# net.add_node(5, label='1')

# # weights -1.5 = 1, -1 =1.5, -0.5 = 2, 0 = 2.5, 0.5 = 3, 1 = 3.5, 1.5=4
# net.add_edge(1, 3,titile="1")
# net.add_edge(1, 4,title="-1")
# net.add_edge(2, 3,title="1")
# net.add_edge(2, 4,title="-1")
# net.add_edge(3, 5,title="1")
# net.add_edge(4, 5,title="1")


# net.show('nodes.html')

import graphviz as G

PENWIDTH = '15'
FONT = 'Hilda 10'
DENSE = True
SPARSE = False


layer_nodes = [2, 2, 1]
connections = [DENSE, DENSE] 
assert len(connections) == (len(layer_nodes) - 1), '"connections" array should be 1 less than the #layers'
for i, type_of_connections in enumerate(connections):
    if type_of_connections == SPARSE:
        assert layer_nodes[i] == layer_nodes[i+1], "If connection type is SPARSE then the number of nodes in the adjacent layers must be equal"

dot = G.Digraph(comment='Neural Network', 
                graph_attr={'nodesep':'0.04', 'ranksep':'0.05', 'bgcolor':'white', 'splines':'line', 'rankdir':'LR', 'fontname':FONT},
                node_attr={'fixedsize':'true', 'label':"", 'style':'filled', 'color':'none', 'fillcolor':'gray', 'shape':'circle', 'penwidth':PENWIDTH, 'width':'0.4', 'height':'0.4'},
                edge_attr={'color':'gray30', 'arrowsize':'.4'})

for layer_no in range(len(layer_nodes)):
    with dot.subgraph(name='cluster_'+str(layer_no)) as c:
        c.attr(color='transparent') # comment this if graph background is needed
        if layer_no == 0:                 # first layer
            c.attr(label='Input')
        elif layer_no == len(layer_nodes)-1:   # last layer
            c.attr(label='Output')
        else:                      # layers in between
            c.attr(label='Hidden')
        for a in range(layer_nodes[layer_no]):
            if layer_no == 0: # or i == len(layers)-1: # first or last layer
                c.node('l'+str(layer_no)+str(a), '', fillcolor='black')#, fontcolor='white'
            if layer_no == len(layer_nodes)-1:
                c.node('l'+str(layer_no)+str(a), '', fontcolor='white', fillcolor='black')#, fontcolor='white'
            else:
                # unicode characters can be used to inside the nodes as follows
                # for a list of unicode characters refer this https://pythonforundergradengineers.com/unicode-characters-in-python.html
                c.node('l'+str(layer_no)+str(a), '\u03C3', fontsize='12') # to place "sigma" inside the nodes of a layer

for layer_no in range(len(layer_nodes)-1):
    for node_no in range(layer_nodes[layer_no]):
        if connections[layer_no] == DENSE:
            for b in range(layer_nodes[layer_no+1]):
                dot.edge('l'+str(layer_no)+str(node_no), 'l'+str(layer_no+1)+str(b),)
        elif connections[layer_no] == SPARSE:
            dot.edge('l'+str(layer_no)+str(node_no), 'l'+str(layer_no+1)+str(node_no))                

dot
dot.format = 'JPEG' # or PDF, SVG, JPEG, PNG, etc. 
dot.render('./example_network')