from DataStructures.linklist import LinkList
from Layers.ActivationLayer import ActivationLayer
from Layers.DenseLayer import DenseLayer
from Layers.InputLayer import InputLayer
from Layers.ActivationFunctions import sigmoid,sigmoid_derivative,relu,relu_derivative
import numpy as np
from Layers.lossFunctions import binary_cross_entropy

nn=LinkList()
data = np.array([
    [0.1, 1.0, 0.3, 1.2, 0.5, 1.5, 0.4, 1.6],
    [0.2, 0.8, 0.4, 1.0, 0.7, 1.3, 0.3, 1.4]
])

finaldata = np.array([0, 1, 0, 1, 0, 1, 0, 1])

nn.insert_node(DenseLayer(len(data[0]),8))
nn.print_previous_output_size()
nn.insert_node(ActivationLayer(relu,relu_derivative))
nn.print_previous_output_size()
nn.insert_node(DenseLayer(None,8))
nn.print_previous_output_size()
nn.insert_node(ActivationLayer(sigmoid,sigmoid_derivative))
nn.print_previous_output_size()

for i in range(5000):
    predicted_output=nn.forward_propogation(data)
    error=finaldata-predicted_output
    loss=binary_cross_entropy(finaldata,predicted_output)
    print(f"loss at {i} :",loss)
    gradients_stack=nn.backward_propagation(error)
   
    nn.update_parameters(gradients_stack,learning_rate=0.01)
    


"""node=nn.get_all_nodes()
print(node)
for i,n in enumerate(node):
    if isinstance(n,DenseLayer):
        print(f"weights dense layer {i}: ",n.weights)
        print(f"bias dense layer: {i}",n.bias)
        print(f"neurons dense layer: {i}",n.neurons)
        print(f"inputdata dense layer{i}",n.dense_input_data)
        print(f"outputdata dense layer {i}",n.dense_output_data)
    elif isinstance(n,ActivationLayer):
        print(f"input activation layer  {i}",n.act_input_data)
        print(f"output activation layer  {i}",n.act_output_data)

print("predicted_output",predicted_output)
"""
