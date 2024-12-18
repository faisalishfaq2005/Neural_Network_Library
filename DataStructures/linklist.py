from DataStructures.stack import Stack
from Layers.DenseLayer import DenseLayer
from Layers.ActivationLayer import ActivationLayer
from Layers.updatingparams import UpdatingParams
import numpy as np

class Node:
    def __init__(self,data):
        self.data=data
        self.next_pointer=None
        self.previous_pointer=None

class LinkList:
    def __init__(self):
        self.head_pointer=None
        self.previous_output_size = None

    def is_empty(self):
        if self.head_pointer==None:
            return True
        else:
            return False

    def insert_node(self, layer):
        if isinstance(layer, DenseLayer):
            # If it's the first DenseLayer, infer input size from the first input data
            if self.previous_output_size!= None:
               
                layer.input_neurons = self.previous_output_size
                
                layer.weights = np.random.rand(layer.input_neurons, layer.neurons)*0.01
                layer.bias = np.random.rand(1, layer.neurons)*0.01
        
            self.previous_output_size = layer.neurons
       
        
       
        node = Node(layer)
        if self.head_pointer is None:
            self.head_pointer = node
            return
        
      
        current = self.head_pointer
        while current.next_pointer:
            current = current.next_pointer
        
      
        current.next_pointer = node
        node.previous_pointer = current

    def insert_node_without_updates(self,layer):
        node = Node(layer)
        if self.head_pointer is None:
            self.head_pointer = node
            return
        
      
        current = self.head_pointer
        while current.next_pointer:
            current = current.next_pointer
        
      
        current.next_pointer = node
        node.previous_pointer = current

    def print_previous_output_size(self):
        print("previos output size", self.previous_output_size)
    def get_all_nodes(self):
        nodes=[]
        current=self.head_pointer
        while current:
            nodes.append(current.data)
            current=current.next_pointer
        return nodes
    
    def forward_propogation(self,input_data):
        current_layer=self.head_pointer
        forward_data=input_data
        while current_layer.next_pointer:
            forward_data=current_layer.data.forward(forward_data)
            
                
            current_layer=current_layer.next_pointer
        forward_data=current_layer.data.forward(forward_data)
        return forward_data #in the end will return the final output after one epoch so save it in a variable
    
    
    def backward_propagation(self,error):
        stack=Stack()
        backward_data=error
        current_layer=self.head_pointer
        while current_layer.next_pointer is not None:
            current_layer=current_layer.next_pointer
        while current_layer.previous_pointer is not None:
            if isinstance(current_layer.data,DenseLayer):
                backward_data,updating_params=current_layer.data.backward(backward_data)
                
                stack.push(updating_params)
            elif isinstance(current_layer.data,ActivationLayer):
                backward_data=current_layer.data.backward(backward_data)
            
            current_layer=current_layer.previous_pointer

            
        if isinstance(current_layer.data,DenseLayer):
                backward_data,updating_params=current_layer.data.backward(backward_data)
                
                stack.push(updating_params)
        elif isinstance(current_layer.data,ActivationLayer):
            backward_data=current_layer.data.backward(backward_data)
        
        return stack
    

    def update_parameters(self,stack,learning_rate):
        current_layer=self.head_pointer
        while current_layer.next_pointer is not None:
            if isinstance(current_layer.data,DenseLayer):
                updating_params=stack.pop()
                current_layer.data.weights += learning_rate* updating_params.Weights_gradients
                current_layer.data.bias += learning_rate*updating_params.Bias_gradients

            current_layer=current_layer.next_pointer

        if isinstance(current_layer.data,DenseLayer):
            updating_params=stack.pop()
            current_layer.data.weights += learning_rate* updating_params.Weights_gradients
            current_layer.data.bias += learning_rate*updating_params.Bias_gradients

    def print_nodes(self):
        current_layer=self.head_pointer
        while current_layer.next_pointer is not None:
            print(current_layer.data)
            current_layer=current_layer.next_pointer

        




        







