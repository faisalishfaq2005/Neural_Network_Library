import numpy as np
from Layers.updatingparams import UpdatingParams
class DenseLayer:
    def __init__(self,input_neurons,neurons):
        self.input_neurons=input_neurons
        self.neurons=neurons

        
        
        if input_neurons!=None:
            #limit = np.sqrt(6 / (input_neurons + neurons))  # For Xavier
            #self.weights = np.random.uniform(-limit, limit, (self.input_neurons, self.neurons))
            #self.bias = np.zeros((1, self.neurons))
            self.weights=np.random.rand(self.input_neurons,self.neurons)*0.01
            self.bias=np.random.rand(1,self.neurons)*0.01


    def forward(self,input_data):
        self.dense_input_data=input_data
        self.dense_output_data=np.dot(self.dense_input_data,self.weights)+self.bias
        return self.dense_output_data
    
    def backward(self,output_error):
        self.weight_gradient = np.dot(self.dense_input_data.T, output_error)
        self.bias_gradient = np.sum(output_error, axis=0, keepdims=True)
        updating_params=UpdatingParams(self.weight_gradient,self.bias_gradient)
        return np.dot(output_error, self.weights.T),updating_params

        
