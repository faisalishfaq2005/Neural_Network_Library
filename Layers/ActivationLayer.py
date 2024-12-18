class ActivationLayer:
    def __init__(self,activation_function,activation_derivative):
        self.Activation_Function=activation_function
        self.Activation_Derivative=activation_derivative
        self.act_input_data=None
        self.act_output_data=None

    def forward(self,input_data):
        self.act_input_data=input_data
        self.act_output_data=self.Activation_Function(input_data)
        return self.act_output_data
    
    def backward(self, output_error):
        return output_error * self.Activation_Derivative(self.act_input_data)
    
    
