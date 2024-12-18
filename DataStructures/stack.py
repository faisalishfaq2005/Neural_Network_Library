class Stack:
    def __init__(self):
        self.stack=[]
    
    def push(self,data):
        self.stack.append(data)

    def is_empty(self):
        if len(self.stack)==0:
            return True
        else:
            return False
        
    def display_all(self):
        for ele in self.stack:
            print(f"ele in stack {ele.Weights_gradients}   {ele.Bias_gradients}" )
    def pop(self):
        if self.is_empty():
            print("stack empty cannot pop")
        else:
            data=self.stack.pop()
            return data
    
    def peek(self):
        data=self.stack[-1]
        return data
    
