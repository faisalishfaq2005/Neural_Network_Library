class Queue:
    def __init__(self):
        self.queue=[]
    
    def is_empty(self):
        if len(self.queue)==0:
            return True
        else:
            return False
        

    def enqueue(self,data):
        self.queue.append(data)
    
    def dequeue(self):
        if self.is_empty()==True:
            print("queue is empty cannot dequeue")
        else:
            data=self.queue.pop(0)
            return data
    
