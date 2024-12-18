from DataStructures.queueADT import Queue

class Batch:
    def __init__(self,feature_array,output_array):
        self.feature_array=feature_array
        self.output_array=output_array



def batch_split1(features_array,output_array,batch_size):
    batch_queue=Queue()
    batches_features=[features_array[i:i+batch_size] for i in range(0,len(features_array),batch_size)]
    batches_output=[output_array[i:i+batch_size] for i in range(0,len(output_array),batch_size)]
    for bf,bo in zip(batches_features,batches_output):
        batch=Batch(bf,bo)
        batch_queue.enqueue(batch)
    return batch_queue

