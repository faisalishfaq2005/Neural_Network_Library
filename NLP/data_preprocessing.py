import re
def preprocess_text(text):
    
    stop_words = set([
        'the', 'and', 'to', 'of', 'a', 'in', 'that', 'it', 'is', 'for', 'with', 'on', 'as', 'at', 'this', 'by', 'an'
    ])
    
    
    text = re.sub(r'[^\w\s]', '', text.lower())
    
   
    words = text.split()
   
    words = [word for word in words if word not in stop_words]
    
    return words
