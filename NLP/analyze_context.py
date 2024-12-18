def analyze_context(graph, word):
    
    if word not in graph.graph:
        return f"'{word}' not found in the graph."
    
    
    neighbors = graph.neighbors(word)
    context = {neighbor: graph.graph[word][neighbor] for neighbor in neighbors}
    
    return context