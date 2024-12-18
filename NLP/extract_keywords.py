def extract_keywords(graph):
    centrality = {}
    for node in graph.nodes():
        centrality[node] = graph.degree(node)
    
   
    sorted_keywords = sorted(centrality.items(), key=lambda x: x[1], reverse=True)
    
    return sorted_keywords[:10]