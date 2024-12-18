from DataStructures.Graphs import Graph
def build_graph(words):
    graph = Graph()
    
    window_size = 2
    for i in range(len(words) - window_size + 1):
        for j in range(i + 1, i + window_size):
            word1 = words[i]
            word2 = words[j]
            if word1 != word2:
                graph.add_edge(word1, word2, weight=1)
    
    return graph