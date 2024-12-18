def cluster_words(graph, num_clusters):
    """
    Improved clustering: Group words into clusters based on connected components using BFS.
    If the number of clusters is greater than the actual number of components,
    clusters are merged. If it's smaller, we split the larger clusters into smaller ones.
    """
    visited = set()  
    clusters = []  

  
    for node in graph.nodes():
        if node not in visited:
            cluster = graph.bfs(node) 
            clusters.append(cluster)
            visited.update(cluster)

    print("Initial clusters:", clusters)

    while len(clusters) > num_clusters:
        print(f"Current clusters: {clusters} (Total: {len(clusters)})")

        smallest_cluster = min(clusters, key=len)
        clusters.remove(smallest_cluster)

        clusters[0].extend(smallest_cluster)

    while len(clusters) < num_clusters:
        print(f"Current clusters: {clusters} (Total: {len(clusters)})")

        largest_cluster = max(clusters, key=len)
        clusters.remove(largest_cluster)

        mid = len(largest_cluster) // 2
        clusters.append(largest_cluster[:mid])
        clusters.append(largest_cluster[mid:])


    print("Final clusters:", clusters)  
    return clusters
