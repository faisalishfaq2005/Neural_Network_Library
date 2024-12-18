import re
from collections import deque

class Graph:
    def __init__(self):
        self.graph = {}

    def add_edge(self, node1, node2, weight=1):
        if node1 not in self.graph:
            self.graph[node1] = {}
        if node2 not in self.graph:
            self.graph[node2] = {}
        
        self.graph[node1][node2] = weight
        self.graph[node2][node1] = weight

    def neighbors(self, node):
        """Return the neighbors (connected nodes) of a given node."""
        return self.graph.get(node, {})

    def degree(self, node):
        """Return the degree of a node (i.e., the number of its neighbors)."""
        return len(self.graph.get(node, {}))

    def nodes(self):
        """Return the list of all nodes in the graph."""
        return list(self.graph.keys())

    def edges(self):
        """Return the list of all edges in the graph."""
        edges = []
        for node1 in self.graph:
            for node2, weight in self.graph[node1].items():
                if (node2, node1) not in edges:  # Avoid duplicates
                    edges.append((node1, node2, weight))
        return edges

    def bfs(self, start_node):
        """
        Perform BFS starting from start_node to find all connected nodes.
        Returns a list of nodes in the connected component.
        """
        visited = set()
        queue = deque([start_node])
        visited.add(start_node)
        cluster = []

        while queue:
            node = queue.popleft()
            cluster.append(node)
            for neighbor in self.neighbors(node):
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)
        
        return cluster




