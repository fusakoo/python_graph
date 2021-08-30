# Course: CS261 - Data Structures
# Author: Fusako Obata
# Description:  Implement DirectedGraph class that's designed to support the 
#   following type of graph: directed, weighted (positive edge weights only), 
#   no duplicate edges, no loops. Cycles are allowed. 
#   
#   Directed graphs are stored as a two dimensional matrix (list of lists). 
#   Element on the i-th row and j-th column in the matrix is the weight of the 
#   edge going from the vertex with index i to the vertex with index j.

import heapq
from collections import deque

class DirectedGraph:
    """
    Class to implement directed weighted graph
    - duplicate edges not allowed
    - loops not allowed
    - only positive edge weights
    - vertex names are integers
    """

    def __init__(self, start_edges=None):
        """
        Store graph info as adjacency matrix
        DO NOT CHANGE THIS METHOD IN ANY WAY
        """
        self.v_count = 0
        self.adj_matrix = []

        # populate graph with initial vertices and edges (if provided)
        # before using, implement add_vertex() and add_edge() methods
        if start_edges is not None:
            v_count = 0
            for u, v, _ in start_edges:
                v_count = max(v_count, u, v)
            for _ in range(v_count + 1):
                self.add_vertex()
            for u, v, weight in start_edges:
                self.add_edge(u, v, weight)

    def __str__(self):
        """
        Return content of the graph in human-readable form
        DO NOT CHANGE THIS METHOD IN ANY WAY
        """
        if self.v_count == 0:
            return 'EMPTY GRAPH\n'
        out = '   |'
        out += ' '.join(['{:2}'.format(i) for i in range(self.v_count)]) + '\n'
        out += '-' * (self.v_count * 3 + 3) + '\n'
        for i in range(self.v_count):
            row = self.adj_matrix[i]
            out += '{:2} |'.format(i)
            out += ' '.join(['{:2}'.format(w) for w in row]) + '\n'
        out = f"GRAPH ({self.v_count} vertices):\n{out}"
        return out

    # ------------------------------------------------------------------ #

    def add_vertex(self) -> int:
        """
        Add new vertex to the graph
        """
        self.adj_matrix.append([])
        self.v_count += 1
        for vertex in self.adj_matrix:
            while len(vertex) != self.v_count:
                vertex.append(0)
        return self.v_count

    def add_edge(self, src: int, dst: int, weight=1) -> None:
        """
        Add edge to the graph, connecting the two vertices with the provided indices
        """
        # Following scenarios does nothing
        # Scenario 1) src and dst refer to the same vertex
        if src == dst:
            return
        # Scenario 2) weight is not a positive integer
        elif weight < 0:
            return
        # Scenario 3) Either (or both) vertex indices do not exist in the graph
        elif src not in range(self.v_count) or dst not in range(self.v_count):
            return

        source = self.adj_matrix[src]
        source[dst] = weight


    def remove_edge(self, src: int, dst: int) -> None:
        """
        Removes an edge between the two vertices with provided indices.
        """
        if src == dst:
            return
        elif src not in range(self.v_count) or dst not in range(self.v_count):
            return
        
        source = self.adj_matrix[src]
        if source[dst] != 0:
            source[dst] = 0
        else:
            return


    def get_vertices(self) -> list:
        """
        Returns a list of the vertices of the graph
        """
        if self.v_count == 0:
            return []

        output_list = []
        for index in range(self.v_count):
            output_list.append(index)

        return output_list

    def get_edges(self) -> list:
        """
        Returns a list of edges in the graph. Each edge is returned as a tuple
        of two incident vertex indices and weight.
        -> First element of tuple = source vertex
        -> Second element of tuple = destination vertex
        -> Third element of tuple = weight of edge
        """
        if self.v_count == 0:
            return []

        output_list = []
        for source in range(self.v_count):
            for dest in range(self.v_count):
                if self.adj_matrix[source][dest] != 0:
                    weight = self.adj_matrix[source][dest]
                    output_list.append((source,dest,weight))
        return output_list

    def is_valid_path(self, path: list()) -> bool:
        """
        Takes a list of vertex indices and returns True if the sequence of vertices
        represents a valid path in the graph. An empty path is considered valid.
        """
        # If path (the list of vertex names) is empty, consider valid. Return True
        if path is False:
            return True
        # If path's length is 1
        if len(path) == 1:
            # If the path includes a vertex that is not in the graph, return False
            if path[0] not in range(self.v_count):
                return False
            # Else, return True
            else:
                return True
        # Check whether the sequence of vertices represents a valid path in the graph
        for index in range(len(path)-1):
            src = path[index]
            dst = path[index+1]
            if self.adj_matrix[src][dst] == 0:
                return False
        return True

    def dfs(self, v_start, v_end=None) -> list:
        """
        Return list of vertices visited during DFS search
        Vertices are picked in ascending order
        """
        # If v_start isn't in the graph, return an empty list
        if v_start not in range(self.v_count):
            return []
        # Step 1) Initialize empty set of visited vertices 
        reachable = []
        # Step 2) Initialize stack with v_start
        stack = [v_start]   

        while len(stack) != 0:
            # Step 3) If the queue is not empty, pop a vertex
            vertex = stack.pop() 
            # Step 3.5) If the vertex is v_end, end the loop and return the visited vertices
            if vertex == v_end:
                reachable.append(vertex)
                return reachable
            # Step 4) If vertex is not in the set of visited vertices,
            if vertex not in reachable:
                # -> Add vertex to the set of visited vertices
                reachable.append(vertex)
                successors = []
                for dest in range(self.v_count):
                    weight = self.adj_matrix[vertex][dest]
                    if weight != 0:
                        successors.append(dest)
                # -> Push each vertex that's a direct successor of vertex to stack
                while successors:
                    successor = successors.pop()
                    stack.append(successor)

        return reachable


    def bfs(self, v_start, v_end=None) -> list:
        """
        Return list of vertices visited during BFS search
        Vertices are picked in ascending order
        """
        # If v_start isn't in the graph, return an empty list
        if v_start not in range(self.v_count):
            return []
        # Step 1) Initialize empty set of visited vertices 
        reachable = []
        # Step 2) Initialize queue with v_start
        queue = deque()  # Set a queue (dequeue based)
        queue.append(v_start)

        while len(queue) != 0:
            # Step 3) If the queue is not empty, pop (dequeue) a vertex
            vertex = queue.pop()
            # Step 3.5) If the vertex is v_end, end the loop and return the visited vertices
            if vertex == v_end:
                reachable.append(vertex)
                return reachable
            # Step 4) If vertex is not in the set of visited vertices,
            if vertex not in reachable:
                # -> Add vertex to the set of visited vertices
                reachable.append(vertex)
                successors = deque()
                for dest in range(self.v_count):
                    weight = self.adj_matrix[vertex][dest]
                    if weight != 0:
                        successors.append(dest)
                # -> If each successor is not in the set of visited vertices, enqueue to queue
                while successors:
                    successor = successors.popleft()
                    if successor not in reachable:
                        queue.appendleft(successor)
        
        return reachable

    def _rec_has_cycle(self, vertex, ver_visited, rec_stack):
        """
        Recursive helper function that goes through the graph via a DFS style search
        and marks the visited vertices        
        """
        ver_visited[vertex] = True      # Mark the current vertex as visited (= True)
        rec_stack[vertex] = True        # Same, but will be toggled False if vertex has no destination

        # Check all the destination vertices of the source vertex
        for dest in range(self.v_count):
            weight = self.adj_matrix[vertex][dest]
            # If there is a weight, there is an edge
            if weight != 0:
                # If the vertex hasn't been visited already, check
                if ver_visited[dest] == False:
                    # Destination will be the next vertex to be checked on; see if it returns True
                    # Otherwise, set back the rec_state of vertex to False
                    if self._rec_has_cycle(dest, ver_visited, rec_stack) is True:
                        return True
                # If the destination's index is in the rec_state, return True (there's a cycle)
                elif rec_stack[dest] == True:
                    return True
        # If there's no destination, set state at vertex to False (and back out of the recursion)
        rec_stack[vertex] = False
        return False


    def has_cycle(self):
        """
        Return True if graph contains a cycle, False otherwise
        """
        # Initialize lists to keep track of which vertex we've already visited as well as stack for recursion
        # All values in list will be initialized to  "False" 
        ver_visited = [False] * self.v_count
        rec_stack = [False] * self.v_count

        # Recursively go through the vertices in graph and check if each has been
        # visited via a DFS search (that's done recursively)
        for vertex in range(self.v_count):
            if ver_visited[vertex] == False:
                if self._rec_has_cycle(vertex, ver_visited, rec_stack) is True:
                    return True
        # If there was no cycle detected, it should return False at this point
        return False
        

    def dijkstra(self, src: int) -> list:
        """
        Implements the Dijkstra algorithm to compute the length of the shortest
        path from a given vertex to all other vertices in the graph. It returns
        a list with one value per each vertex in the graph, where value at 
        index 0 is the length of the shortest path from vertex src to v1, etc.
        If certain vertex is not reachable from src, returns inf.
        """
        reachable = dict()      # Dictionary to keep the weight (distance) to each vertex
        pri_queue = []
        heapq.heappush(pri_queue, (0, src))
        
        # Iterate while the priority queue is not empty
        while len(pri_queue) != 0:
            # Remove the first element from the priority queue and assign to vertex
            vertex = heapq.heappop(pri_queue)
            index = vertex[1]
            weight = vertex[0]
            if index not in reachable:
                # Add v to the visited map with weight (distance)
                reachable[index] = weight
                for dest in range(self.v_count):
                    dest_weight = self.adj_matrix[index][dest]
                    if dest_weight > 0:
                        heapq.heappush(pri_queue, (weight + dest_weight, dest))

        output_list = []
        for index in range(self.v_count):
            # If a certain vertex was not reachable from src, it would not be in the dict
            # -> Append value INFINITY (inf) to the output list
            if index not in reachable:
                output_list.append(float('inf'))
            # Otherwise, append the weight from the starting index to reach all other vertices
            else:
                output_list.append(reachable[index])
        
        return output_list


if __name__ == '__main__':

    # print("\nPDF - method add_vertex() / add_edge example 1")
    # print("----------------------------------------------")
    # g = DirectedGraph()
    # print(g)
    # for _ in range(5):
    #     g.add_vertex()
    # print(g)

    # edges = [(0, 1, 10), (4, 0, 12), (1, 4, 15), (4, 3, 3),
    #          (3, 1, 5), (2, 1, 23), (3, 2, 7)]
    # for src, dst, weight in edges:
    #     g.add_edge(src, dst, weight)
    # print(g)


    # print("\nPDF - method get_edges() example 1")
    # print("----------------------------------")
    # g = DirectedGraph()
    # print(g.get_edges(), g.get_vertices(), sep='\n')
    # edges = [(0, 1, 10), (4, 0, 12), (1, 4, 15), (4, 3, 3),
    #          (3, 1, 5), (2, 1, 23), (3, 2, 7)]
    # g = DirectedGraph(edges)
    # print(g.get_edges(), g.get_vertices(), sep='\n')


    print("\nPDF - method is_valid_path() example 1")
    print("--------------------------------------")
    edges = [(0, 1, 10), (4, 0, 12), (1, 4, 15), (4, 3, 3),
             (3, 1, 5), (2, 1, 23), (3, 2, 7)]
    g = DirectedGraph(edges)
    test_cases = [[0, 1, 4, 3], [1, 3, 2, 1], [0, 4], [4, 0], [], [2]]
    for path in test_cases:
        print(path, g.is_valid_path(path))


    # print("\nPDF - method dfs() and bfs() example 1")
    # print("--------------------------------------")
    # edges = [(0, 1, 10), (4, 0, 12), (1, 4, 15), (4, 3, 3),
    #          (3, 1, 5), (2, 1, 23), (3, 2, 7)]
    # g = DirectedGraph(edges)
    # for start in range(5):
    #     print(f'{start} DFS:{g.dfs(start)} BFS:{g.bfs(start)}')


    # print("\nPDF - method has_cycle() example 1")
    # print("----------------------------------")
    # edges = [(0, 1, 10), (4, 0, 12), (1, 4, 15), (4, 3, 3),
    #          (3, 1, 5), (2, 1, 23), (3, 2, 7)]
    # g = DirectedGraph(edges)

    # edges_to_remove = [(3, 1), (4, 0), (3, 2)]
    # for src, dst in edges_to_remove:
    #     g.remove_edge(src, dst)
    #     print(g.get_edges(), g.has_cycle(), sep='\n')

    # edges_to_add = [(4, 3), (2, 3), (1, 3), (4, 0)]
    # for src, dst in edges_to_add:
    #     g.add_edge(src, dst)
    #     print(g.get_edges(), g.has_cycle(), sep='\n')
    # print('\n', g)


    # print("\nPDF - dijkstra() example 1")
    # print("--------------------------")
    # edges = [(0, 1, 10), (4, 0, 12), (1, 4, 15), (4, 3, 3),
    #          (3, 1, 5), (2, 1, 23), (3, 2, 7)]
    # g = DirectedGraph(edges)
    # for i in range(5):
    #     print(f'DIJKSTRA {i} {g.dijkstra(i)}')
    # g.remove_edge(4, 3)
    # print('\n', g)
    # for i in range(5):
    #     print(f'DIJKSTRA {i} {g.dijkstra(i)}')
