# Course: CS261 - Data Structures
# Author: Fusako Obata
# Description:  Implement UndirectedGraph class that's designed to support 
#   following type of graph: undirected, unweighted, no diplicate edges, no loops.
#   Cycles are allowed. Undirected Graph is stored as a Python dictionary of lists
#   where keys are vertex names (strings) and associated values are Python lists 
#   with names (in any order) of vertices connected to the 'key' vertex.

from collections import deque

class UndirectedGraph:
    """
    Class to implement undirected graph
    - duplicate edges not allowed
    - loops not allowed
    - no edge weights
    - vertex names are strings
    """

    def __init__(self, start_edges=None):
        """
        Store graph info as adjacency list
        DO NOT CHANGE THIS METHOD IN ANY WAY
        """
        self.adj_list = dict()

        # populate graph with initial vertices and edges (if provided)
        # before using, implement add_vertex() and add_edge() methods
        if start_edges is not None:
            for u, v in start_edges:
                self.add_edge(u, v)

    def __str__(self):
        """
        Return content of the graph in human-readable form
        DO NOT CHANGE THIS METHOD IN ANY WAY
        """
        out = [f'{v}: {self.adj_list[v]}' for v in self.adj_list]
        out = '\n  '.join(out)
        if len(out) < 70:
            out = out.replace('\n  ', ', ')
            return f'GRAPH: {{{out}}}'
        return f'GRAPH: {{\n  {out}}}'

    # ------------------------------------------------------------------ #

    def add_vertex(self, v: str) -> None:
        """
        Add new vertex to the graph
        """
        if v in self.adj_list:
            return
        self.adj_list[v] = []
        

    def add_edge(self, u: str, v: str) -> None:
        """
        Add edge to the graph
        """
        # If u and v refer to the same vertex, do nothing
        if u == v:
            return
        # Case 1) Both vertex names do not exist in graph
        # -> Create the vertices and create the edge
        if u not in self.adj_list and v not in self.adj_list:
            self.adj_list[u] = [v]
            self.adj_list[v] = [u]
        # Case 2) If either vertex names do not exist in graph
        # -> Create the vertex and create the edge
        elif u not in self.adj_list:
            self.adj_list[u] = [v]
            self.adj_list[v].append(u)
            self.adj_list[v].sort()
        elif v not in self.adj_list:
            self.adj_list[u].append(v)
            self.adj_list[u].sort()
            self.adj_list[v] = [u]
        # Case 3) Both vertex names exist in graph
        # -> Create the edge 
        else:
            # If an edge already exists in graph, do nothing
            if v in self.adj_list[u] or u in self.adj_list[v]:
                return
            # Else, create the edge
            self.adj_list[u].append(v)
            self.adj_list[u].sort()
            self.adj_list[v].append(u)
            self.adj_list[v].sort()


    def remove_edge(self, v: str, u: str) -> None:
        """
        Remove edge from the graph
        """
        if u == v:
            return
        elif u not in self.adj_list and v not in self.adj_list:
            return
        elif u not in self.adj_list or v not in self.adj_list:
            return
        elif u not in self.adj_list[v] or v not in self.adj_list[u]:
            return
        self.adj_list[u].remove(v)
        self.adj_list[v].remove(u)


    def remove_vertex(self, v: str) -> None:
        """
        Remove vertex and all connected edges
        """
        if v not in self.adj_list:
            return
        for key in self.adj_list:
            if v in self.adj_list[key]:
                self.adj_list[key].remove(v)
        del self.adj_list[v]
        

    def get_vertices(self) -> list:
        """
        Return list of vertices in the graph (any order)
        """
        return list(self.adj_list.keys())
       

    def get_edges(self) -> list:
        """
        Return list of edges in the graph (any order)
        """
        output_list = []
        for key in self.adj_list:
            for edges in self.adj_list[key]:
                for edge in edges:
                    if (key,edge) not in output_list and (edge,key) not in output_list:
                        output_list.append((key,edge))
        return output_list
        

    def is_valid_path(self, path: list()) -> bool:
        """
        Return true if provided path is valid, False otherwise
        """
        # If path (the list of vertex names) is empty, consider valid. Return True
        if path is False:
            return True
        # If path's length is 1
        if len(path) == 1:
            # If the path includes a vertex that is not in the graph, return False
            if path[0] not in self.adj_list:
                return False
            # Else, return True
            else:
                return True
        # Check whether the sequence of vertices represents a valid path in the graph
        for index in range(len(path)-1):
            vertex_1 = path[index]
            vertex_2 = path[index+1]
            if vertex_2 not in self.adj_list[vertex_1]:
                return False
        return True
            
       

    def dfs(self, v_start, v_end=None) -> list:
        """
        Return list of vertices visited during DFS search
        Vertices are picked in alphabetical order
        """
        # If v_start isn't in the graph, return an empty list
        if v_start not in self.adj_list:
            return []
        # Step 1) Initialize empty set of visited vertices 
        reachable = []
        # Step 2) Initialize stack with v_start
        stack = [v_start]   

        while len(stack) != 0:
            # Step 3) If the stack is not empty, pop a vertex
            vertex = stack.pop() 
            # Step 3.5) If the vertex is v_end, end the loop and return the visited vertices
            if vertex == v_end:
                reachable.append(vertex)
                return reachable
            # Step 4) If vertex is not in the set of visited vertices,
            if vertex not in reachable:
                # -> Add vertex to the set of visited vertices
                reachable.append(vertex)
                successors = list(self.adj_list[vertex])
                # -> Push each vertex that's a direct successor of vertex to stack
                while successors:
                    successor = successors.pop()
                    stack.append(successor)

        return reachable
       

    def bfs(self, v_start, v_end=None) -> list:
        """
        Return list of vertices visited during BFS search
        Vertices are picked in alphabetical order
        """
        # If v_start isn't in the graph, return an empty list
        if v_start not in self.adj_list:
            return []
        # Step 1) Initialize empty set of visited vertices 
        reachable = []
        # Step 2) Initialize queue with v_start
        queue = deque(v_start)  # Set a queue (dequeue based)

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
                successors = deque(self.adj_list[vertex])
                # -> If each successor is not in the set of visited vertices, enqueue to queue
                while successors:
                    successor = successors.popleft()
                    if successor not in reachable:
                        queue.appendleft(successor)
        
        return reachable

    def count_connected_components(self, stack = None, count = 0, ver_visited = None):
        """
        Return number of connected componets in the graph
        """
        if self.adj_list is False:
            return 0
        if stack == None:
            stack = [self.get_vertices()[0]]
            ver_visited = []

        reachable = []
        # Do a DFS search
        while len(stack) != 0:
            vertex = stack.pop() 
            if vertex not in reachable:
                reachable.append(vertex)
                successors = list(self.adj_list[vertex])
                while successors:
                    successor = successors.pop()
                    stack.append(successor)            
        # Once it runs, update the count
        count += 1
        # Update how many vertices we've visited
        ver_visited = sorted(ver_visited + reachable)
        # Check what are the remaining vertices we have not visited
        vertices_missed = list(set(self.get_vertices()) - set(ver_visited)) + list(set(ver_visited) - set(self.get_vertices()))
        # If we have not visited all the vertices, continue checking the remaining vertices
        if ver_visited != sorted(self.get_vertices()):
            stack = [vertices_missed[0]]
            return self.count_connected_components(stack, count, ver_visited)
        # Otherwise, return the count
        else:
            return count


    def has_cycle(self):
        """
        Return True if graph contains a cycle, False otherwise
        """
        # Step 1) Initialize empty set of visited vertices 
        reachable = []
        vertices = self.get_vertices()  # Get a list of vertices from the graph

        # Iterate until we go through all the vertices in graph
        while len(vertices) != 0:
            # Pop the first vertex from the list of vertices
            stack = [vertices.pop(0)]
            # Iterate until the stack is empty
            while len(stack) != 0:
                # If the stack is not empty, pop a vertex
                vertex = stack.pop() 
                # If vertex is not in the set of visited vertices,
                if vertex not in reachable:
                    # -> Add vertex to the set of visited vertices
                    reachable.append(vertex)
                    successors = list(self.adj_list[vertex])
                    # -> Push each vertex that's a direct successor of vertex to stack
                    while successors:
                        successor = successors.pop()
                        if successor in stack:
                            if parent != vertex:
                                return True
                        stack.append(successor)
                    parent = vertex
        return False
   


if __name__ == '__main__':

    # print("\nPDF - method add_vertex() / add_edge example 1")
    # print("----------------------------------------------")
    # g = UndirectedGraph()
    # print(g)

    # for v in 'ABCDE':
    #     g.add_vertex(v)
    # print(g)

    # g.add_vertex('A')
    # print(g)

    # for u, v in ['AB', 'AC', 'BC', 'BD', 'CD', 'CE', 'DE', ('B', 'C')]:
    #     g.add_edge(u, v)
    # print(g)


    # print("\nPDF - method remove_edge() / remove_vertex example 1")
    # print("----------------------------------------------------")
    # g = UndirectedGraph(['AB', 'AC', 'BC', 'BD', 'CD', 'CE', 'DE'])
    # g.remove_vertex('DOES NOT EXIST')
    # g.remove_edge('A', 'B')
    # g.remove_edge('X', 'B')
    # print(g)
    # g.remove_vertex('D')
    # print(g)


    # print("\nPDF - method get_vertices() / get_edges() example 1")
    # print("---------------------------------------------------")
    # g = UndirectedGraph()
    # print(g.get_edges(), g.get_vertices(), sep='\n')
    # g = UndirectedGraph(['AB', 'AC', 'BC', 'BD', 'CD', 'CE'])
    # print(g.get_edges(), g.get_vertices(), sep='\n')


    # print("\nPDF - method is_valid_path() example 1")
    # print("--------------------------------------")
    # g = UndirectedGraph(['AB', 'AC', 'BC', 'BD', 'CD', 'CE', 'DE'])
    # test_cases = ['ABC', 'ADE', 'ECABDCBE', 'ACDECB', '', 'D', 'Z']
    # for path in test_cases:
    #     print(list(path), g.is_valid_path(list(path)))


    # print("\nPDF - method dfs() and bfs() example 1")
    # print("--------------------------------------")
    # edges = ['AE', 'AC', 'BE', 'CE', 'CD', 'CB', 'BD', 'ED', 'BH', 'QG', 'FG']
    # g = UndirectedGraph(edges)
    # test_cases = 'ABCDEGH'
    # for case in test_cases:
    #     print(f'{case} DFS:{g.dfs(case)} BFS:{g.bfs(case)}')
    # print('-----')
    # for i in range(1, len(test_cases)):
    #     v1, v2 = test_cases[i], test_cases[-1 - i]
    #     print(f'{v1}-{v2} DFS:{g.dfs(v1, v2)} BFS:{g.bfs(v1, v2)}')


    # print("\nPDF - method count_connected_components() example 1")
    # print("---------------------------------------------------")
    # edges = ['AE', 'AC', 'BE', 'CE', 'CD', 'CB', 'BD', 'ED', 'BH', 'QG', 'FG']
    # g = UndirectedGraph(edges)
    # test_cases = (
    #     'add QH', 'remove FG', 'remove GQ', 'remove HQ',
    #     'remove AE', 'remove CA', 'remove EB', 'remove CE', 'remove DE',
    #     'remove BC', 'add EA', 'add EF', 'add GQ', 'add AC', 'add DQ',
    #     'add EG', 'add QH', 'remove CD', 'remove BD', 'remove QG')
    # for case in test_cases:
    #     command, edge = case.split()
    #     u, v = edge
    #     g.add_edge(u, v) if command == 'add' else g.remove_edge(u, v)
    #     print(g.count_connected_components(), end=' ')
    # print()


    print("\nPDF - method has_cycle() example 1")
    print("----------------------------------")
    edges = ['AE', 'AC', 'BE', 'CE', 'CD', 'CB', 'BD', 'ED', 'BH', 'QG', 'FG']
    g = UndirectedGraph(edges)
    test_cases = (
        'add QH', 'remove FG', 'remove GQ', 'remove HQ',
        'remove AE', 'remove CA', 'remove EB', 'remove CE', 'remove DE',
        'remove BC', 'add EA', 'add EF', 'add GQ', 'add AC', 'add DQ',
        'add EG', 'add QH', 'remove CD', 'remove BD', 'remove QG',
        'add FG', 'remove GE')
    for case in test_cases:
        command, edge = case.split()
        u, v = edge
        g.add_edge(u, v) if command == 'add' else g.remove_edge(u, v)
        print('{:<10}'.format(case), g.has_cycle())
