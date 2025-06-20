"""
# 🧠 Advanced Graph Algorithms Revision Notes

Below is a **comprehensive revision guide on Advanced Graph Algorithms**, covering:

- 🔹 **Dijkstra’s Algorithm** – for shortest paths  
- 🟢 **Prim’s Algorithm** – for minimum spanning trees  
- 🔸 **Kruskal’s Algorithm** – for minimum spanning trees  
- 🟠 **Topological Sort** – for directed acyclic graphs (DAGs)

## 1. 🔹 Dijkstra’s Algorithm  
### Shortest Path in Weighted Graph (No Negative Weights)

### 💡 Idea:
Use a **priority queue (min-heap)** to always expand the node with the smallest known distance from the source.

### ✅ When to Use:
- Weighted, **directed or undirected** graph
- **No negative edge weights**
- Find **shortest path from one node to all others**

### 📌 Steps:
1. Initialize distances to `infinity`, except source = `0`.
2. Use min-heap to pick next node with smallest distance.
3. For each neighbor, relax the edge: update if shorter path found.

### 🧑‍💻 Python Code:

```python
import heapq

def dijkstra(graph, start):
    dist = {node: float('inf') for node in graph}
    dist[start] = 0
    heap = [(0, start)]

    while heap:
        curr_dist, u = heapq.heappop(heap)
        if curr_dist > dist[u]:
            continue
        for v, weight in graph[u]:
            if dist[v] > dist[u] + weight:
                dist[v] = dist[u] + weight
                heapq.heappush(heap, (dist[v], v))
    return dist
```

### ⏱️ Time Complexity:
- **Adjacency List + Min-Heap**: `O((V + E) log V)`
- **Adjacency Matrix**: `O(V²)`

---

## 2. 🟢 Prim’s Algorithm  
### Minimum Spanning Tree (MST)

### 💡 Idea:
Start from any node, greedily add the **minimum weight edge** connecting the MST to a new node.

### ✅ When to Use:
- Weighted, **undirected**, connected graph
- Find **Minimum Spanning Tree (MST)**

### 📌 Steps:
1. Start from an arbitrary node.
2. Use min-heap to select next edge with minimum weight.
3. Add node to MST, update neighbors.

### 🧑‍💻 Python Code:

```python
def prim(graph, start):
    mst = []
    visited = set([start])
    edges = [(cost, start, to) for to, cost in graph[start]]
    heapq.heapify(edges)
    
    while edges:
        cost, u, v = heapq.heappop(edges)
        if v not in visited:
            visited.add(v)
            mst.append((u, v, cost))
            for to, cost in graph[v]:
                if to not in visited:
                    heapq.heappush(edges, (cost, v, to))
    return mst
```

### ⏱️ Time Complexity:
- **Adjacency List + Min-Heap**: `O(E log V)`
- **Adjacency Matrix**: `O(V²)`

---

## 3. 🔸 Kruskal’s Algorithm  
### Minimum Spanning Tree (MST)

### 💡 Idea:
Sort all edges by weight and pick them one by one, **avoiding cycles** using **Union-Find (Disjoint Set Union - DSU)**.

### ✅ When to Use:
- Weighted, **undirected**, connected graph
- Find **Minimum Spanning Tree (MST)**

### 📌 Steps:
1. Sort all edges by weight.
2. Use DSU to detect cycles.
3. Pick edge if it connects disjoint sets.

### 🧑‍💻 Python Code:

```python
class DSU:
    def __init__(self, n):
        self.parent = list(range(n))
    
    def find(self, x):
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]
    
    def union(self, x, y):
        xroot = self.find(x)
        yroot = self.find(y)
        if xroot == yroot:
            return False
        self.parent[yroot] = xroot
        return True

def kruskal(graph, n):
    dsu = DSU(n)
    mst = []
    graph.sort(key=lambda x: x[2])  # sort by weight
    for u, v, w in graph:
        if dsu.union(u, v):
            mst.append((u, v, w))
    return mst
```

### ⏱️ Time Complexity:
- **Sorting Edges**: `O(E log E)`
- **Union-Find Operations**: Nearly `O(α(V))` per operation → almost constant

---

## 4. 🟠 Topological Sort  
### Linear Ordering of DAG

### 💡 Idea:
Order nodes such that for every directed edge `u → v`, `u` comes before `v`.

### ✅ When to Use:
- Directed Acyclic Graph (DAG)
- Task scheduling with dependencies
- Course prerequisite problems

### 📌 Two Main Methods:
1. **Kahn’s Algorithm (BFS-based)**
2. **DFS-based with Stack**

### 🧑‍💻 Kahn's Algorithm (BFS):

```python
from collections import defaultdict, deque

def topological_sort_kahn(graph, num_nodes):
    indegree = [0] * num_nodes
    for u in graph:
        for v in graph[u]:
            indegree[v] += 1
    
    queue = deque([i for i in range(num_nodes) if indegree[i] == 0])
    order = []

    while queue:
        u = queue.popleft()
        order.append(u)
        for v in graph[u]:
            indegree[v] -= 1
            if indegree[v] == 0:
                queue.append(v)
    
    if len(order) != num_nodes:
        return []  # cycle exists
    return order
```

### 🧑‍💻 DFS-Based:

```python
def topological_sort_dfs(graph, num_nodes):
    visited = [False] * num_nodes
    stack = []

    def dfs(u):
        visited[u] = True
        for v in graph[u]:
            if not visited[v]:
                dfs(v)
        stack.append(u)

    for u in range(num_nodes):
        if not visited[u]:
            dfs(u)

    return stack[::-1]
```

### ⏱️ Time Complexity:
- **O(V + E)** – both methods are linear in size of graph

---

## ✅ Summary Table

| Algorithm       | Purpose                     | Data Structure Used     | Time Complexity   |
|----------------|-----------------------------|--------------------------|--------------------|
| Dijkstra        | Single-source shortest path | Min-heap                 | O((V+E) log V)     |
| Prim            | MST                         | Min-heap                 | O(E log V)         |
| Kruskal         | MST                         | Union-Find               | O(E log E)         |
| Topological Sort| Linear ordering of DAG      | Queue / Stack + DFS/BFS  | O(V + E)           |

---

## ✅ Real-world Applications

| Algorithm       | Application Examples                                                  |
|----------------|------------------------------------------------------------------------|
| Dijkstra        | GPS navigation, network routing                                       |
| Prim/Kruskal    | Network design, clustering, laying cables                             |
| Topological Sort| Build systems, course prerequisites, task scheduling                  |
"""
#=================================================================================================

