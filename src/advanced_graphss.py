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

#743. Network Delay Time

import heapq
from collections import defaultdict

class Solution:
    def networkDelayTime(self, times: List[List[int]], n: int, k: int) -> int:
        # Step 1: Build adjacency list
        graph = defaultdict(list)
        for u, v, w in times:
            graph[u].append((v, w))
        
        # Step 2: Use min-heap for Dijkstra
        min_heap = [(0, k)]  # (distance, node)
        visited = {}
        
        while min_heap:
            dist, node = heapq.heappop(min_heap)
            
            if node in visited:
                continue
            
            visited[node] = dist
            
            # Visit neighbors
            for neighbor, weight in graph[node]:
                if neighbor not in visited:
                    heapq.heappush(min_heap, (dist + weight, neighbor))
        
        # Step 3: If all nodes visited, return max delay
        return max(visited.values()) if len(visited) == n else -1

#============================================================================

#332. Reconstruct Itinerary

from collections import defaultdict
from typing import List

class Solution:
    def findItinerary(self, tickets: List[List[str]]) -> List[str]:
        # Step 1: Build adjacency list with lex order using min-heap
        graph = defaultdict(list)
        
        # Sort in reverse order so we can pop from end (faster than inserting at front)
        for src, dst in sorted(tickets, reverse=True):
            graph[src].append(dst)
        
        result = []

        def dfs(node):
            # Visit all neighbors in lex order
            while graph[node]:
                next_node = graph[node].pop()
                dfs(next_node)
            result.append(node)

        # Start from JFK
        dfs("JFK")

        # Result is reversed because we append at end
        return result[::-1]

#==================================================================================

#1584. Min Cost to Connect All Points

class Solution:
    def minCostConnectPoints(self, points: List[List[int]]) -> int:
        n = len(points)
        if n == 0:
            return 0

        heap = [(0, 0)]  # (cost, node_index)
        total_cost = 0
        visited = set()
        min_heap = []

        while heap:
            cost, u = heapq.heappop(heap)

            if u in visited:
                continue

            visited.add(u)
            total_cost += cost

            # Add all unvisited neighbors to heap
            for v in range(n):
                if v not in visited:
                    dist = abs(points[u][0] - points[v][0]) + abs(points[u][1] - points[v][1])
                    heapq.heappush(heap, (dist, v))

        return total_cost if len(visited) == n else -1

#=============================================================================

#778. Swim in Rising Water

import heapq
from typing import List

class Solution:
    def swimInWater(self, grid: List[List[int]]) -> int:
        n = len(grid)
        visited = [[False] * n for _ in range(n)]
        min_heap = [(grid[0][0], 0, 0)]  # (max_time, row, col)

        directions = [(-1,0), (1,0), (0,-1), (0,1)]

        while min_heap:
            time, r, c = heapq.heappop(min_heap)

            if r == n - 1 and c == n - 1:
                return time

            if visited[r][c]:
                continue
            visited[r][c] = True

            for dr, dc in directions:
                nr, nc = r + dr, c + dc
                if 0 <= nr < n and 0 <= nc < n and not visited[nr][nc]:
                    # Push the new path with max time required
                    new_time = max(time, grid[nr][nc])
                    heapq.heappush(min_heap, (new_time, nr, nc))

        return -1  # Shouldn't reach here if grid is valid

#================================================================================

#787. Cheapest Flights Within K Stops

class Solution:
    def findCheapestPrice(self, n: int, flights: List[List[int]], src: int, dst: int, k: int) -> int:
        # Initialize cost array
        cost = [float('inf')] * n
        cost[src] = 0

        # Run Bellman-Ford for k+1 iterations
        for _ in range(k + 1):
            curr_cost = cost[:]
            updated = False

            for u, v, w in flights:
                if cost[u] != float('inf'):
                    if cost[v] > cost[u] + w:
                        curr_cost[v] = min(curr_cost[v], cost[u] + w)
                        updated = True

            cost = curr_cost
            if not updated:
                break  # Early exit if no updates

        return cost[dst] if cost[dst] != float('inf') else -1

#===============================================================================================

#269: Alien Dictionary

from collections import defaultdict, Counter, deque
from typing import List

class Solution:
    def alienOrder(self, words: List[str]) -> str:
        # Step 1: Build graph and in-degree count
        graph = defaultdict(set)
        in_degree = Counter({c: 0 for word in words for c in word})

        # Step 2: Build edges from adjacent words
        for i in range(len(words) - 1):
            word1, word2 = words[i], words[i + 1]

            min_len = min(len(word1), len(word2))
            prefix_match = True

            for j in range(min_len):
                if word1[j] != word2[j]:
                    # Found a rule: word1[j] -> word2[j]
                    if word2[j] not in graph[word1[j]]:
                        graph[word1[j]].add(word2[j])
                        in_degree[word2[j]] += 1
                    prefix_match = False
                    break

            # Edge case: word1 is longer than word2 but all chars match (invalid dict)
            if prefix_match and len(word1) > len(word2):
                return ""

        # Step 3: Kahn's Algorithm for Topological Sort
        queue = deque([c for c in in_degree if in_degree[c] == 0])
        order = []

        while queue:
            char = queue.popleft()
            order.append(char)

            for neighbor in graph[char]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)

        # Step 4: Check if all characters were included (no cycles)
        if len(order) < len(in_degree):
            return ""  # Cycle detected

        return ''.join(order)
    
#=====================================================================================