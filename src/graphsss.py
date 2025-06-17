"""
# 🌐 Intro to Graphs

## 🧠 What Is a Graph?

A **graph** is a collection of **nodes (vertices)** connected by **edges**.

Think of it like this:

> A graph is like a **map of cities connected by roads**

- Nodes = cities
- Edges = roads between cities

You can use graphs to represent:
- Social networks
- Web pages and links
- Maps and routes
- Dependencies in projects

---

## 🔁 Types of Graphs

| Type | Description | Example |
|------|-------------|---------|
| **Undirected** | Edges go both ways (like Facebook friends) | A <-> B |
| **Directed** | Edge goes one way only (like Twitter follows) | A -> B |
| **Weighted** | Each edge has a value (like road distance) | A -5-> B |
| **Unweighted** | All edges are equal | A -> B |

---

## 📦 Graph Representations

There are two main ways to represent a graph:

### 1. **Adjacency List**
- Most common for algorithms
- Each node points to its neighbors

#### Example:

```python
graph = {
    'A': ['B', 'C'],
    'B': ['A', 'D'],
    'C': ['A', 'D'],
    'D': ['B', 'C']
}
```

This means:
- `A` is connected to `B` and `C`
- `B` is connected to `A` and `D`, etc.

✅ Good for sparse graphs (few connections)

---

### 2. **Adjacency Matrix**
- Uses a 2D matrix to show if there’s an edge between nodes

#### Example:

```
     A  B  C  D
   A 0  1  1  0
   B 1  0  0  1
   C 1  0  0  1
   D 0  1  1  0
```

✅ Good for dense graphs (many connections)

But uses more memory than needed for large, sparse graphs ❌

---

## 🧮 Common Graph Problems

| Problem | Use Case |
|--------|----------|
| BFS / DFS | Traverse or search a graph |
| Shortest Path | Google Maps, routing |
| Cycle Detection | Deadlock detection in OS |
| Topological Sort | Task scheduling |
| MST / Union-Find | Network design |
| Dijkstra / A* | GPS navigation |
| Min Cut / Max Flow | Traffic optimization |

---

# 🔍 Graph Traversal: BFS vs DFS

Let’s say we want to visit all nodes starting from a source node.

We can do that with either:
- **DFS (Depth-First Search)** – go as deep as possible before backtracking
- **BFS (Breadth-First Search)** – explore level-by-level

---

## 🔄 Breadth-First Search (BFS)

### 🔁 Idea:
Use a **queue** to explore neighbor nodes first, then move outward.

### 🧒 Analogy:
Imagine you're throwing a stone into a pond → ripples spread out evenly.

### ✅ Use Cases:
- Shortest path in unweighted graph
- Level order traversal
- Connected components

### 🧪 Example (Matrix BFS):

Given grid:

```python
grid = [
  [1, 1, 0],
  [0, 1, 1]
]
```

To find number of islands:
- Start at `(0,0)` → mark all connected 1s
- Then find next unvisited 1 → new island

---

## 🧭 Depth-First Search (DFS)

### 🔁 Idea:
Go as deep as possible, then backtrack when stuck.

### 🧒 Analogy:
Like exploring a maze: go down one path until you hit a dead end, then backtrack.

### ✅ Use Cases:
- Detect cycles
- Topological sort
- Solving puzzles like Sudoku

### 🧪 Example (DFS on adjacency list):

```python
graph = {
    'A': ['B', 'C'],
    'B': ['A', 'D'],
    'C': ['A', 'D'],
    'D': ['B', 'C']
}

def dfs(node, visited):
    visited.add(node)
    print(node)
    for neighbor in graph[node]:
        if neighbor not in visited:
            dfs(neighbor, visited)

dfs('A', set())
```

Output: `A B D C` (or any valid order depending on direction)

---

# 🧱 Adjacency List (and More)

## 1. **Adjacency List**
Most commonly used representation.

### Python Example:
```python
graph = {
    0: [1, 2],
    1: [2],
    2: [0, 3],
    3: [1, 2]
}
```

- Node 0 is connected to 1 and 2
- Node 3 is connected to 1 and 2

---

## 2. **Adjacency Matrix**

A 2D array where `matrix[i][j] == 1` means there's an edge from `i` to `j`.

```python
matrix = [
  [0, 1, 1, 0],
  [0, 0, 0, 1],
  [1, 0, 0, 1],
  [0, 1, 1, 0]
]
```

✅ Easy to check connection between two nodes  
❌ Takes O(n²) space → not efficient for sparse graphs

---

## 3. **Edge List**

Just a list of edges:

```python
edges = [(0, 1), (0, 2), (1, 3), (2, 3)]
```

✅ Simple and easy to store  
❌ Hard to get neighbors of a node quickly

---

## 4. **Incidence Matrix** (Advanced)

Used for **hypergraphs** or graphs with multiple edges per node.

Each row is a node, each column is an edge.

```
      e1  e2  e3
A     1   1   0
B     1   0   1
C     0   1   1
```

✅ Useful in network flow problems  
❌ Rarely used in coding interviews

---

# 🧩 Graph Algorithms Overview

| Algorithm | Purpose | Time Complexity |
|----------|---------|------------------|
| BFS | Explore layer by layer | O(V + E) |
| DFS | Go deep before backtracking | O(V + E) |
| Dijkstra | Find shortest path (non-negative weights) | O(E log V) |
| Bellman-Ford | Same as above, but handles negative weights | O(V * E) |
| Floyd-Warshall | All-pairs shortest path | O(V³) |
| Kruskal / Prim | Minimum Spanning Tree | O(E log V) |
| Topological Sort | Order tasks with dependencies | O(V + E) |
| Union-Find | Detect cycles | O(α(V)) per operation |

---

# 🎯 Real-World Graph Examples

| Use Case | How It Works |
|----------|--------------|
| Social Networks | Users are nodes, friendships are edges |
| Google Maps | Cities are nodes, roads are edges |
| Web Crawling | Pages are nodes, hyperlinks are edges |
| Dependency Resolution | Tasks are nodes, arrows show order |
| Course Prerequisites | Courses are nodes, directed edges show requirements |

---

# 🧒 Kids-Friendly Analogy: Friends at a Party

Imagine kids at a party:
- Some kids are friends
- You draw lines between those who know each other

If you want to invite someone's whole friend group:
- Use **DFS/BFS** to find everyone connected to them

That’s how social media finds your "suggested" friends!

---

## 🧠 Summary Table

| Concept | Description |
|--------|-------------|
| **Node (Vertex)** | Individual item (person, city, task) |
| **Edge** | Connection between two nodes |
| **BFS** | Goes wide first → best for shortest paths |
| **DFS** | Goes deep first → best for path finding |
| **Adjacency List** | Best for most problems |
| **Adjacency Matrix** | Fast lookup, high memory usage |
| **Edge List** | Simple, hard to traverse |

"""

