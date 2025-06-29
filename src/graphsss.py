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

#================================================================================
from collections import deque
from typing import List

#200. Number of Islands

class Solution:
    def numIslands(self, grid: List[List[str]]) -> int:
        if not grid:
            return 0

        rows, cols = len(grid), len(grid[0])

        def dfs(r,c):
            if r<0 or c< 0 or r>=rows or c>=cols or grid[r][c] == "0":
                return
            grid[r][c]="0"

            dfs(r+1,c)
            dfs(r-1,c)
            dfs(r,c+1)
            dfs(r,c-1)

        count = 0
        for r in range(rows):
            for c in range(cols):
                if grid[r][c]=="1":
                    dfs(r,c)
                    count += 1
        return count 
    
#=========================================================================

#695. Max Area of Island

class Solution:
    def maxAreaOfIsland(self, grid: List[List[int]]) -> int:
        if not grid:
            return 0
        rows= len(grid)
        cols = len(grid[0])

        def dfs(r,c):
            if r<0 or c<0 or r>= rows or c>= cols or grid[r][c] == 0:
                return 0
            grid[r][c]=0

            area =1
            area += dfs(r+1,c)
            area+= dfs(r-1,c)
            area+= dfs(r,c+1)
            area+= dfs(r, c-1)
            return area

        max_area = 0
        for r in range(rows):
            for c in range(cols):
                if grid[r][c]==1:
                    current_area = dfs(r,c)
                    max_area = max(max_area, current_area)

        return max_area

#====================================================================

#133. Clone Graph


# Definition for a Node.
class Node:
    def __init__(self, val = 0, neighbors = None):
        self.val = val
        self.neighbors = neighbors if neighbors is not None else []


from typing import Optional
class Solution:
    def cloneGraph(self, node: Optional['Node']) -> Optional['Node']:
        if not node:
            return None

        old_to_new = {}
        queue = deque([node])
        old_to_new[node] = Node(node.val)

        while queue:
            curr = queue.popleft()
            for neighbor in curr.neighbors:
                if neighbor not in old_to_new:
                    old_to_new[neighbor] = Node(neighbor.val)
                    queue.append(neighbor)
                old_to_new[curr].neighbors.append(old_to_new[neighbor])

        return old_to_new[node]

#=============================================================================

#LeetCode Problem #286: Walls and Gates

class Solution:
    def islandsAndTreasure(self, grid: List[List[int]]) -> None:
        ROWS, COLS = len(grid), len(grid[0])
        queue = deque()
        directions = [(-1,0),(1,0),(0,-1),(0,1)]

        for r in range(ROWS):
            for c in range(COLS):
                if grid[r][c]==0:
                    queue.append((r,c))

        while queue:
            r,c = queue.popleft()
            for dr,dc in directions:
                nr,nc = r+dr, c+dc

                if 0<=nr <ROWS and 0<=nc<COLS and grid[nr][nc] == 2147483647:
                    grid[nr][nc] = grid[r][c]+1
                    queue.append((nr,nc))

#======================================================================================

#994. Rotting Oranges

class Solution:
    def orangesRotting(self, grid: List[List[int]]) -> int:
        if not grid or not grid[0]:
            return 0

        rows, cols = len(grid), len(grid[0])
        queue = deque()
        fresh = 0
        for r in range(rows):
            for c in range(cols):
                if grid[r][c]==1:
                    fresh += 1
                elif grid[r][c] == 2:
                    queue.append((r,c))

        directions=[(-1,0),(1,0),(0,-1),(0,1)]
        time = 0

        while queue and fresh > 0:
            for _ in range(len(queue)):
                r,c = queue.popleft()
                for dr,dc in directions:
                    nr,nc = r+dr, c+dc

                    if 0<=nr<rows and 0<=nc<cols and grid[nr][nc]==1:
                        grid[nr][nc]=2

                        fresh -= 1
                        queue.append((nr,nc))
            time += 1
        return time if fresh == 0 else -1

#======================================================================================

#417. Pacific Atlantic Water Flow

class Solution:
    def pacificAtlantic(self, heights: List[List[int]]) -> List[List[int]]:
        if not heights or not heights[0]:
            return []
        rows, cols = len(heights), len(heights[0])
        directions = [(-1,0),(1,0),(0,-1),(0,1)]

        can_reach_pacific = [[False]*cols for _ in range(rows)]
        can_reach_atlantic = [[False]*cols for _ in range(rows)]

        def dfs(r,c,visited):
            if visited[r][c]:
                return 
            visited[r][c] = True

            for dr,dc in directions:
                nr,nc = r+dr, c+dc

                if 0<=nr<rows and 0<=nc<cols:
                    if heights[nr][nc] >= heights[r][c]:
                        dfs(nr,nc,visited)

        for r in range(rows):
            dfs(r,0,can_reach_pacific)
            dfs(r,cols-1,can_reach_atlantic)

        for c in range(cols):
            dfs(0,c,can_reach_pacific)
            dfs(rows-1, c, can_reach_atlantic)

        result = []

        for r in range(rows):
            for c in range(cols):
                if can_reach_pacific[r][c] and can_reach_atlantic[r][c]:
                    result.append([r,c])

        return result
    
#====================================================================================

#130. Surrounded Regions

class Solution:
    def solve(self, board: List[List[str]]) -> None:
        """
        Do not return anything, modify board in-place instead.
        """
        if not board or not board[0]:
            return

        rows, cols = len(board), len(board[0])

        def dfs(r,c):
            if r<0 or c<0 or r>= rows or c>= cols or board[r][c] != "O":
                return

            board[r][c] = "T"

            dfs(r+1,c)
            dfs(r-1,c)
            dfs(r,c+1)
            dfs(r,c-1)

        for r in range(rows):
            for c in range(cols):
                if board[r][c] == "O" and (r==0 or r==rows-1 or c==0 or c== cols-1):
                    dfs(r,c)
        
        for r in range(rows):
            for c in range(cols):
                if board[r][c]=="O":
                    board[r][c]="X"

        for r in range(rows):
            for c in range(cols):
                if board[r][c]=="T":
                    board[r][c]= "O"

#=====================================================================================

#207. Course Schedule

class Solution:
    def canFinish(self, numCourses: int, prerequisites: List[List[int]]) -> bool:
        graph = [[] for _ in range(numCourses)]
        in_degree = [0]*numCourses
        for dest, src in prerequisites:
            graph[src].append(dest)
            in_degree[dest] += 1

        queue= deque()
        for r in range(numCourses):
            if in_degree[r] == 0:
                queue.append(r)
        count = 0

        while queue:
            curr = queue.popleft()
            count += 1
            for neighbor in graph[curr]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor]==0:
                    queue.append(neighbor)

        return count == numCourses
    
#========================================================================================

#210. Course Schedule II

class Solution:
    def findOrder(self, numCourses: int, prerequisites: List[List[int]]) -> List[int]:
        graph = [[] for _ in range(numCourses)]
        in_degree = [0]*numCourses

        for dest, src in prerequisites:
            graph[src].append(dest)
            in_degree[dest] += 1
        queue = deque([i for i in range(numCourses) if in_degree[i] ==0])

        result = []
        while queue:
            curr = queue.popleft()
            result.append(curr)
            for neighbor in graph[curr]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)
            
        return result if len(result) == numCourses else []

#=============================================================================================

#LeetCode 261: Graph Valid Tree

class Solution:
    def validTree(self, n: int, edges: List[List[int]]) -> bool:
        # Edge case: no edges → empty or single node is always a tree
        if n == 0:
            return True

        parent = list(range(n))

        def find(x):
            if parent[x] != x:
                parent[x] = find(parent[x])
            return parent[x]

        def union(x, y):
            root_x = find(x)
            root_y = find(y)
            if root_x == root_y:
                return False  # Already connected → cycle!
            parent[root_x] = root_y
            return True

        for u, v in edges:
            if not union(u, v):
                return False  # Cycle found

        # Finally: should have exactly n - 1 unions
        return sum(find(i) == i for i in range(n)) == 1
    
#==============================================================================================

#LeetCode 547: Number of Connected Components in an Undirected Graph

from typing import List
from collections import defaultdict

class Solution:
    def countComponents(self, n: int, edges: List[List[int]]) -> int:
        # Step 1: Build adjacency list
        graph = defaultdict(list)
        for u, v in edges:
            graph[u].append(v)
            graph[v].append(u)

        visited = set()
        
        def dfs(node):
            for neighbor in graph[node]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    dfs(neighbor)

        components = 0
        for i in range(n):
            if i not in visited:
                visited.add(i)
                dfs(i)
                components += 1

        return components
    
#============================================================================

#684. Redundant Connection

class Solution:
    def findRedundantConnection(self, edges: List[List[int]]) -> List[int]:
        parent = list(range(1001))
        def find(x):
            if parent[x] != x:
                parent[x] = find(parent[x])
            return parent[x]
        
        def union(x,y):
            root_x = find(x)
            root_y = find(y)
            if root_x == root_y:
                return False
            
            parent[root_x]= root_y

            return True

        for u,v in edges:
            if not union(u,v):
                return [u,v]
            
        return []
    
#==================================================================================

#127. Word Ladder

class Solution:
    def ladderLength(self, beginWord: str, endWord: str, wordList: List[str]) -> int:
        if endWord not in wordList:
            return 0

        L = len(beginWord)
        pattern_map = defaultdict(list)
        for word in wordList:
            for i in range(L):
                pattern = word[:i] + "*" +word[i+1:]
                pattern_map[pattern].append(word)
        visited = set([beginWord])
        queue = deque([(beginWord,1)])

        while queue:
            current_word, level = queue.popleft()
            for i in range(L):
                pattern = current_word[:i]+"*"+current_word[i+1:]
                for neighbor in pattern_map[pattern]:
                    if neighbor == endWord:
                        return level +1
                    if neighbor not in visited:
                        visited.add(neighbor)
                        queue.append((neighbor, level+1))
                
        return 0

