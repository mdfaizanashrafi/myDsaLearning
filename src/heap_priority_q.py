
"""
# 📝 **Notes: Heaps and Priority Queues**

## 🔹 1. Introduction

### What is a Heap?
A **heap** is a specialized tree-based data structure that satisfies the **heap property**. It is commonly used to implement **priority queues** and for efficient sorting (e.g., heap sort).

Heaps are typically implemented using arrays and can be visualized as complete binary trees.

---

## 🔹 2. Types of Heaps

There are two main types of heaps:

| Type | Description |
|------|-------------|
| **Max-Heap** | The value of each parent node is greater than or equal to the values of its children. Root contains the maximum element. |
| **Min-Heap** | The value of each parent node is less than or equal to the values of its children. Root contains the minimum element. |

> In both cases, the tree must be a **complete binary tree** (i.e., all levels fully filled except possibly the last one, which is filled left to right).

---

## 🔹 3. Structure of a Heap

Each node in the heap can be represented in an array with the following relationships:

Given index `i`:
- Parent: `floor((i - 1) / 2)`
- Left child: `2 * i + 1`
- Right child: `2 * i + 2`

Example:
```
Index:     0   1   2   3   4   5
Array:    [10, 9, 8, 7, 6, 5]
```
This represents a max-heap.

---

## 🔹 4. Basic Operations on Heap

| Operation | Description | Time Complexity |
|----------|-------------|------------------|
| **Insert** | Adds an element to the heap while maintaining the heap property | O(log n) |
| **Extract Max/Min** | Removes and returns the root (max or min) | O(log n) |
| **Peek (Top)** | Returns the root without removing it | O(1) |
| **Heapify** | Restores heap property after insertion or deletion | O(log n) |
| **Build Heap** | Converts an arbitrary array into a heap | O(n) |
| **Delete** | Deletes a specific element from the heap | O(log n) |

---

## 🔹 5. Heapify

Heapify is the process of converting a binary tree into a heap.

### Max Heapify Algorithm (for array):

```python
def max_heapify(arr, n, i):
    largest = i
    left = 2 * i + 1
    right = 2 * i + 2

    if left < n and arr[left] > arr[largest]:
        largest = left

    if right < n and arr[right] > arr[largest]:
        largest = right

    if largest != i:
        arr[i], arr[largest] = arr[largest], arr[i]
        max_heapify(arr, n, largest)
```

Similarly, we have `min_heapify`.

---

## 🔹 6. Building a Heap

To build a heap from an array of size `n`, we apply `heapify` starting from the last non-leaf node down to the root.

```python
def build_max_heap(arr):
    n = len(arr)
    for i in range(n // 2 - 1, -1, -1):
        max_heapify(arr, n, i)
```

Time complexity: **O(n)**

---

## 🔹 7. Insertion in Heap

Steps:
1. Add the new element at the end of the array.
2. Compare it with its parent and swap if necessary (bubble up).
3. Repeat until heap property is restored.

```python
def insert(arr, value):
    arr.append(value)
    i = len(arr) - 1
    while i > 0 and arr[(i - 1) // 2] < arr[i]:
        parent = (i - 1) // 2
        arr[i], arr[parent] = arr[parent], arr[i]
        i = parent
```

---

## 🔹 8. Deletion in Heap

Only the root can be deleted directly.

Steps:
1. Replace root with the last element.
2. Remove the last element.
3. Heapify the root downwards.

```python
def delete_root(arr):
    n = len(arr)
    if n == 0:
        return None
    root = arr[0]
    arr[0] = arr[-1]
    arr.pop()
    max_heapify(arr, len(arr), 0)
    return root
```

---

## 🔹 8. Applications of Heaps

- **Priority Queue Implementation**
- **Heap Sort**
- **Graph Algorithms** (e.g., Dijkstra’s algorithm, Prim’s MST)
- **Finding k-th Largest/Smallest Element**
- **Merging k Sorted Arrays**

---

## 🔹 9. Priority Queue

### What is a Priority Queue?

A **priority queue** is an abstract data type similar to a queue, where each element has a priority associated with it. Elements with higher priority are served before those with lower priority.

It is typically implemented using a **heap**.

### Key Operations

| Operation | Description |
|----------|-------------|
| `insert(item, priority)` | Inserts an item with given priority |
| `extract_max()` / `extract_min()` | Removes and returns the highest/lowest priority item |
| `peek()` | Returns the highest/lowest priority item without removing |
| `is_empty()` | Checks if queue is empty |

### Implementations

- Using **array** (unsorted or sorted): O(n) or O(1) insert, O(n) extract
- Using **linked list**: Similar to array
- Using **binary heap**: Efficient – O(log n) insert and extract

---

## 🔹 10. Python Implementation (Using heapq)

Python provides a built-in module `heapq` for implementing **min-heaps**.

### Example:

```python
import heapq

# Min-heap example
heap = []
heapq.heappush(heap, 3)
heapq.heappush(heap, 1)
heapq.heappush(heap, 2)

print(heapq.heappop(heap))  # Output: 1
```

For **max-heap**, you can invert the values by inserting negatives.

```python
heap = []
heapq.heappush(heap, -10)
heapq.heappush(heap, -5)
heapq.heappush(heap, -7)

print(-heapq.heappop(heap))  # Output: 10
```

---

## 🔹 11. Time and Space Complexities

| Operation | Time Complexity | Space Complexity |
|----------|------------------|------------------|
| Build Heap | O(n) | O(n) |
| Insert | O(log n) | O(1) |
| Extract Max/Min | O(log n) | O(1) |
| Peek | O(1) | O(1) |
| Delete | O(log n) | O(1) |

---

## 🔹 12. Variants of Heaps

| Variant | Description |
|---------|-------------|
| **Binary Heap** | Standard heap, as discussed above |
| **Binomial Heap** | Supports faster union operations |
| **Fibonacci Heap** | Amortized better performance for some operations (used in advanced algorithms like Dijkstra's) |
| **Ternary Heap** | Each node has up to 3 children |
| **k-ary Heap** | Generalization with k children per node |

---

## 🔹 13. Comparison: Stack, Queue vs Priority Queue

| Data Structure | Order | Use Case |
|----------------|-------|----------|
| Stack | LIFO | Function calls, recursion |
| Queue | FIFO | Task scheduling |
| Priority Queue | Based on priority | Job scheduling, graph algorithms |

---

## 🔹 14. Summary Table

| Feature | Binary Heap (Min/Max) | Priority Queue |
|--------|------------------------|----------------|
| Underlying Structure | Array (Complete Binary Tree) | Usually implemented with heap |
| Insert | O(log n) | O(log n) |
| Delete | O(log n) | O(log n) |
| Access Min/Max | O(1) | O(1) |
| Best for | Dynamic prioritization | Scheduling, optimization problems |

---

## 🔹 15. Practice Problems (LeetCode / GFG Style)

1. **Implement a Max Heap from scratch**
2. **Merge K Sorted Lists**
3. **Find Kth Largest Element in an Array**
4. **Rearrange String k Distance Apart**
5. **Reorganize String**
6. **Task Scheduler**
7. **K Closest Points to Origin**
8. **Top K Frequent Words**
"""

#===========================================================================================

#703: Kth largest term in a stream:
import heapq

class KthLargest:

    def __init__(self, k: int, nums: List[int]):
        self.k = k
        self.min_heap = []
        for num in nums:
            self.add(num)

    def add(self, val: int) -> int:
        heapq.heappush(self.min_heap, val)
        if len(self.min_heap) > self.k:
            heapq.heappop(self.min_heap)
        return self.min_heap[0]

#=============================================================================

#1046. Last Stone Weight

class Solution:
    def lastStoneWeight(self, stones: List[int]) -> int:
        max_heap = [-stone for stone in stones]
        heapq.heapify(max_heap)
        while len(max_heap)>1:
            y = -heapq.heappop(max_heap)
            x= -heapq.heappop(max_heap)
            if y != x:
                new_stone = y-x
                heapq.heappush(max_heap, -new_stone)
            
        return -max_heap[0] if max_heap else 0

#===========================================================================================

#973. K Closest Points to Origin

class Solution:
    def kClosest(self, points: List[List[int]], k: int) -> List[List[int]]:
        max_heap = []
        for x,y in points:
            dist = -(x*x + y*y)
            heapq.heappush(max_heap, (dist, [x,y]))
            if len(max_heap) > k:
                heapq.heappop(max_heap)

        return [point for (_, point) in max_heap]

#================================================================================

#215. Kth Largest Element in an Array

class Solution:
    def findKthLargest(self, nums: List[int], k: int) -> int:
        min_heap = []
        for num in nums:
            heapq.heappush(min_heap, num)
            if len(min_heap) > k:
                heapq.heappop(min_heap)
        return min_heap[0]

#============================================================================

#621. Task Scheduler

from collections import Counter
from typing import List

class Solution:
    def leastInterval(self, tasks: List[str], n: int) -> int:
        if n == 0:
            return len(tasks)  # No cooldown needed

        freq = list(Counter(tasks).values())
        max_freq = max(freq)
        max_count = freq.count(max_freq)

        # Minimum time based on scheduling most frequent tasks
        units_needed = (max_freq - 1) * (n + 1) + max_count

        # Either fit into scheduled slots or just do all tasks sequentially
        return max(len(tasks), units_needed)

#=====================================================================================

#355: Design Twitter

from collections import defaultdict
from typing import List
import heapq

class Twitter:

    def __init__(self):
        self.follow_map = defaultdict(set)  # user -> {followees}
        self.tweets = defaultdict(list)      # user -> [(time, tweetId)]
        self.time = 0                       # Lower numbers = newer tweets

    def postTweet(self, userId: int, tweetId: int) -> None:
        self.tweets[userId].append((self.time, tweetId))
        self.time -= 1  # Decrement so newer tweets have lower time values

    def getNewsFeed(self, userId: int) -> List[int]:
        result = []
        max_heap = []

        # Make sure user follows themselves
        self.follow(userId, userId)

        # Push latest tweet from each followee into heap
        for followee in self.follow_map[userId]:
            if self.tweets[followee]:
                index = len(self.tweets[followee]) - 1
                time, tweetId = self.tweets[followee][index]
                heapq.heappush(max_heap, (time, tweetId, followee, index))

        while max_heap and len(result) < 10:
            time, tweetId, followee, index = heapq.heappop(max_heap)
            result.append(tweetId)

            if index > 0:
                new_index = index - 1
                new_time, new_tweetId = self.tweets[followee][new_index]
                heapq.heappush(max_heap, (new_time, new_tweetId, followee, new_index))

        return result

    def follow(self, followerId: int, followeeId: int) -> None:
        self.follow_map[followerId].add(followeeId)

    def unfollow(self, followerId: int, followeeId: int) -> None:
        if followerId in self.follow_map and followeeId in self.follow_map[followerId]:
            self.follow_map[followerId].remove(followeeId)
        
#========================================================================

#295. Find Median from Data Stream
from heapq import heappush, heappop

class MedianFinder:

    def __init__(self):
        self.max_heap = []
        self.min_heap = []

    def addNum(self, num: int) -> None:
        heappush(self.max_heap, -num)
        if self.max_heap and self.min_heap and (-self.max_heap[0] > self.min_heap[0]):
            val = - heappop(self.max_heap)
            heappush(self.min_heap, val)

        if len(self.max_heap) > len(self.min_heap) +1:
            val = -heappop(self.max_heap)
            heappush(self.min_heap, val)

        elif len(self.min_heap) > len(self.max_heap):
            val = heappop(self.min_heap)
            heappush(self.max_heap, -val)

    def findMedian(self) -> float:
        if len(self.max_heap)>len(self.min_heap):
            return -self.max_heap[0]

        else:
            return (-self.max_heap[0]+self.min_heap[0])/2

#======================================================================================

