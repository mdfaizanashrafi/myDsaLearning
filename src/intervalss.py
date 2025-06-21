"""
# 📅 **Intervals Revision Notes for DSA**

Interval problems are commonly seen in coding interviews and involve working with ranges of numbers or time intervals. They often require sorting, merging, overlapping checks, and greedy strategies.

---

## ✅ 1. **What is an Interval?**

An interval is represented as:
```python
[start, end]
```
- `start` ≤ `end`
- It includes all values from `start` to `end`, inclusive.

Examples:
- `[1, 3]` → 1, 2, 3
- `[5, 7]` → 5, 6, 7

---

## ✅ 2. **Common Operations on Intervals**

### 🔁 Merge Overlapping Intervals
Goal: Combine intervals that overlap or are adjacent.

**Steps**:
1. Sort intervals by start time.
2. Iterate and merge if current interval overlaps with the last merged one.

```python
def merge(intervals):
    if not intervals:
        return []
    
    intervals.sort(key=lambda x: x[0])
    merged = [intervals[0]]

    for curr in intervals[1:]:
        prev = merged[-1]
        if curr[0] <= prev[1]:  # Overlap
            prev[1] = max(prev[1], curr[1])
        else:
            merged.append(curr)
    return merged
```

### 🧩 Insert New Interval
Insert a new interval into a list of non-overlapping intervals and merge if necessary.

```python
def insert(intervals, newInterval):
    res = []
    for i, interval in enumerate(intervals):
        if interval[1] < newInterval[0]:
            res.append(interval)
        elif interval[0] > newInterval[1]:
            res.append(newInterval)
            newInterval = interval
        else:
            newInterval[0] = min(newInterval[0], interval[0])
            newInterval[1] = max(newInterval[1], interval[1])
    res.append(newInterval)
    return res
```

### ❌ Erase Interval
Remove an interval or part of it from a list of non-overlapping intervals.

### 🔄 Non-overlapping Intervals
Goal: Remove minimum number of intervals so that the rest don't overlap.

**Approach**:
- Sort by end time.
- Greedily pick intervals that end earliest and don't overlap.

```python
def eraseOverlapIntervals(intervals):
    if not intervals:
        return 0
    
    intervals.sort(key=lambda x: x[1])
    count = 0
    end = float('-inf')

    for start, finish in intervals:
        if start >= end:
            end = finish
        else:
            count += 1
    return count
```

---

## ✅ 3. **Overlapping & Intersection Checks**

### 🧠 Check if Two Intervals Overlap
```python
def is_overlap(a, b):
    return not (a[1] < b[0] or b[1] < a[0])
```

### 🧮 Find Intersection of Two Intervals
```python
def intersection(a, b):
    start = max(a[0], b[0])
    end = min(a[1], b[1])
    if start <= end:
        return [start, end]
    return None
```

---

## ✅ 4. **Meeting Rooms / Scheduling Problems**

### 🚪 Meeting Rooms I – Can AttendMeetings
Check if any intervals overlap.

```python
def canAttendMeetings(intervals):
    intervals.sort()
    for i in range(1, len(intervals)):
        if intervals[i][0] < intervals[i-1][1]:
            return False
    return True
```

### 📅 Meeting Rooms II – Minimum Meeting Rooms Needed
Use a min-heap to track end times.

```python
import heapq

def minMeetingRooms(intervals):
    if not intervals:
        return 0

    intervals.sort(key=lambda x: x[0])
    free_rooms = []

    for start, end in intervals:
        if free_rooms and free_rooms[0] <= start:
            heapq.heappop(free_rooms)
        heapq.heappush(free_rooms, end)

    return len(free_rooms)
```

---

## ✅ 5. **Other Common Interval Problems**

| Problem Title                        | Description                                           |
|-------------------------------------|-------------------------------------------------------|
| **Merge Intervals**                 | Combine overlapping intervals                         |
| **Insert Interval**                 | Insert and merge into existing list                  |
| **Non-overlapping Intervals**       | Remove minimum to make intervals non-overlapping     |
| **Meeting Rooms I/II**              | Scheduling meetings                                   |
| **Find Right Interval**             | Find first interval whose start ≥ end of current     |
| **Range Module**                    | Track covered ranges dynamically                     |
| **Employee Free Time**              | Find common free intervals across employees          |
| **Partition Labels**                | Partition string into disjoint intervals             |

---

## ✅ 6. **Tips for Solving Interval Problems**

- **Sort intervals** by start or end time — usually helps!
- Use **greedy algorithms** when minimizing/maximizing something.
- For scheduling, think about **heap-based solutions**.
- Always consider **edge cases**: empty input, single interval, full overlap.
- Avoid modifying the input unless allowed — prefer making copies.

---

## ✅ 7. **Time Complexity Summary**

| Operation                            | Time Complexity         |
|-------------------------------------|--------------------------|
| Sorting intervals                   | O(n log n)               |
| Merging intervals                   | O(n log n)               |
| Inserting interval                  | O(n log n)               |
| Checking overlaps                   | O(n²) worst case         |
| Meeting Rooms II (Heap approach)    | O(n log n)               |
| Non-overlapping intervals (Greedy)  | O(n log n)               |

---

## ✅ 8. **Python Tips**

- Use `lambda` to sort by start or end:
  ```python
  intervals.sort(key=lambda x: x[0])
  ```
- Use `heapq` for meeting rooms and priority queues.
- Avoid slicing unless needed — use indices instead.
"""
#=====================================================================================

#56. Merge Intervals

class Solution:
    def merge(self, intervals: List[List[int]]) -> List[List[int]]:
        if not intervals:
            return []

    # Step 1: Sort intervals based on start time
        intervals.sort(key=lambda x: x[0])

        merged = [intervals[0]]

        for curr in intervals[1:]:
            last = merged[-1]
    
        # Step 2: If current interval overlaps with last, merge
            if curr[0] <= last[1]:
                last[1] = max(last[1], curr[1])
            else:
                merged.append(curr)

        return merged

#===============================================================================

#435. Non-overlapping Intervals

class Solution:
    def eraseOverlapIntervals(self, intervals: List[List[int]]) -> int:
        if not intervals:
            return 0

    # Step 1: Sort intervals by end time
        intervals.sort(key=lambda x: x[1])

    # Step 2: Initialize
        count = 0
        end = float('-inf')

        for start, curr_end in intervals:
            if start >= end:
            # No overlap, keep this interval
                end = curr_end
            else:
            # Overlap found, remove this interval
                count += 1

        return count  

#=================================================================================

#252: Metting Rooms:

from typing import List

# Definition of Interval
class Interval:
    def __init__(self, start: int, end: int):
        self.start = start
        self.end = end

class Solution:
    def canAttendMeetings(self, intervals: List[Interval]) -> bool:
        # Sort intervals based on their start times
        intervals.sort(key=lambda x: x.start)

        # Check for overlap between consecutive intervals
        for i in range(1, len(intervals)):
            if intervals[i].start < intervals[i - 1].end:
                return False
        return True


#===============================================================================

#LeetCode 253: Meeting Rooms II

import heapq
from typing import List

class Interval:
    def __init__(self, start: int, end: int):
        self.start = start
        self.end = end

class Solution:
    def minMeetingRooms(self, intervals: List[Interval]) -> int:
        if not intervals:
            return 0

        # Step 1: Sort intervals based on start time
        intervals.sort(key=lambda x: x.start)

        # Step 2: Use a min-heap to keep track of end times
        min_heap = []

        for interval in intervals:
            # Free up a room if possible
            if min_heap and interval.start >= min_heap[0]:
                heapq.heappop(min_heap)
            # Allocate a new room
            heapq.heappush(min_heap, interval.end)

        # Size of heap is the number of rooms needed
        return len(min_heap)
    
#======================================================================================

#1851. Minimum Interval to Include Each Query

from typing import List
import heapq

class Solution:
    def minInterval(self, intervals: List[List[int]], queries: List[int]) -> List[int]:
        # Sort intervals by start time
        intervals.sort()
        
        # Store queries with original indices
        indexed_queries = sorted((q, i) for i, q in enumerate(queries))
        
        result = [0] * len(queries)
        min_heap = []
        i = 0  # Pointer to intervals
        
        for query, idx in indexed_queries:
            # Add all intervals that could include this query
            while i < len(intervals) and intervals[i][0] <= query:
                l, r = intervals[i]
                heapq.heappush(min_heap, (r - l + 1, r))
                i += 1
            
            # Remove intervals from heap that don't include query
            while min_heap and min_heap[0][1] < query:
                heapq.heappop(min_heap)
            
            # Top of heap is the smallest valid interval
            if min_heap:
                result[idx] = min_heap[0][0]
            else:
                result[idx] = -1
        
        return result

