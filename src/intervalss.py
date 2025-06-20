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
#============================================================================

