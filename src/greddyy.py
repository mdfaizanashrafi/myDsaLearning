"""
# 🧠 Greedy Algorithms Revision Notes  

## ✅ 1. **What is a Greedy Algorithm?**

- Makes the **best possible choice at each step**
- No backtracking or reconsideration
- Often used in **optimization problems**

### 🔍 When to Use:
- Optimal substructure
- Greedy-choice property holds: a globally optimal solution can be arrived at by making a locally optimal (greedy) choice.

---

## ✅ 2. **Key Concepts**

| Concept                | Description                                                                 |
|------------------------|-----------------------------------------------------------------------------|
| Greedy-choice property | A local optimum leads to global optimum                                     |
| Optimal substructure   | An optimal solution to the problem contains optimal solutions to subproblems |
| No backtracking        | Once a decision is made, it cannot be undone                               |

---

## ✅ 3. **Common Greedy Problems & Solutions**

### 🔺 **KADANE'S ALGORITHM** – Maximum Subarray Sum  
**Problem**: Given an integer array `nums`, find the contiguous subarray (containing at least one number) which has the **largest sum**, and return its sum.

#### 💡 Idea:
At each index, decide whether to:
- Start a new subarray at current element, or
- Continue the previous subarray

This is greedy because we always choose the best option at each step.

#### 🧑‍💻 Python Code:

```python
def maxSubArray(nums):
    max_current = max_global = nums[0]
    for num in nums[1:]:
        max_current = max(num, max_current + num)
        max_global = max(max_global, max_current)
    return max_global
```

#### 📈 Example:

```python
Input: [-2,1,-3,4,-1,2,1,-5,4]
Output: 6
Explanation: [4,-1,2,1] has the largest sum = 6.
```

#### ⏱️ Time Complexity:
- **O(n)** – single pass through the array

#### 💾 Space Complexity:
- **O(1)** – constant space

---

### 🪙 Coin Change (if coins are standard like in US currency)
```python
def coinChange(coins, amount):
    coins.sort(reverse=True)
    count = 0
    for coin in coins:
        if amount == 0:
            break
        count += amount // coin
        amount %= coin
    return count if amount == 0 else -1
```

### 📅 Activity Selection / Meeting Rooms II
Select maximum number of non-overlapping intervals:

```python
def activitySelection(intervals):
    intervals.sort(key=lambda x: x[1])
    count = 0
    end = -1
    for start, finish in intervals:
        if start >= end:
            count += 1
            end = finish
    return count
```

### 🚗 Jump Game
Can you jump from index `0` to last?

```python
def canJump(nums):
    farthest = 0
    for i in range(len(nums)):
        if i > farthest:
            return False
        farthest = max(farthest, i + nums[i])
    return True
```

### 💰 Jump Game II (Minimum Jumps)
```python
def jump(nums):
    jumps = 0
    farthest = 0
    current_end = 0
    for i in range(len(nums) - 1):
        farthest = max(farthest, i + nums[i])
        if i == current_end:
            jumps += 1
            current_end = farthest
    return jumps
```

### 🧊 Gas Station Circular Route
Find starting gas station index to complete circuit:

```python
def canCompleteCircuit(gas, cost):
    total_gas = 0
    curr_gas = 0
    start = 0
    for i in range(len(gas)):
        diff = gas[i] - cost[i]
        total_gas += diff
        curr_gas += diff
        if curr_gas < 0:
            curr_gas = 0
            start = i + 1
    return start if total_gas >= 0 else -1
```

---

## ✅ 4. **Huffman Coding (Greedy for Compression)**

- Build a binary tree where more frequent characters get shorter codes.
- Uses **min-heap** to select two least frequent nodes repeatedly.

---

## ✅ 5. **Fractional Knapsack Problem**

You can take fractions of items (unlike 0/1 knapsack):

```python
def fractionalKnapsack(items, capacity):
    # items = [(value, weight), ...]
    items.sort(key=lambda x: x[0]/x[1], reverse=True)
    total_value = 0
    for value, weight in items:
        if capacity == 0:
            break
        amount = min(weight, capacity)
        total_value += amount * (value / weight)
        capacity -= amount
    return total_value
```

---

## ✅ 6. **Greedy vs Dynamic Programming**

| Feature                  | Greedy                          | Dynamic Programming                   |
|--------------------------|----------------------------------|---------------------------------------|
| Choice made              | Locally optimal                  | Considers all possibilities           |
| Time complexity          | Usually faster (O(n log n))      | Slower (O(n²) or more)                |
| Guarantee of optimality  | Only sometimes                   | Always (if applied correctly)         |
| Backtracking             | ❌ Not done                      | ✅ Done in some approaches             |
| Examples                 | Activity selection, Huffman     | 0/1 Knapsack, Longest Common Subsequence |

---

## ✅ 7. **When Greedy Fails**

- **0/1 Knapsack** – taking the most valuable item may not lead to optimal solution.
- **Shortest Path in Graphs with negative edges** – Dijkstra’s fails; use Bellman-Ford instead.
- **Making change with arbitrary coin denominations** – may need DP.

---

## ✅ 8. **Tips for Solving Greedy Problems**

- Think about what **local optimal choice** would look like.
- Try small examples to verify if greedy approach works.
- Prove correctness mathematically if possible.
- If greedy doesn’t work, consider **Dynamic Programming** or **Backtracking**.

---

## ✅ 9. **Time Complexity Summary**

| Problem Type                     | Time Complexity     |
|----------------------------------|----------------------|
| Sorting intervals                | O(n log n)           |
| Merging intervals                | O(n log n)           |
| Activity selection               | O(n log n)           |
| Jump Game                        | O(n)                 |
| Jump Game II                     | O(n)                 |
| Gas Station                      | O(n)                 |
| Fractional Knapsack              | O(n log n)           |
| Huffman Coding                   | O(n log n)           |
| Kadane’s Algorithm               | O(n)                 |

---

## ✅ 10. **Real-world Applications**

- **Scheduling jobs** (CPU scheduling, meeting rooms)
- **Compression algorithms** (Huffman coding)
- **Networking** (Dijkstra's shortest path)
- **Optimization in finance** (portfolio selection)
- **Resource allocation**
"""