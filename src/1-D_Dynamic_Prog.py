"""
# 🧠 1D Dynamic Programming in Python – Full Revision Notes

---

## 📌 What is 1D Dynamic Programming?

**Dynamic Programming (DP)** is an optimization technique used to **break problems into overlapping subproblems**, solve each once, and **store the result**.

**1D DP** uses a **one-dimensional array** (list) to store intermediate results.

---

## 📦 Syntax

```python
# Generic syntax
dp = [initial_value] * (n + 1)
dp[base_case_index] = base_case_value

for i in range(start, n+1):
    dp[i] = recurrence_relation
return dp[n]
```

---

## 🧩 Use Cases of 1D DP

| Use Case Category         | Real-World Example                              |
| ------------------------- | ----------------------------------------------- |
| Combinatorics             | Number of ways to reach a step, climb stairs    |
| Optimization              | Max/min profit, cost, reward                    |
| Subset Problems           | Partition, subset sum, knapsack                 |
| Finance & Budgeting       | Max investment returns without adjacent years   |
| Robotics & Gaming         | Ways to reach a target using certain moves      |
| Memory-Efficient Planning | Reducing space in tabulated problems            |
| Competitive Programming   | Problems with 1 parameter (e.g., index, weight) |

---

## 🔣 Types of 1D DP Problems

### 1. **Fibonacci-Type (Linear Recurrence)**

```python
dp[i] = dp[i-1] + dp[i-2]
```

🟢 E.g. Climbing stairs, ways to tile a board, Fibonacci number.

---

### 2. **Optimization (Max/Min Value)**

```python
dp[i] = max(dp[i-1], dp[i-2] + nums[i])
```

🟢 E.g. House Robber, Max Sum Without Adjacent Elements.

---

### 3. **Subset-Based (True/False)**

```python
dp[i] = dp[i] or dp[i - num]
```

🟢 E.g. Subset sum, Partition equal subset sum.

---

### 4. **Knapsack-Type (Capacity-Based)**

```python
dp[w] = max(dp[w], dp[w - wt[i]] + val[i])
```

🟢 E.g. 0/1 Knapsack (space optimized version).

---

### 5. **Counting Problems (Number of Ways)**

```python
dp[i] += dp[i - coin]
```

🟢 E.g. Coin Change (number of combinations).

---

## 🔁 DP Table Pattern

```python
# Example: Climbing Stairs
dp = [0] * (n+1)
dp[0], dp[1] = 1, 1
for i in range(2, n+1):
    dp[i] = dp[i-1] + dp[i-2]
return dp[n]
```

---

## 🧠 Optimization: Constant Space

```python
a, b = 0, 1
for _ in range(n):
    a, b = b, a + b
return b
```

---

## 🔥 Examples

---

### 1. **Fibonacci Number**

```python
def fib(n):
    if n <= 1: return n
    dp = [0, 1]
    for i in range(2, n+1):
        dp.append(dp[i-1] + dp[i-2])
    return dp[n]
```

---

### 2. **Climbing Stairs**

```python
def climbStairs(n):
    dp = [0] * (n+1)
    dp[0], dp[1] = 1, 1
    for i in range(2, n+1):
        dp[i] = dp[i-1] + dp[i-2]
    return dp[n]
```

---

### 3. **House Robber**

```python
def rob(nums):
    if not nums: return 0
    if len(nums) <= 2: return max(nums)
    dp = [0] * len(nums)
    dp[0], dp[1] = nums[0], max(nums[0], nums[1])
    for i in range(2, len(nums)):
        dp[i] = max(dp[i-1], dp[i-2] + nums[i])
    return dp[-1]
```

---

### 4. **Min Cost Climbing Stairs**

```python
def minCostClimbingStairs(cost):
    n = len(cost)
    dp = [0] * n
    dp[0], dp[1] = cost[0], cost[1]
    for i in range(2, n):
        dp[i] = cost[i] + min(dp[i-1], dp[i-2])
    return min(dp[-1], dp[-2])
```

---

### 5. **Partition Equal Subset Sum**

```python
def canPartition(nums):
    total = sum(nums)
    if total % 2 != 0: return False
    target = total // 2
    dp = [False] * (target + 1)
    dp[0] = True
    for num in nums:
        for i in range(target, num - 1, -1):
            dp[i] = dp[i] or dp[i - num]
    return dp[target]
```

---

## 📎 Interview Tips

✅ Always define:

* Problem type (counting? optimization? decision?)
* State (What does `dp[i]` mean?)
* Base case (Initialize carefully!)
* Recurrence relation

✅ If a problem asks for:

* Max/min → Think `max()/min()` in recurrence
* True/False → Think `dp[i] = dp[i] or dp[i - num]`
* Count → Use `+=` to accumulate ways

✅ Optimize Space:

* If `dp[i]` only depends on `dp[i-1]`, `dp[i-2]`, reduce to 2 variables.

---

## 🧪 Practice Problems

| Problem Name               | Platform     |
| -------------------------- | ------------ |
| Fibonacci Number           | Leetcode 509 |
| Climbing Stairs            | Leetcode 70  |
| House Robber               | Leetcode 198 |
| Min Cost Climbing Stairs   | Leetcode 746 |
| Partition Equal Subset Sum | Leetcode 416 |
| Coin Change                | Leetcode 322 |
| Maximum Subarray           | Leetcode 53  |
| Delete and Earn            | Leetcode 740 |
| Jump Game II               | Leetcode 45  |
"""
#=========================================================================================

#70. Climbing Stairs

class Solution:
    def climbStairs(self, n: int) -> int:
        if n == 1:
            return 1
        if n == 2:
            return 2

        a, b = 1, 2
        for _ in range(3, n + 1):
            a, b = b, a + b

        return b

#=========================================================================

#746. Min Cost Climbing Stairs

class Solution:
    def minCostClimbingStairs(self, cost: List[int]) -> int:
        n = len(cost)
        if n==0:
            return 0
        if n== 1:
            return cost[0]
        
        dp = [0]*n
        dp[0] = cost[0]
        dp[1] = cost[1]

        for i in range(2,n):
            dp[i] = cost[i]+min(dp[i-1],dp[i-2])

        return min(dp[-1],dp[-2])

#============================================================================

#198. House Robber

class Solution:
    def rob(self, nums: List[int]) -> int:
        n = len(nums)
        if n==0:
            return 0
        if n==1:
            return nums[0]

        dp = [0]*n
        dp[0],dp[1] = nums[0], max(nums[0],nums[1])

        for i in range(2,n):
            dp[i]=max(dp[i-1],dp[i-2]+nums[i])
        
        return dp[-1]

#============================================================================

#213. House Robber II

class Solution:
    def rob(self, nums: List[int]) -> int:
        if len(nums) == 1:
            return nums[0]

        if len(nums)==2:
            return max(nums)

        def rob_linear(houses):
            prev, curr = 0,0

            for num in houses:
                temp = curr
                curr = max(curr, prev+num)
                prev = temp

            return curr

        return max(rob_linear(nums[:-1]), rob_linear(nums[1:]))

#=====================================================================================

#5. Longest Palindromic Substring

class Solution:
    def longestPalindrome(self, s: str) -> str:
        n = len(s)
        dp = [[False]*n for _ in range(n)]
        longest = ""

        for end in range(n):
            for start in range(end+1):
                if start == end:
                    dp[start][end] = True

                elif start +1 == end:
                    dp[start][end] = (s[start]==s[end])
                
                else:
                    dp[start][end] = (s[start]==s[end] and dp[start+1][end-1])
                
                if dp[start][end]:
                    if end-start+1 > len(longest):
                        longest = s[start:end+1]
                    
        return longest

