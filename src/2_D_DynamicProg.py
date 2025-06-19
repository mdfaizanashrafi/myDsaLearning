
"""
# 🧠 2D Dynamic Programming in Python – Full Revision Notes

---

## 📌 What is 2D Dynamic Programming?

**2D DP** uses a **2D table (matrix)** to solve problems where results depend on **two parameters** (e.g., `i` and `j` — like index, length, capacity, rows/columns, etc.).

Used in problems involving:

* Strings (substrings, subsequences)
* Grids (movement, path)
* Knapsack variants
* Matrix operations

---

## 📦 Syntax

```python
dp = [[initial_value] * (cols + 1) for _ in range(rows + 1)]

# Example update
for i in range(1, rows+1):
    for j in range(1, cols+1):
        dp[i][j] = recurrence_relation
```

---

## 🧩 Use Cases of 2D DP

| Use Case Category          | Real-World Example                              |
| -------------------------- | ----------------------------------------------- |
| Text Comparison            | Spellcheck, versioning (edit distance)          |
| Grid-Based Navigation      | Robot movement, map traversal                   |
| Memory-Driven Optimization | Multiple-choice scheduling, time-table planning |
| Bioinformatics             | DNA alignment (LCS)                             |
| E-Commerce                 | Package pricing (Knapsack)                      |
| Gaming/AI                  | Grid pathfinding                                |

---

## 🔣 Types of 2D DP Problems

### 1. **Grid DP**

```python
dp[i][j] = dp[i-1][j] + dp[i][j-1]
```

🟢 E.g. Unique paths, min path sum

---

### 2. **String Matching**

```python
dp[i][j] = ... based on s1[i-1], s2[j-1]
```

🟢 E.g. Edit distance, Longest Common Subsequence (LCS), Palindromic Substrings

---

### 3. **Knapsack Variants**

```python
dp[i][w] = max(dp[i-1][w], dp[i-1][w - wt[i-1]] + val[i-1])
```

🟢 E.g. 0/1 Knapsack

---

### 4. **Partitioning / Palindromes**

```python
dp[i][j] = True if s[i:j+1] is palindrome and dp[i+1][j-1]
```

🟢 E.g. Palindrome partitioning, longest palindromic substring

---

## 🔁 2D DP Table Setup

```python
# Example: Grid Path Sum
dp = [[0] * n for _ in range(m)]
dp[0][0] = grid[0][0]
for i in range(m):
    for j in range(n):
        if i > 0: dp[i][j] = min(dp[i][j], dp[i-1][j] + grid[i][j])
        if j > 0: dp[i][j] = min(dp[i][j], dp[i][j-1] + grid[i][j])
```

---

## 🔥 Examples

---

### 1. **Unique Paths (Grid DP)**

```python
def uniquePaths(m, n):
    dp = [[1]*n for _ in range(m)]
    for i in range(1, m):
        for j in range(1, n):
            dp[i][j] = dp[i-1][j] + dp[i][j-1]
    return dp[m-1][n-1]
```

---

### 2. **Minimum Path Sum (Grid DP)**

```python
def minPathSum(grid):
    m, n = len(grid), len(grid[0])
    dp = [[0]*n for _ in range(m)]
    dp[0][0] = grid[0][0]
    for i in range(m):
        for j in range(n):
            if i > 0:
                dp[i][j] = grid[i][j] + dp[i-1][j]
            if j > 0:
                val = grid[i][j] + dp[i][j-1]
                dp[i][j] = min(dp[i][j], val) if i > 0 else val
    return dp[m-1][n-1]
```

---

### 3. **Longest Common Subsequence (LCS)**

```python
def longestCommonSubsequence(text1, text2):
    m, n = len(text1), len(text2)
    dp = [[0]*(n+1) for _ in range(m+1)]
    for i in range(1, m+1):
        for j in range(1, n+1):
            if text1[i-1] == text2[j-1]:
                dp[i][j] = 1 + dp[i-1][j-1]
            else:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])
    return dp[m][n]
```

---

### 4. **Edit Distance (Levenshtein Distance)**

```python
def minDistance(word1, word2):
    m, n = len(word1), len(word2)
    dp = [[0]*(n+1) for _ in range(m+1)]
    for i in range(m+1):
        for j in range(n+1):
            if i == 0: dp[i][j] = j
            elif j == 0: dp[i][j] = i
            elif word1[i-1] == word2[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = 1 + min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])
    return dp[m][n]
```

---

### 5. **0/1 Knapsack**

```python
def knapsack(weights, values, capacity):
    n = len(values)
    dp = [[0] * (capacity + 1) for _ in range(n + 1)]
    for i in range(1, n+1):
        for w in range(capacity + 1):
            if weights[i-1] <= w:
                dp[i][w] = max(dp[i-1][w], values[i-1] + dp[i-1][w - weights[i-1]])
            else:
                dp[i][w] = dp[i-1][w]
    return dp[n][capacity]
```

---

## 🧠 Optimization Tip

* **Space optimization to 1D** is possible if you're only using the **previous row**, like in knapsack or LCS.
* Use two rows (`dp[0], dp[1]`) and toggle between them with `i % 2`.

---

## 🧰 Quick Interview Tips

✅ Clearly define:

* `dp[i][j]`: What does it represent?
* Base cases: Often `dp[0][*]` or `dp[*][0]`
* Recurrence relation

✅ Use padding (`dp[i+1][j+1]`) to avoid index errors.

✅ In string problems, compare `s1[i-1]` and `s2[j-1]`, not `i, j`.

✅ Always test edge cases: empty string, 1x1 grid, capacity 0, etc.

---

## 🧪 Practice Problems

| Problem Name                  | Platform              |
| ----------------------------- | --------------------- |
| Unique Paths                  | Leetcode 62           |
| Minimum Path Sum              | Leetcode 64           |
| Longest Common Subsequence    | Leetcode 1143         |
| Edit Distance                 | Leetcode 72           |
| 0/1 Knapsack                  | InterviewBit / Custom |
| Longest Palindromic Substring | Leetcode 5            |
| Distinct Subsequences         | Leetcode 115          |
| Interleaving String           | Leetcode 97           |
| Maximal Square                | Leetcode 221          |
"""
"""
## 🎒 1. Knapsack Problem

### 🧠 Concept:

You have:

* A **bag (knapsack)** with a maximum **weight capacity**.
* A set of **items**, each with a **weight** and a **value**.
* Goal: **Maximize the total value** of items in the bag **without exceeding the weight limit**.

### 🧰 Real-Life Analogy:

You're going hiking with a backpack that can carry **10 kg**. You have several items (food, tent, clothes), each with weight and usefulness (value). You want to **pack the most useful combination** without overloading.

---

### 💡 Variants:

* **0/1 Knapsack**: You either **take or skip** an item (can't break items).
* **Fractional Knapsack**: You can **take a part** of an item (used in greedy, not DP).
* **Unbounded Knapsack**: You can **take an item multiple times**.

---

### 🧾 Example (0/1 Knapsack):

| Item | Weight | Value |
| ---- | ------ | ----- |
| A    | 1      | 1     |
| B    | 3      | 4     |
| C    | 4      | 5     |

* Bag capacity = 4
* Best choice: B (weight 3, value 4) + A (weight 1, value 1)
  ✅ Total = 5
  ❌ Can’t pick C alone (5) because it weighs 4, leaving no room for more value.

---

## 🔤 2. Longest Common Subsequence (LCS)

### 🧠 Concept:

Given two strings, find the **longest sequence of characters that appear in both**, **in the same order**, **but not necessarily contiguous**.

### 🧰 Real-Life Analogy:

Compare two versions of a document and highlight the longest **sequence of matching phrases** in both — not necessarily word-by-word.

---

### 💡 Use Cases:

* DNA sequence alignment
* File diff tools (like Git)
* Spell-checkers or plagiarism checkers

---

### 🧾 Example:

```text
text1 = "abcde"
text2 = "ace"
```

✅ LCS = `"ace"` (length 3)

Not contiguous in "abcde", but same order.

---

## ✍️ 3. Levenshtein Distance (Edit Distance)

### 🧠 Concept:

The **minimum number of operations** required to **convert one string into another**, using:

1. **Insert**
2. **Delete**
3. **Replace**

### 🧰 Real-Life Analogy:

Spell checker trying to auto-correct `hte` to `the` with **minimum changes**.

---

### 💡 Use Cases:

* Auto-correct and spell-check
* DNA mutation analysis
* Natural language processing (NLP)
* Typo detection in search engines

---

### 🧾 Example:

```text
word1 = "kitten"
word2 = "sitting"
```

✅ Minimum operations: **3**

* kitten → sitten (replace ‘k’ with ‘s’)
* sitten → sittin (replace ‘e’ with ‘i’)
* sittin → sitting (insert ‘g’)

---

## 🔁 Quick Comparison

| Metric         | Knapsack                     | LCS                           | Edit Distance (Levenshtein)      |
| -------------- | ---------------------------- | ----------------------------- | -------------------------------- |
| Based on       | Weights & Values             | Matching character sequence   | String transformation operations |
| Type           | Optimization problem         | Matching pattern              | Transformation / correction      |
| Returns        | Max value (int)              | Longest subsequence (length)  | Minimum number of edits (int)    |
| Real-world Use | Packing, resource allocation | Document diff, DNA similarity | Spell correction, typo detection |
"""

#===========================================================================================

#62. Unique Paths

class Solution:
    def uniquePaths(self, m: int, n: int) -> int:
        dp = [1]*n
        for _ in range(1,m):
            for j in range(1,n):
                dp[j] += dp[j-1]

        return dp[-1]
