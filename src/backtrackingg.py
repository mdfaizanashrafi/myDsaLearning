"""
# 🧠 Backtracking Revision Notes

> A complete guide to backtracking with analogies, diagrams, and code examples.

---

## 🧩 What is Backtracking?

Backtracking is a **recursive algorithm** for solving problems by trying all possible choices and **backing up (backtracking)** when a choice doesn’t lead to a solution.

### 🔁 It’s like:
- Solving a maze: try one path → hit dead end → go back and try another
- Picking clothes: try outfit A → doesn’t work → try outfit B
- Choosing ice cream flavors: try vanilla → not good → try chocolate

You **try something**, and if it doesn’t work out, you **go back** and try again.

---

## 🔍 Where Is It Used?

| Problem Type | Example Problems |
|--------------|------------------|
| Maze Solving | Word Search, Sudoku |
| Subset Generation | Subsets, Combinations |
| Permutation Generation | Permutations |
| Constraint Satisfaction | N-Queens, Sudoku |

---

## 🧱 General Backtracking Template

```python
def backtrack(choices, current_path):
    if base_case:
        save_result()
        return
    for choice in choices:
        make_choice(choice)
        backtrack(remaining_choices, new_path)
        unmake_choice(choice)  # Backtrack
```

---

## 🌲 Tree Structure of Backtracking

All backtracking can be visualized as a **tree of decisions**.

Each node = a decision  
Each branch = a possible next choice  
Leaves = full solutions or dead ends

### 📊 Example: Subset Tree (for [1,2,3])

```
                  []
           /      |      \
         [1]     [2]     [3]
       /   \        \
    [1,2] [1,3]    [2,3]
     /
[1,2,3]
```

Each level adds a new number. You stop at the leaves.

---

## 🧒 Simple Analogy Recap

| Real-Life Analogy | Backtracking Meaning |
|------------------|---------------------|
| Lost in a maze | Try paths until you find the exit |
| Choosing outfits | Try one → doesn’t fit → try another |
| Solving a puzzle | Try a piece → wrong → put back and try another |
| Party planning | Seat guest A → conflict → move to seat B |

---

## 🎯 Common Backtracking Problems

| Problem | Key Idea | Code Pattern |
|--------|----------|--------------|
| **Subsets** | Include or exclude each item | `path.append(i)` then `path.pop()` |
| **Combinations** | Choose k items from n | Only pick forward items |
| **Permutations** | Rearranging all items | Pick any unused item |
| **N-Queens** | Place queens safely | Check row/col/diagonal |
| **Sudoku Solver** | Fill board without conflicts | Try numbers 1–9 |
| **Word Search** | Find word in grid | DFS + backtracking on grid |

---

## 🧩 Step-by-Step Backtracking Framework

Use this structure for any backtracking problem:

```python
result = []

def backtrack(start=0, path=[]):
    if meets_condition:
        result.append(path[:])
        return
    for i in range(start, len(choices)):
        path.append(choices[i])
        backtrack(i + 1, path)
        path.pop()

backtrack()
return result
```

Depending on the problem, change:
- When to stop (`if meets_condition`)
- What choices are available
- How to track used choices
- Whether to allow duplicates or not

---

## 🧮 Comparison Table

| Feature | Subsets | Combinations | Permutations | N-Queens |
|--------|---------|--------------|---------------|-----------|
| Order Matters? | ❌ | ❌ | ✅ | ❌ |
| All Elements Used? | ❌ | ✅ | ✅ | ✅ |
| Reuse Allowed? | ❌ | ❌ | ❌ | ❌ |
| Use Cases | Generate all subsets | Choose k elements | Rearrange all | Safe queen placement |
| Time Complexity | O(2ⁿ) | O(n choose k) | O(n!) | O(n!) |

---

## 🧪 Code Snippets

### 1. **Subsets**

```python
def subsets(nums):
    result = []

    def dfs(start, path):
        result.append(path[:])
        for i in range(start, len(nums)):
            path.append(nums[i])
            dfs(i + 1, path)
            path.pop()

    dfs(0, [])
    return result
```

---

### 2. **Combinations**

```python
def combine(n, k):
    result = []

    def dfs(start, path):
        if len(path) == k:
            result.append(path[:])
            return
        for i in range(start, n + 1):
            path.append(i)
            dfs(i + 1, path)
            path.pop()

    dfs(1, [])
    return result
```

---

### 3. **Permutations**

```python
def permute(nums):
    result = []

    def dfs(path, remaining):
        if not remaining:
            result.append(path[:])
            return
        for i in range(len(remaining)):
            dfs(path + [remaining[i]], remaining[:i] + remaining[i+1:])

    dfs([], nums)
    return result
```

---

### 4. **N-Queens**

```python
def solveNQueens(n):
    result = []
    cols = set()
    pos_diag = set()  # row + col
    neg_diag = set()  # row - col
    state = []

    def dfs(row):
        if row == n:
            result.append(["." * r + "Q" + "." * (n - r - 1) for r in state])
            return
        for col in range(n):
            if col in cols or (row + col) in pos_diag or (row - col) in neg_diag:
                continue
            cols.add(col)
            pos_diag.add(row + col)
            neg_diag.add(row - col)
            state.append(col)

            dfs(row + 1)

            cols.remove(col)
            pos_diag.remove(row + col)
            neg_diag.remove(row - col)
            state.pop()

    dfs(0)
    return result
```

---

## 🧭 Visualization Summary

| Problem | Tree Shape |
|--------|------------|
| Subsets | Binary tree (include/exclude) |
| Combinations | Narrowing branches |
| Permutations | Full branching every time |
| N-Queens | Pruned by rules |

---

## 💡 Tips for Mastering Backtracking

✅ Understand the **base case**  
✅ Know what to **pass into recursive call**  
✅ Track **used choices** correctly  
✅ **Prune early** if invalid  
✅ **Restore state** after recursion returns  
✅ Practice building **trees manually**  
✅ Use print statements or debuggers to see how it explores

---

## 📝 Quick Revision Checklist

✅ Can you generate all subsets of `[1,2,3]` by hand?  
✅ Can you explain why permutations have more results than combinations?  
✅ Can you write basic backtracking template from memory?  
✅ Can you explain what happens in N-Queens step-by-step?  
✅ Do you know when to pop or remove choices?  
✅ Do you understand how to prune invalid paths?

"""
#==================================================================================================

#Subsets: Leetcode 78

class Solution:
    def subsets(self, nums: List[int]) -> List[List[int]]:
        result = []
        def backtrack(start, path):
            result.append(path[:])
            for i in range(start, len(nums)):
                path.append(nums[i])
                backtrack(i+1, path)
                path.pop()
        
        backtrack(0,[])
        return result
    
#===========================================================================================
#39. Combination Sum

class Solution:
    def combinationSum(self, candidates: List[int], target: int) -> List[List[int]]:
        result = []
        def backtrack(start, path, total):
            if total == target:
                result.append(path[:])
                return

            if total>target:
                return
            for i in range(start, len(candidates)):
                num = candidates[i]
                path.append(num)
                backtrack(i, path, total+num)
                path.pop()
        
        backtrack(0,[],0)
        return result 

#=================================================================================

