"""
# 📘 Data Structure Deep Dive: Trie (Prefix Tree)

---

## 🌱 What is a Trie?

A **Trie** (pronounced "try") is a **tree-based data structure** used for storing and searching associative arrays where the keys are strings.

Also known as:

* **Prefix Tree**
* **Radix Tree** (variant)
* **Digital Tree**

### 🔍 Use Case

Efficiently store and search for words or prefixes in a collection of strings.

---

## 🧠 Think Like a Dictionary

Imagine using a dictionary to find all words starting with `app`.
A Trie makes this **prefix-based search** fast and efficient by organizing characters as branches of a tree.

---

## 📚 Basic Trie Structure

Each node represents a character. A path from the root to a marked node forms a valid word.

### 🧾 Example

Words to insert: `"apple"`, `"app"`, `"apricot"`, `"banana"`

```
(root)
 |
 a
 |
 p
 |
 p
 / \
l   r
|   |
e   i
    |
    c
    |
    o
    |
    t
        \
         b
          \
           a
            \
             n
              \
               a
                \
                 n
                  \
                   a
```

* `"app"` ends at second `p`
* `"apple"` ends at `e`
* `"apricot"` ends at `t`
* `"banana"` ends at final `a`

---

## 🧩 Key Concepts

| Term       | Description                             |
| ---------- | --------------------------------------- |
| Root       | Starting point (empty node)             |
| Node       | Represents one character                |
| Edge       | Connection between nodes                |
| Word       | Sequence of characters ending at marker |
| End Marker | Marks the end of a complete word        |

---

## ✅ Why Use a Trie?

| Benefit                        | Description                                                 |
| ------------------------------ | ----------------------------------------------------------- |
| Fast Prefix Search             | Find words with prefix in **O(k)** time (k = prefix length) |
| Space Efficient (Prefix Share) | Common prefixes are stored only once                        |
| Autocomplete Suggestions       | Ideal for real-time suggestions                             |
| Spell Checking                 | Check if a word exists efficiently                          |

---

## ⏱️ Time Complexity

| Operation     | Time Complexity |
| ------------- | --------------- |
| Insert        | O(L)            |
| Search        | O(L)            |
| Prefix Search | O(L)            |
| Delete        | O(L)            |

Where `L` is the length of the word.

---

## 🧱 Implementing Trie in Python

### 📌 Step 1: Define Trie Node

```python
class TrieNode:
    def __init__(self):
        self.children = {}
        self.is_end_of_word = False
```

---

### 📌 Step 2: Define Trie Structure

```python
class Trie:
    def __init__(self):
        self.root = TrieNode()

    def insert(self, word: str) -> None:
        node = self.root
        for char in word:
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]
        node.is_end_of_word = True

    def search(self, word: str) -> bool:
        node = self.root
        for char in word:
            if char not in node.children:
                return False
            node = node.children[char]
        return node.is_end_of_word

    def startsWith(self, prefix: str) -> bool:
        node = self.root
        for char in prefix:
            if char not in node.children:
                return False
            node = node.children[char]
        return True
```

---

### 🧪 Test It

```python
trie = Trie()
trie.insert("apple")
print(trie.search("apple"))   # True
print(trie.search("app"))     # False
print(trie.startsWith("app")) # True
trie.insert("app")
print(trie.search("app"))     # True
```

✅ **Output:**

```
True
False
True
True
```

---

## 🔄 Advanced Features

### 🔎 1. Autocomplete / Suggest Words

```python
def suggest_words(node, prefix, results):
    if node.is_end_of_word:
        results.append(prefix)
    for char, child in node.children.items():
        suggest_words(child, prefix + char, results)

def get_suggestions(self, prefix):
    node = self.root
    for char in prefix:
        if char not in node.children:
            return []
        node = node.children[char]
    results = []
    suggest_words(node, prefix, results)
    return results
```

---

### 🗑️ 2. Delete a Word

```python
def delete(self, word: str) -> None:
    def helper(node, word, depth):
        if depth == len(word):
            if node.is_end_of_word:
                node.is_end_of_word = False
            return len(node.children) == 0
        char = word[depth]
        if char not in node.children:
            return False
        should_delete = helper(node.children[char], word, depth + 1)
        if should_delete:
            del node.children[char]
            return len(node.children) == 0
        return False
    helper(self.root, word, 0)
```

---

### 🧬 3. Search with Wildcards (Pattern Matching)

Supports search like `"a.c"`:

```python
def search_with_dot(self, word: str) -> bool:
    def dfs(node, index):
        if index == len(word):
            return node.is_end_of_word
        char = word[index]
        if char == '.':
            for child in node.children.values():
                if dfs(child, index + 1):
                    return True
            return False
        else:
            if char not in node.children:
                return False
            return dfs(node.children[char], index + 1)
    return dfs(self.root, 0)
```

---

## 🌐 Real-World Applications

| Application     | Usage                                  |
| --------------- | -------------------------------------- |
| Autocomplete    | Suggest words while typing             |
| Spell Checker   | Validate if a word exists              |
| IP Routing      | Match longest prefix in routing tables |
| T9 Predictive   | Predict based on numeric input         |
| Boggle Solver   | Efficient word search in grids         |
| Browser History | Efficient prefix-based search          |

---

## 🔍 Variants of Trie

| Variant                 | Description                                             |
| ----------------------- | ------------------------------------------------------- |
| **Radix Tree**          | Compresses common prefixes into a single edge           |
| **Suffix Tree**         | Builds all suffixes; used in string matching            |
| **Ternary Search Trie** | Hybrid between binary search and trie (memory friendly) |
| **Compressed Trie**     | Skips single-child nodes for space saving               |


"""
#===========================================================================================================

#208. Implement Trie (Prefix Tree)

class TrieNode:
    def __init__(self):
        self.children = {}
        self.is_end_of_word = False

class Trie:

    def __init__(self):
        self.root = TrieNode()

    def insert(self, word: str) -> None:
        node = self.root
        for char in word:
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]
        node.is_end_of_word = True

    def search(self, word: str) -> bool:
        node= self.root
        for char in word:
            if char not in node.children:
                return False

            node = node.children[char]
        return node.is_end_of_word

    def startsWith(self, prefix: str) -> bool:
        node = self.root
        for char in prefix:
            if char not in node.children:
                return False

            node = node.children[char]
        return True


# Your Trie object will be instantiated and called as such:
# obj = Trie()
# obj.insert(word)
# param_2 = obj.search(word)
# param_3 = obj.startsWith(prefix)

#=========================================================================

#211. Design Add and Search Words Data Structure

from typing import Dict
class TrieNode:
    def __init__(self):
        self.children = {}
        self.is_end_of_word = False

class WordDictionary:

    def __init__(self):
        self.root = TrieNode()        

    def addWord(self, word: str) -> None:
        node = self.root
        for char in word:
            if char not in node.children:
                node.children[char] = TrieNode()

            node = node.children[char]
        
        node.is_end_of_word = True

    def search(self, word: str) -> bool:
        def dfs(node, index):
            if index == len(word):
                return node.is_end_of_word
            current_char = word[index]

            if current_char == ".":
                for child in node.children.values():
                    if dfs(child, index+1):
                        return True
                return False
            else:
                if current_char in node.children:
                    return dfs(node.children[current_char], index+1)
                else:
                    return False
        
        return dfs(self.root, 0)
    
#=============================================================================

#212. Word Search II

from typing import List

class TrieNode:
    def __init__(self):
        self.children = {}
        self.is_word = None  # Will store the full word at end node

class Solution:
    def findWords(self, board: List[List[str]], words: List[str]) -> List[str]:
        root = TrieNode()
        result = set()

        # Build Trie
        for word in words:
            node = root
            for char in word:
                if char not in node.children:
                    node.children[char] = TrieNode()
                node = node.children[char]
            node.is_word = word  # Mark the complete word

        rows, cols = len(board), len(board[0])

        def dfs(r, c, node):
            char = board[r][c]

            # If current char not in Trie path → stop
            if char not in node.children:
                return

            next_node = node.children[char]

            # If word found → add to result
            if next_node.is_word:
                result.add(next_node.is_word)
                next_node.is_word = None  # Avoid duplicate results

            # Backtrack
            board[r][c] = "#"  # Mark as visited

            # Explore neighbors
            for dr, dc in [(-1,0), (1,0), (0,-1), (0,1)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < rows and 0 <= nc < cols and board[nr][nc] != "#":
                    dfs(nr, nc, next_node)

            # Restore character
            board[r][c] = char

        # Start DFS from every cell
        for r in range(rows):
            for c in range(cols):
                dfs(r, c, root)

        return list(result)
    

