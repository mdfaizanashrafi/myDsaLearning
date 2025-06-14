#Tree DATA STRUCTURE

"""🌳 What Is a Tree?
A tree is a non-linear hierarchical data structure made up of nodes , where each node has:

A value (data)
Zero or more children (sub-nodes)
There is a special node called the root , and every node is connected by edges .

🧱 Basic Terminology

Root: Topmost node in the tree
Child: A node directly connected below another
Parent: Node above another node
Siblings: Nodes with same parent
Leaf: Node with no children
Subtree: Any node + its descendants
Depth: Number of edges from root to this node
Height: Max depth of any leaf in the tree

        A
      / | \
     B  C  D
    / \     \
   E   F     G
      /
     H

Root: A
Children of B: E, F
Leaf nodes: E, H, G
Height: 3 (longest path from root to leaf)

🌲 Types of Trees

1. 🔗 General Tree
Can have any number of children
Used in XML/HTML DOM trees
2. 🧮 Binary Tree
Each node can have at most two children : left & right
Used in binary search trees, heap data structures
3. 🔍 Binary Search Tree (BST)
Left child ≤ parent ≤ Right child
Used for efficient searching, insertion, deletion
4. 🏔 Heap (Max/Min Heap)
Special binary tree used to implement priority queues
5. 📁 Trie
Efficient prefix-based tree for strings
Used in auto-complete features
6. 🔄 Balanced Trees
AVL Trees, Red-Black Trees → automatically balance themselves to ensure O(log n) time operations
     

class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


# Build the tree:
#       1
#      / \
#     2   3
#    /
#   4

root = TreeNode(1)
root.left = TreeNode(2)
root.right = TreeNode(3)
root.left.left = TreeNode(4)

⏱ Time & Space Complexity

Traversal (DFS/BFS): O(n), n = number of nodes: O(h) call stack or O(w) queue

Insert/Delete/Search in BST: O(log n) average, O(n) worst: O(h)

AVL Tree (Balanced): O(log n) always: O(1) or O(log n)

#REAL WORLD USE CASES:

🌳 1. File Systems
Use Case: Organize files and folders in a hierarchical structure.

🧱 2. DOM (HTML)
Use Case: Represent HTML elements as a tree to enable dynamic web page updates.

🔄 3. Binary Search Trees (BSTs)
Use Case: Enable fast searching, insertion, and deletion in ordered datasets.

🏔 4. Heaps
Use Case: Implement efficient priority queues for scheduling or algorithms like Dijkstra.

🧬 5. Trie (Prefix Tree)
Use Case: Accelerate prefix-based searches for autocomplete and spell checking.

🗂 6. B-Trees / B+ Trees
Use Case: Index large datasets in databases and file systems efficiently.

🎮 7. Game Trees
Use Case: Model possible moves in AI games like chess or Go.

🧮 8. Expression Trees
Use Case: Parse and evaluate mathematical expressions or code logic.

📦 9. XML/JSON Structures
Use Case: Represent nested data in APIs and config files for easy traversal.

🧭 10. Network Routing
Use Case: Prevent cycles and find optimal paths in network topologies.

📝 11. Undo/Redo Functionality
Use Case: Efficiently manage text edits in editors using rope trees.

🧬 12. Decision Trees (ML)
Use Case: Make predictions based on branching decision paths in machine learning.

🧮 13. Segment Trees
Use Case: Perform fast range queries and updates on arrays.

🧩 14. Abstract Syntax Trees (ASTs)
Use Case: Parse and compile source code in compilers and interpreters.
     
       """

#===============================================================================

#226. Invert Binary Tree

class Solution:
    def invertTree(self, root: Optional[TreeNode]) -> Optional[TreeNode]:
        if not root:
            return None

        queue=deque([root])
        while queue:
            node = queue.popleft()

            node.left, node.right = node.right, node.left

            if node.left:
                queue.append(node.left)
            if node.right:
                queue.append(node.right)
        
        return root
    
#===========================================================================

#104. Maximum Depth of Binary Tree

class Solution:
    def maxDepth(self, root: Optional[TreeNode]) -> int:
        if not root:
            return 0

        left_depth = self.maxDepth(root.left)
        right_depth = self.maxDepth(root.right)

        return 1+max(left_depth, right_depth)
    
#============================================================================

#543. Diameter of Binary Tree

class Solution:
    def diameterOfBinaryTree(self, root: Optional[TreeNode]) -> int:
        self.diameter =0

        def dfs(node):
            if not node:
                return 0

            left = dfs(node.left)
            right = dfs(node.right)

            self.diameter = max(self.diameter, left+right)

            return 1+max(left, right)
        
        dfs(root)
        return self.diameter

#================================================================

#110: Balannced Binary tree

class Solution:
    def isBalanced(self, root: Optional[TreeNode]) -> bool:
        def dfs(node):
            if not node:
                return 0

            left_height = dfs(node.left)
            right_height = dfs(node.right)

            if left_height == -1 or right_height == -1:
                return -1

            if abs(left_height - right_height) > 1:
                return -1

            return 1+max(left_height, right_height)

        return dfs(root) != -1
    
  #============================================================

  