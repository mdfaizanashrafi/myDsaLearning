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
    
#========================================================================

#100. Same Binary Tree

class Solution:
    def isSameTree(self, p: Optional[TreeNode], q: Optional[TreeNode]) -> bool:
        if not p and not q:
            return True

        if not p or not q:
            return False

        if p.val != q.val:
            return False
        
        return self.isSameTree(p.left, q.left) and self.isSameTree(p.right, q.right)

#====================================================================================

#572. Subtree of Another Tree

class Solution:
    def isSubtree(self, root: Optional[TreeNode], subRoot: Optional[TreeNode]) -> bool:
        def isSame(p,q):
            if not p and not q:
                return True

            if not p or not q:
                return False

            return p.val == q.val and isSame(p.left, q.left) and isSame(p.right, q.right)
        
        
        def dfs(node):
            if not node:
                return False
            
            if isSame(node, subRoot):
                return True

            return dfs(node.left) or dfs(node.right)
        
        return dfs(root)
    
#===========================================================================

#235. Lowest Common Ancestor of a Binary Search Tree

class Solution:
    def lowestCommonAncestor(self, root: 'TreeNode', p: 'TreeNode', q: 'TreeNode') -> 'TreeNode':
        while root:
            if p.val < root.val and q.val < root.val:
                root = root.left
            
            elif p.val > root.val and q.val > root.val:
                root = root.right
            
            else:
                return root
            
        return None
    
#==================================================================================

#102. Binary Tree Level Order Traversal

class Solution:
    def levelOrder(self, root: Optional[TreeNode]) -> List[List[int]]:
        if not root:
            return []
        
        result = []
        queue = deque([root])

        while queue:
            level_size = len(queue)
            current_level = []

            for _ in range(level_size):
                node= queue.popleft()
                current_level.append(node.val)

                if node.left:
                    queue.append(node.left)
                
                if node.right:
                    queue.append(node.right)
                
            result.append(current_level)
        return result
    
#=================================================================

#199. Binary Tree Right Side View

class Solution:
    def rightSideView(self, root: Optional[TreeNode]) -> List[int]:
        if not root:
            return []
        
        result = []
        queue = deque([root])

        while queue:
            level_size = len(queue)
            for i in range(level_size):
                node=queue.popleft()

                if i == level_size-1:
                    result.append(node.val)
                if node.left:
                    queue.append(node.left)
                if node.right:
                    queue.append(node.right)
            
        return result
    
#==========================================================================
#1448. Count Good Nodes in Binary Tree

class Solution:
    def goodNodes(self, root: TreeNode) -> int:
        def dfs(node, max_so_far):
            if not node:
                return 0

            count = 0
            if node.val >= max_so_far:
                count += 1
                max_so_far = node.val

            count += dfs(node.left, max_so_far)
            count += dfs(node.right, max_so_far)

            return count
    
        return dfs(root, float('-inf'))
    
#======================================================================

#98. Validate Binary Search Tree

class Solution:
    def isValidBST(self, root: Optional[TreeNode]) -> bool:
        def dfs(node, low, high):
            if not node:
                return True

            if node.val <= low or node.val >= high:
                return False

            return dfs(node.left, low, node.val) and dfs(node.right, node.val, high)

        return dfs(root, float('-inf'), float('inf'))

#============================================================================

#230. Kth Smallest Element in a BST

class Solution:
    def kthSmallest(self, root: Optional[TreeNode], k: int) -> int:
        curr = root
        while curr:
            if not curr.left:
                k -= 1
                if k == 0:
                    return curr.val
                curr = curr.right
            else:
                prev = curr.left
                while prev.right and prev.right != curr:
                    prev = prev.right
            
                if not prev.right:
                    prev.right = curr
                    curr = curr.left
                else:
                    prev.right = None
                    k-=1
                    if k==0:
                        return curr.val
                    curr = curr.right

        return -1
    
#===================================================================================

#105. Construct Binary Tree from Preorder and Inorder Traversal

class Solution:
    def buildTree(self, preorder: List[int], inorder: List[int]) -> Optional[TreeNode]:
        inorder_index_map = {val: idx for idx, val in enumerate(inorder)}
        self.pre_idx = 0

        def build(start, end):
            if start > end:
                return None
            
            root_val = preorder[self.pre_idx]
            root = TreeNode(root_val)
            self.pre_idx +=1
            index = inorder_index_map[root_val]

            root.left = build(start, index -1)
            root.right = build(index +1, end)

            return root

        return build(0, len(inorder)-1)
    
#==========================================================================

#124. Binary Tree Maximum Path Sum

class Solution:
    def maxPathSum(self, root: Optional[TreeNode]) -> int:
        self.max_sum = float('-inf')

        def dfs(node):
            if not node:
                return 0

            left_gain = max(dfs(node.left), 0)
            right_gain = max(dfs(node.right), 0)

            current_max_path = node.val + left_gain + right_gain
            self.max_sum = max(self.max_sum, current_max_path)

            return node.val + max(left_gain, right_gain)

        dfs(root)
        return self.max_sum

#========================================================================

#297. Serialize and Deserialize Binary Tree

class Codec:

    def serialize(self, root):
        """Encodes a tree to a single string.
        
        :type root: TreeNode
        :rtype: str
        """
        def dfs(node):
            if not node:
                return ["null"]
            return [str(node.val)]+dfs(node.left)+dfs(node.right)
        return ",".join(dfs(root))

    def deserialize(self, data):
        """Decodes your encoded data to tree.
        
        :type data: str
        :rtype: TreeNode
        """
        nodes = data.split(",")
        self.i = 0

        def build():
            if nodes[self.i] == "null":
                self.i += 1
                return None

            node = TreeNode(int(nodes[self.i]))
            self.i += 1

            node.left = build()
            node.right = build()

            return node

        return build()
        

# Your Codec object will be instantiated and called as such:
# ser = Codec()
# deser = Codec()
# ans = deser.deserialize(ser.serialize(root))

#=======================================================================


