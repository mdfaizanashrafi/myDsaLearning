"""🧠 What is a Linked List?
A linked list is a linear data structure where elements are stored in nodes , and each node points to the next node in the sequence.

Think of it like:

A treasure hunt where each clue leads you to the next location. 

Each node has two parts:

Data : The value it stores
Pointer/Reference : To the next node"""

"""The Fast and Slow Pointer technique (also known as the Tortoise and Hare algorithm ) is a powerful method used in linked lists and sometimes arrays , especially when you want to:

Detect cycles
Find the middle of a list
Find duplicates
Remove the Nth node from the end
This technique uses two pointers:

Slow pointer : moves 1 step at a time
Fast pointer : moves 2 steps at a time
They start at the same point, but eventually:

If there's a cycle , they will meet
If the list is acyclic , fast pointer will reach the end"""

#======================================================
#206. Reverse Linked List

class Solution:
    def reverseList(self, head: Optional[ListNode]) -> Optional[ListNode]:
        prev = None
        current = head
        while current:
            next_node = current.next
            current.next = prev
            prev = current
            current = next_node
        
        return prev


#===================================================

#21: Merge Two Sorted Linked Lists

class Solution:
    def mergeTwoLists(self, list1: Optional[ListNode], list2: Optional[ListNode]) -> Optional[ListNode]:
        dummy = ListNode(0)
        tail = dummy
        while list1 and list2:
            if list1.val <= list2.val:
                tail.next = list1
                list1 = list1.next
            else:
                tail.next = list2
                list2 = list2.next
            tail=tail.next
        tail.next=list1 or list2
        return dummy.next

#===========================================================
#141. Linked List Cycle

class Solution:
    def hasCycle(self, head: Optional[ListNode]) -> bool:
        slow = head
        fast = head

        while fast and fast.next:
            slow = slow.next
            fast = fast.next.next

            if slow == fast :
                return True

        return False
    
#===========================================================

#143. Reorder List

class Solution :   
    def reorderList(self, head: Optional[ListNode]) -> None:
        """
        Do not return anything, modify head in-place instead.
        """
        if not head or not head.next:
            return
        
        slow, fast = head, head
        while fast and fast.next:
            slow = slow.next
            fast = fast.next.next
        
        prev, curr = None, slow

        while curr:
            temp = curr.next
            curr.next = prev
            prev = curr
            curr = temp

        first, second = head, prev

        while second.next:
            temp = first.next
            first.next = second
            first = temp

            temp = second.next
            second.next = first
            second = temp


#==============================================================================
#Copy list with random pointer  #138

class Solution:
    def copyRandomList(self, head: 'Optional[Node]') -> 'Optional[Node]':
        if not head:
            return None

        curr = head
        while curr:
            clone = Node(curr.val)
            clone.next = curr.next
            curr.next = clone
            curr = clone.next

        curr = head
        while curr:
            if curr.random:
                curr.next.random = curr.random.next
            curr = curr.next.next
            
        cloned_head = head.next
        curr = head
        while curr and curr.next:
            clone = curr.next
            curr = clone.next

            if clone.next:
                clone.next = clone.next.next
            
        return cloned_head
    
#==============================================================================

#19. Remove Nth Node From End of List

class Solution:
    def removeNthFromEnd(self, head: Optional[ListNode], n: int) -> Optional[ListNode]:
        dummy = ListNode(0)
        dummy.next = head
        fast = slow = dummy

        for _ in range(n+1):
            fast = fast.next

        while fast:
            fast = fast.next
            slow = slow.next

        slow.next = slow.next.next

        return dummy.next
    
#==========================================================================
# 2. Add Two Numbers

class Solution:
    def addTwoNumbers(self, l1: Optional[ListNode], l2: Optional[ListNode]) -> Optional[ListNode]:
        dummy = ListNode(0)
        curr = dummy
        carry = 0


        while l1 or l2 or carry:
            val1 = l1.val if l1 else 0
            val2 = l2.val if l2 else 0

            total = val1 + val2 + carry
            carry = total//10
            digit = total % 10
            curr.next = ListNode(digit)

            curr = curr.next
            if l1:
                l1 = l1.next
            if l2:
                l2 = l2.next

        return dummy.next

#==========================================================================
#287. Find the Duplicate Number

class Solution:
    def findDuplicate(self, nums: List[int]) -> int:
        slow = nums[0]
        fast = nums[0]
        while True:
            slow = nums[slow]
            fast = nums[nums[fast]]
            if slow == fast:
                break

        
        slow = nums[0]
        while slow != fast:
            slow = nums[slow]
            fast = nums[fast]

        
        return slow
    
#=================================================================

#146: LRU Cache

class ListNode:
    def __init__(self, key=0, val=0):
        self.key = key
        self.val = val
        self.prev = None
        self.next = None
        
class LRUCache:
    def __init__(self, capacity: int):
        self.cap = capacity
        self.cache = dict()
        self.head = ListNode()
        self.tail = ListNode()
        self.head.next = self.tail
        self.tail.prev = self.head

    def _add(self, node: ListNode):
        node.prev = self.head
        node.next = self.head.next
        self.head.next.prev = node
        self.head.next = node

    def _remove(self, node: ListNode):
        prev_node = node.prev
        next_node = node.next
        prev_node.next = next_node
        next_node.prev = prev_node

    def get(self, key: int) -> int:
        if key not in self.cache:
            return -1
        node = self.cache[key]
        self._remove(node)
        self._add(node)
        return node.val


    def put(self, key: int, value: int) -> None:
        if key in self.cache:
            self._remove(self.cache[key])

        elif len(self.cache) >= self.cap:
            lru = self.tail.prev
            self._remove(lru)
            del self.cache[lru.key]

        new_node = ListNode(key, value)
        self._add(new_node)
        self.cache[key] = new_node

#=====================================================================

#23. Merge k Sorted Lists

from typing import List, Optional
from heapq import heappush, heappop

class Solution:
    def mergeKLists(self, lists: List[Optional[ListNode]]) -> Optional[ListNode]:
        heap = []
        for i, l in enumerate(lists):
            if l:
                heappush(heap, (l.val, i, l))
        
        dummy = ListNode(0)
        curr = dummy
        while heap:
            val, idx, node = heappop(heap)
            curr.next = node
            curr = curr.next

            if node.next:
                heappush(heap, (node.next.val, idx, node.next))
            
        return dummy.next
    

#===========================================================

#25. Reverse Nodes in k-Group

class Solution:
    def reverseKGroup(self, head: Optional[ListNode], k: int) -> Optional[ListNode]:
        dummy = ListNode(0)
        dummy.next = head
        group_prev = dummy

        while True:
            kth = self.getkth(group_prev, k)
            if not kth:
                break
            group_next = kth.next
            curr = group_prev.next
            prev = group_next

            for _ in range(k):
                temp = curr.next
                curr.next = prev
                prev = curr
                curr = temp

            new_group_head = group_prev.next
            group_prev.next = kth
            group_prev = new_group_head
        
        return dummy.next

    def getkth(self, curr: ListNode, k: int) -> ListNode:
        while curr and k>0:
            curr = curr.next
            k -= 1
        return curr
    
