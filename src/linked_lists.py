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
