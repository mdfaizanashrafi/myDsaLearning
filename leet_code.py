#ARRAYS
#=====================================================

#Q:1929: concatenation of array

#Method 1:
def concatenation_of_array(nums):
    n= len(nums)
    ans=[0]*(2*n)
    for i in range(n):
        ans[i]=nums[i]
        ans[i+n]=nums[i]
    return ans


#Method 2:
def concatenation_of_array(nums):
    return nums + nums

#------------------------------------------------------------

#Q:1920: Build array from permutation

#Method 1:
def build_array_from_permutation(nums):
    n=len(nums)
    ans=[0]*n
    for i in range (n):
        ans[i]=nums[nums[i]]
    return ans

#Method 2:
def build_array_from_permutation(nums):
    return [nums[nums[i]] for i in range(len(nums))]

#method 3:

def buildArray(nums):
    ans = [0] * len(nums)

    for idx, num in enumerate(nums):
        ans[idx] = nums[num]
    return ans

#------------------------------------------------------------
#Q:1470: Shuffle the Array
#i=n+i   total length= 2n

#method 1
def shuffle_array(nums,n):
    new_arr=[]
    for i in range(n):
        new_arr.append(nums[i])
        new_arr.append(nums[n+i])
    return new_arr

#method 2
def shuffle_array_using_zip(array,n):
    return [val for pair in zip(array[:n],array[n:]) for val in pair]

#Q:1480: Running Sum of 1d Array
#method 1
def running_sum(nums):
    sum=0
    run_sum=[]
    for num in nums:
        sum=sum+num
        run_sum.append(sum)
    return run_sum

#Q:2011: Find the calue of variable afterr perfprming operatioms:

def final_value_after_operations(operations):
    x=0
    for operation in operations:
        if "++" in operation:
            x=x+1
        elif "--" in operation:
            x=x-1
    return x

#method 2
def final_value_after_operations2(operations):
    return (operations.count("++X")+operations.count("X++")) - (operations.count("--X")+operations.count("X--"))

#Q:1365: How many numbers are smaller than the current number
#method 1

def smaller_numbers_than_current(nums):
    ans=[]
    for num in nums:
        count=0
        for i in range(len(nums)):
            if num>nums[i]:
                count+=1
                
        ans.append(count)
    return ans

#1389: Create target array in the given order

def target_array(nums,index):
    ans=[]
    for num,idx in zip(nums,index):
        ans.insert(idx,num)
    return ans

def max_num_word_in_str(sentences):
    max_len=0
    for sentence in sentences:
        words=sentence.split(" ")
        max_len=max(len(words),max_len)
        
    return max_len

#Q1 : Two Sum question:

def two_sum(nums,target):
    seen={}
    for idx,num in enumerate(nums):
        complement=target - num #comp + num = target
        if complement in seen:
            return [seen[complement],idx]
        seen[num]=idx

#Q242: Valid Anagram:
from collections import defaultdict
def anagram(strs):
    groups = defaultdict(list)
    for str in strs:
        freq=[0]*26
        for ch in str:
            freq[ord(ch)]-freq[ord('a')]
        
        key= tuple(freq)
        groups[key].append(str)
    return list(groups.values())

#=================================================
#Top K Elements: #leetcode 347

class Solution:
    def topKFrequent(self, nums: List[int], k: int) -> List[int]:

    #Step1: count freq using defaultdict    
        freq = defaultdict(int)
        for num in nums:
            freq[num] += 1

    # Step 2: Create bucket
        if not freq:
            return []
    
        max_freq = max(freq.values())
        bucket = [[] for _ in range(max_freq + 1)]

        for num, count in freq.items():
            bucket[count].append(num)

    # Step 3: Collect top k elements
        result = []
        for i in range(len(bucket) - 1, 0, -1):
            for num in bucket[i]:
                result.append(num)
                if len(result) == k:
                    return result

        return result
    
#================================================================
#Encode and Decode: #Leetcode: 271

class Solution:

    def encode(self, strs: List[str]) -> str:
        #encode a list of string to a single string with # and length number
        encoded=''
        for s in strs:
            encoded +=f"{len(s)}#{s}"
        return encoded

    def decode(self, s: str) -> List[str]:
        #decode the encoded string using the # and len
        decoded=[]
        i=0
        while i<len(s):
            #find the positions of #
            j= s.find('#',i)
            if j==-1:
                break
            
            #get the length of the string
            length= int(s[i:j])

            decoded.append(s[j+1:j+1+length])
            i=j+1+length
        return decoded
    
#==========================================

#Leetcode 238: Produce of Array Except Self

class Solution:
    def productExceptSelf(self, nums: List[int]) -> List[int]:
        n= len(nums)
        answer=[1]*n

        #calculate the left product
        left_product=1
        for i in range(n):
            answer[i]=left_product
            left_product *= nums[i]

        #multiply with right product
        right_product=1
        for i in range(n-1,-1,-1):
            answer[i] *= right_product
            right_product *= nums[i]

        return answer

#=====================================

#is Valid Sudoku: Leetcode: 36

def isValidSudoku(board):

    #there are 9 row and colmns, and 9(3x3) boxes
    rows= [set() for _ in range(9)]
    cols=[set() for _ in range(9)]
    boxes= [set() for _ in range(9)]

    for r in range(9):
        for c in range(9):
            num=board[r][c]
            if num == '.':
                continue
            if num in rows[r]:
                return False
            if num in cols[c]:
                return False
            
            box_index= (r//3)*3 + (c//3)
            if num in boxes[box_index]:
                return False
            rows[r].add(num)
            cols[c].add(num)
            boxes[box_index].add(num)
    return True

#====================================================
# Longest consecutive sequence of number #leetcode: 128

def longestConsecutive(nums):
    num_set= set(nums) 
    longest= 0
    for num in num_set:
        if num-1 not in num_set:
            current_num = num
            current_length=1
        
            while current_num+1 in num_set:
                current_num +=1
                current_length +=1

            longest= max(longest, current_length)
        
    return longest

#==================================================
#evaluate polish notation #leetcode 150

class Solution:
    def evalRPN(self, tokens: List[str]) -> int:
        stack=[]
        for token in tokens:
            if token in {'+','-','*','/'}:
                b= stack.pop()
                a=stack.pop()
                if token=='+':
                    result=a+b
                elif token=='-':
                    result=a-b
                elif token=='*':
                    result=a*b
                elif token=='/':
                    result=int(a/b)
                stack.append(result)
            else:
                stack.append(int(token))
        return stack[0]
        

#======================================================
#min stack #155

class MinStack:
    def __init__(self):
        self.stack = []
        self.min_stack = []

    def push(self, val: int) -> None:
        self.stack.append(val)
        if not self.min_stack or val <= self.min_stack[-1]:
            self.min_stack.append(val)

    def pop(self) -> None:
        if self.stack.pop() == self.min_stack[-1]:
            self.min_stack.pop()

    def top(self) -> int:
        return self.stack[-1]

    def getMin(self) -> int:
        return self.min_stack[-1]
    
#==================================================

#Valid parentheisi: #20

class Solution:
    def isValid(self, s: str) -> bool:
        stack=[]
        bracket_map={')':'(','}':'{',']':'['}
        for char in s:
            if char in bracket_map.values():
                stack.append(char)
            elif char in bracket_map:
                if not stack or stack[-1] != bracket_map[char]:
                    return False
                stack.pop()
            else:
                return False
        return not stack


#============================================
#valid palindrome #125

class Solution:
    def isPalindrome(self, s: str) -> bool:
        left, right= 0, len(s)-1
        while left<right:
            while left<right and not s[left].isalnum():
                left+=1
            while left<right and not s[right].isalnum():
                right -=1
            
            if s[left].lower() != s[right].lower():
                return False
            
            left +=1
            right -=1
        
        return True
            
#==========================================================

#reversee polish notation #150 

class Solution:
    def evalRPN(self, tokens: List[str]) -> int:
        stack=[]
        for token in tokens:
            if token in {'+','-','*','/'}:
                b= stack.pop()
                a=stack.pop()
                if token=='+':
                    result=a+b
                elif token=='-':
                    result=a-b
                elif token=='*':
                    result=a*b
                elif token=='/':
                    result=int(a/b)
                stack.append(result)
            else:
                stack.append(int(token))
        return stack[0]
        
#=====================================================

#generate parenthesis #22

class Solution:
    def generateParenthesis(self, n: int) -> List[str]:
        result=[]
        def backtrack(current, open_count, close_count):
            if len(current)==2*n:
                result.append(current)
                return
            if open_count < n:
                backtrack(current+"(", open_count+1,close_count)
            if close_count<open_count:
                backtrack(current+")", open_count, close_count+1)


        backtrack("",0,0)
        return result


#=========================================================

#daily temperature #739

class Solution:
    def dailyTemperatures(self, temperatures: List[int]) -> List[int]:
        result=[0]*len(temperatures)
        stack=[]

        for current_idx in range(len(temperatures)):
            while stack and temperatures[current_idx] > temperatures[stack[-1]]:
                previous_idx= stack.pop()
                result[previous_idx]=current_idx-previous_idx

            stack.append(current_idx)
        return result

#=============================================================

#car fleet #leetcode: 853

class Solution:
    def carFleet(self, target: int, position: List[int], speed: List[int]) -> int:
        cars= sorted(zip(position,speed), key=lambda x:-x[0])

        times=[(target-pos)/spd for pos,spd in cars]

        fleets=[]
        for time in times:
            if not fleets or time > fleets[-1]:
                fleets.append(time)
            
        return len(fleets)

#===================================================================
#largest rectangle histogram #leetcode 84

from typing import List

class Solution:
    def largestRectangleArea(self, heights: List[int]) -> int:
        stack = []
        max_area = 0
        heights.append(0)  # Add a sentinel to help pop all remaining bars at the end

        for i, h in enumerate(heights):
            while stack and heights[stack[-1]] >= h:
                height = heights[stack.pop()]
                # Calculate width
                width = i if not stack else i - stack[-1] - 1
                area = height * width
                max_area = max(max_area, area)
            stack.append(i)
        
        return max_area
    
#===========================================================

#Two Sum II #leetcode 167

class Solution:
    def twoSum(self, numbers: List[int], target: int) -> List[int]:
        left, right = 0, len(numbers)-1
        while left<right:
            curr_sum = numbers[left] + numbers[right]

            if curr_sum == target:
                return [left+1, right+1]

            elif curr_sum < target:
                left +=1
                
            else: 
                right -= 1
        
        return [-1,-1]
    
#====================================================
#container with most water: #leetcode: 11

class Solution:
    def maxArea(self, height: List[int]) -> int:
        left, right = 0, len(height)-1
        max_area=0
        while left < right:
            width= right -left
            curr_height= min(height[left], height[right])
            area = width*curr_height
            max_area = max(max_area, area)

            if height[left] < height[right]:
                left += 1
            else:
                right -= 1
        return max_area
    

#========================================================
#Tapping rain water: #leetcode 42

class Solution:
    def trap(self, height: List[int]) -> int:
        if not height:
            return 0

        left, right = 0, len(height)-1
        left_max=right_max=0
        water_trapped=0

        while left<right:
            if height[left] < height[right]:
                if height[left] >= left_max:
                    left_max = height[left]
                else:
                    water_trapped += left_max - height[left]
                left +=1
            else:
                if height[right]>= right_max:
                    right_max= height[right]
                else:
                    water_trapped += right_max - height[right]
                right -= 1
        return water_trapped
                     

#===================================================
# 3 sum: leetcode: 

class Solution:
    def threeSum(self, nums: List[int]) -> List[List[int]]:
        nums.sort()
        res=[]
        for i in range(len(nums)):
            if i>0 and nums[i]==nums[i-1]:
                continue
            left, right= i+1, len(nums)-1
            while left<right:
                total= nums[i]+nums[left]+nums[right]
                if total ==0:
                    res.append([nums[i],nums[left],nums[right]])
                    left +=1
                    right -=1
                    while left< right and nums[left] == nums[left]-1:
                        left +=1
                    while left< right and nums[right] == nums[right+1]:
                        right -=1
                elif total<0:
                    left+=1
                else:
                    right -=1

        return res


#======================================================

#valid oalindrome #leetcode 125

class Solution:
    def isPalindrome(self, s: str) -> bool:
        left, right= 0, len(s)-1
        while left<right:
            while left<right and not s[left].isalnum():
                left+=1
            while left<right and not s[right].isalnum():
                right -=1
            
            if s[left].lower() != s[right].lower():
                return False
            
            left +=1
            right -=1
        
        return True


#===========================================
#Binary search #leetcode 704

class Solution:
    def search(self, nums: List[int], target: int) -> int:
        low, high=0, len(nums)-1
        while low<=high:
            mid=(low+high)//2
            if nums[mid] == target:
                return mid
            elif nums[mid]<target:
                low=mid+1
            else:
                high = mid-1

        return -1
        
#====================================================

#search a 2d matrix: #74

class Solution:
    def searchMatrix(self, matrix: List[List[int]], target: int) -> bool:
        rows= len(matrix)
        cols= len(matrix[0])

        top, bottom = 0, rows-1
        while top <= bottom:
            row = (top+bottom)//2
            if target > matrix[row][-1]:
                top = row+1
            elif target < matrix[row][0]:
                bottom = row-1
            else:
                break
        
        if top>bottom:
            return False

        left, right = 0, cols-1
        row = (top+bottom) // 2
        while left<=right:
            mid = (left+right)//2
            if matrix[row][mid]== target:
                return True
            elif matrix[row][mid]<target:
                left = mid+1
            else:
                right= mid-1
            
        return False


#=================================================

#875: koko eating bananaas

class Solution:
    def minEatingSpeed(self, piles: List[int], h: int) -> int:
        low=1
        high = max(piles)
        while low< high:
            mid= (low+high)//2
            total_hours = 0
            for pile in piles:
                total_hours += math.ceil(pile/mid)

            if total_hours <= h:
                high = mid
            else:
                low= mid+1

        return low

#==============================================
# find minimum in rotated array #leetcode 153


class Solution:
    def findMin(self, nums: List[int]) -> int:
        left, right=0, len(nums)-1
        while left< right:
            mid = (right+left)//2
            if nums[mid]> nums[right]:
                left= mid+1

            else:
                right= mid
            
        return nums[left]

#==================================================
#search in rotated sorted array #leetcode:33

class Solution:
    def search(self, nums: List[int], target: int) -> int:
        left, right=0, len(nums)-1
        while left<=right:
            mid = (left+right)//2
            if nums[mid]==target:
                return mid
            if nums[left]<=nums[mid]:
                if nums[left]<=target<nums[mid]:
                    right=mid-1
                else:
                    left= mid+1
            else:
                if nums[mid]<target<=nums[right]:
                    left=mid+1
                else:
                    right= mid-1
            
        return -1

#========================================================
#time based Key-Value store: #leetcode: 981

import bisect
class TimeMap:

    def __init__(self):
        self.store = defaultdict(list)  

    def set(self, key: str, value: str, timestamp: int) -> None:
        self.store[key].append((timestamp, value))

    def get(self, key: str, timestamp: int) -> str:
        if key not in self.store:
            return ""
        
        entries= self.store[key]
        i = bisect.bisect_right(entries, (timestamp, chr(255)))-1

        if i>=0:
            return entries[i][1]
        else:
            return ""


# Your TimeMap object will be instantiated and called as such:
# obj = TimeMap()
# obj.set(key,value,timestamp)
# param_2 = obj.get(key,timestamp)

#========================================================
#Median of two sorted arrays: #leetcode 4

class Solution:
    def findMedianSortedArrays(self, nums1: List[int], nums2: List[int]) -> float:
        if len(nums1) > len(nums2):
            nums1, nums2 = nums2, nums1

        m, n = len(nums1), len(nums2)
        total = m+n
        half= total//2

        left, right = 0, m

        while left <= right:
            i = (left+right) // 2
            j = half - i

            max_left1= float('-inf') if i ==0 else nums1[i-1]
            min_right1 = float('inf') if i==m else nums1[i]

            max_left2 = float('-inf') if j ==0 else nums2[j-1]
            min_right2 = float('inf') if j ==n else nums2[j]

            if max_left1 <= min_right2 and max_left2 <= min_right1:
                if total % 2 ==1:
                    return float(min(min_right1, min_right2))
                else:
                    return (max(max_left1, max_left2) + min(min_right1, min_right2)) / 2
            elif max_left1 > min_right2:
                right = i-1
            else:
                left = i +1

#============================================================================


#Best time to buy and ssell stocks #121

class Solution:
    def maxProfit(self, prices: List[int]) -> int:
        min_price= float('inf')
        max_profit = 0

        for price in prices:
            if price<min_price:
                min_price = price
            else:
                max_profit = max(max_profit, price - min_price)
            
        return max_profit
    
#===========================================================
    #best time to buy stock II  #leetcode 122

    class Solution:
        def maxProfit(self, prices: List[int]) -> int:
            profit = 0

            for i in range(1, len(prices)):
                if prices[i]> prices[i-1]:
                    profit += prices[i] - prices[i-1]

            return profit
    
#=========================================================
#best time to buy stock III  #leetcode 123

class Solution:
    def maxProfit(self, prices: List[int]) -> int:
        n = len(prices)

        if n<=1:
            return 0
        
        left_profit = [0]*n
        min_price = prices[0]

        for i in range(1,n):
            min_price = min(min_price, prices[i])
            left_profit[i] = max(left_profit[i-1], prices[i]-min_price)

        max_price = prices[-1]
        right_profit = 0
        max_total = 0

        for i in  range(n-2, -1, -1):
            max_price = max(max_price, prices[i])
            right_profit = max_price - prices[i]
            max_total = max(max_total, left_profit[i] + right_profit)

        return max_total

#=================================================================

#3. Longest Substring Without Repeating Characters

class Solution:
    def lengthOfLongestSubstring(self, s: str) -> int:
        left = 0
        max_len =0
        seen ={}

        for right in range(len(s)):
            char= s[right]

            if char in seen and seen[char] >= left:
                left = seen[char] +1

            seen[char] = right

            max_len = max(max_len, right - left +1)

        return max_len
    
#==========================================================

#424. Longest Repeating Character Replacement
from collections import defaultdict
class Solution:
    def characterReplacement(self, s: str, k: int) -> int:
        left=0
        max_len = 0
        max_freq = 0
        count = defaultdict(int)

        for right in range (len(s)):
            char = s[right]
            count[char] +=1
            max_freq = max(max_freq, count[char])

            while (right-left+1) - max_freq > k:
                count[s[left]] -=1
                left +=1

            max_len = max(max_len, right-left+1)
        return max_len
    
#==============================================================

#567. Permutation in String

class Solution:
    def checkInclusion(self, s1: str, s2: str) -> bool:
        len_s1 = len(s1)
        len_s2 = len(s2)

        if len_s1 > len_s2:
            return False

        from collections import Counter

        s1_count = Counter(s1)
        window_count = Counter()
        for i in range(len_s1):
            window_count[s2[i]] += 1

        if window_count == s1_count:
            return True

        for i in range(len_s1, len_s2):
            left_char = s2[i-len_s1]
            right_char = s2[i]
            window_count[left_char] -= 1
            if window_count[left_char] == 0:
                del window_count[left_char]

            window_count[right_char] += 1

            if window_count == s1_count:
                return True
        
        return False
    
#===============================================================

#76. Minimum Window Substring

class Solution:
    def minWindow(self, s: str, t: str) -> str:
        from collections import defaultdict

        if not s or not t or len(s) < len(t):
            return ""
        
        t_count = defaultdict(int)
        for char in t:
            t_count[char] += 1

        window_count = defaultdict(int)
        required = len(t_count)
        formed =0
        left = 0
        min_len =float('inf')
        result = (0,0)

        for right in range(len(s)):
            char = s[right]
            window_count[char] += 1

            if char in t_count and window_count[char] == t_count[char]:
                formed +=1

            while formed == required:
                if (right - left +1) < min_len:
                    min_len = right-left+1
                    result = (left, right)
                    
                window_count[s[left]] -= 1
                if s[left] in t_count and window_count[s[left]] < t_count[s[left]]:
                    formed -= 1

                left +=1
            
        return s[result[0] : result[1]+1] if min_len != float('inf') else ""
    
#===============================================================================
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

#==========================================================================

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

#=================================================================================
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
    
#========================================================================

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
    
#==========================================================================================

#138. Copy List with Random Pointer

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
 
 #=========================================================================

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

#======================================================================
#TRIE DATA STRUCTURE
#======================================================================

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

#40. Combination Sum II

class Solution:
    def combinationSum2(self, candidates: List[int], target: int) -> List[List[int]]:
        result = []
        candidates.sort()
        def backtrack(start, path, total):
            if total == target:
                result.append(path[:])
                return
            if total> target:
                return
            prev = -1
            for i in range(start, len(candidates)):
                num = candidates[i]
                if num == prev:
                    continue
                prev = num
                path.append(num)
                backtrack(i+1, path, total+num)
                path.pop()
        
        backtrack(0,[],0)
        return result

#============================================================================

#46: Permutatiosn:

class Solution:
    def permute(self, nums: List[int]) -> List[List[int]]:
        result = []
        def backtrack(path, used):
            if len(path) == len(nums):
                result.append(path[:])
                return

            for i in range(len(nums)):
                if not used[i]:
                    used[i] = True

                    path.append(nums[i])
                    backtrack(path, used)
                    path.pop()
                    used[i] = False

        used = [False]*len(nums)
        backtrack([],used)
        return result
    
#==================================================================================
#90. Subsets II

class Solution:
    def subsetsWithDup(self, nums: List[int]) -> List[List[int]]:
        nums.sort()
        result = []
        def backtrack(start, path):
            result.append(path[:])
            for i in range(start, len(nums)):
                if i> start and nums[i] == nums[i-1]:
                    continue

                path.append(nums[i])
                backtrack(i+1, path)
                path.pop()
        
        backtrack(0,[])
        return result
    
#==============================================================================
#79: Word Search

class Solution:
    def exist(self, board: List[List[str]], word: str) -> bool:
        rows, cols = len(board), len(board[0])
        def dfs(r,c,index):
            if index == len(word):
                return True
            if r<0 or c<0 or r>=rows or c>= cols:
                return False

            if board[r][c] != word[index]:
                return False
            temp = board[r][c]
            board[r][c] = "#"

            found = (
                    dfs(r+1,c,index+1) or 
                    dfs(r-1,c,index+1) or
                    dfs(r,c+1,index+1) or
                    dfs(r,c-1,index+1)
            )

            board[r][c] = temp
            return found

        for r in range(rows):
            for c in range(cols):
                if board[r][c] == word[0]:
                    if dfs(r,c,0):
                        return True
        return False

#=====================================================================================

#131. Palindrome Partitioning

class Solution:
    def partition(self, s: str) -> List[List[str]]:
        result = []

        def is_palindrome(sub):
            return sub == sub[::-1]
        
        def backtrack(start, path):
            if start == len(s):
                result.append(path[:])
                return

            for end in range(start+1, len(s)+1):
                substring = s[start:end]
                if is_palindrome(substring):
                    path.append(substring)
                    backtrack(end,path)
                    path.pop()

        backtrack(0,[])
        return result
        
#=========================================================================

#17. Letter Combinations of a Phone Number

class Solution:
    def letterCombinations(self, digits: str) -> List[str]:
        if not digits:
            return []
        
        phone_map = {
            "2": "abc",
            "3": "def",
            "4": "ghi",
            "5": "jkl",
            "6": "mno",
            "7": "pqrs",
            "8": "tuv",
            "9": "wxyz"
        }
        result =[]
        def backtrack(index, path):
            if index == len(digits):
                result.append("".join(path))
                return
            
            possible_letters =  phone_map[digits[index]]
            for letter in possible_letters:
                path.append(letter)
                backtrack(index+1, path)
                path.pop()
        
        backtrack(0, [])
        return result

#================================================================================

#51: N-Queens

class Solution:
    def solveNQueens(self, n: int) -> List[List[str]]:
        result = []
        col_set = set()
        neg_diag = set()
        pos_diag = set()
        state = []

        def is_safe(row,col):
            return not (col in col_set or (row-col) in neg_diag or (row+col) in pos_diag)
        
        def backtrack(row):
            if row ==n:
                board = []
                for r in range(n):
                    row_str = ['.']*n
                    row_str[state[r]] = 'Q'
                    board.append("".join(row_str))
                
                result.append(board)
                return

            for col in range(n):
                if not is_safe(row,col):
                    continue
                
                state.append(col)
                col_set.add(col)
                neg_diag.add(row-col)
                pos_diag.add(row+col)

                backtrack(row+1)
                state.pop()
                col_set.remove(col)
                neg_diag.remove(row-col)
                pos_diag.remove(row+col)

        backtrack(0)
        return result
    
#===========================================================================================
import heapq
#703: Kth largest term in a stream:

class KthLargest:

    def __init__(self, k: int, nums: List[int]):
        self.k = k
        self.min_heap = []
        for num in nums:
            self.add(num)

    def add(self, val: int) -> int:
        heapq.heappush(self.min_heap, val)
        if len(self.min_heap) > self.k:
            heapq.heappop(self.min_heap)
        return self.min_heap[0]

#=============================================================================

#1046. Last Stone Weight

class Solution:
    def lastStoneWeight(self, stones: List[int]) -> int:
        max_heap = [-stone for stone in stones]
        heapq.heapify(max_heap)
        while len(max_heap)>1:
            y = -heapq.heappop(max_heap)
            x= -heapq.heappop(max_heap)
            if y != x:
                new_stone = y-x
                heapq.heappush(max_heap, -new_stone)
            
        return -max_heap[0] if max_heap else 0

#===========================================================================================

#973. K Closest Points to Origin

class Solution:
    def kClosest(self, points: List[List[int]], k: int) -> List[List[int]]:
        max_heap = []
        for x,y in points:
            dist = -(x*x + y*y)
            heapq.heappush(max_heap, (dist, [x,y]))
            if len(max_heap) > k:
                heapq.heappop(max_heap)

        return [point for (_, point) in max_heap]

#================================================================================

#215. Kth Largest Element in an Array

class Solution:
    def findKthLargest(self, nums: List[int], k: int) -> int:
        min_heap = []
        for num in nums:
            heapq.heappush(min_heap, num)
            if len(min_heap) > k:
                heapq.heappop(min_heap)
        return min_heap[0]

#============================================================================

#621. Task Scheduler

from collections import Counter
from typing import List

class Solution:
    def leastInterval(self, tasks: List[str], n: int) -> int:
        if n == 0:
            return len(tasks)  # No cooldown needed

        freq = list(Counter(tasks).values())
        max_freq = max(freq)
        max_count = freq.count(max_freq)

        # Minimum time based on scheduling most frequent tasks
        units_needed = (max_freq - 1) * (n + 1) + max_count

        # Either fit into scheduled slots or just do all tasks sequentially
        return max(len(tasks), units_needed)

#=====================================================================================

#355: Design Twitter

from collections import defaultdict
from typing import List
import heapq

class Twitter:

    def __init__(self):
        self.follow_map = defaultdict(set)  # user -> {followees}
        self.tweets = defaultdict(list)      # user -> [(time, tweetId)]
        self.time = 0                       # Lower numbers = newer tweets

    def postTweet(self, userId: int, tweetId: int) -> None:
        self.tweets[userId].append((self.time, tweetId))
        self.time -= 1  # Decrement so newer tweets have lower time values

    def getNewsFeed(self, userId: int) -> List[int]:
        result = []
        max_heap = []

        # Make sure user follows themselves
        self.follow(userId, userId)

        # Push latest tweet from each followee into heap
        for followee in self.follow_map[userId]:
            if self.tweets[followee]:
                index = len(self.tweets[followee]) - 1
                time, tweetId = self.tweets[followee][index]
                heapq.heappush(max_heap, (time, tweetId, followee, index))

        while max_heap and len(result) < 10:
            time, tweetId, followee, index = heapq.heappop(max_heap)
            result.append(tweetId)

            if index > 0:
                new_index = index - 1
                new_time, new_tweetId = self.tweets[followee][new_index]
                heapq.heappush(max_heap, (new_time, new_tweetId, followee, new_index))

        return result

    def follow(self, followerId: int, followeeId: int) -> None:
        self.follow_map[followerId].add(followeeId)

    def unfollow(self, followerId: int, followeeId: int) -> None:
        if followerId in self.follow_map and followeeId in self.follow_map[followerId]:
            self.follow_map[followerId].remove(followeeId)
        
#========================================================================

#295. Find Median from Data Stream
from heapq import heappush, heappop

class MedianFinder:

    def __init__(self):
        self.max_heap = []
        self.min_heap = []

    def addNum(self, num: int) -> None:
        heappush(self.max_heap, -num)
        if self.max_heap and self.min_heap and (-self.max_heap[0] > self.min_heap[0]):
            val = - heappop(self.max_heap)
            heappush(self.min_heap, val)

        if len(self.max_heap) > len(self.min_heap) +1:
            val = -heappop(self.max_heap)
            heappush(self.min_heap, val)

        elif len(self.min_heap) > len(self.max_heap):
            val = heappop(self.min_heap)
            heappush(self.max_heap, -val)

    def findMedian(self) -> float:
        if len(self.max_heap)>len(self.min_heap):
            return -self.max_heap[0]

        else:
            return (-self.max_heap[0]+self.min_heap[0])/2

#================================================================================

#200. Number of Islands

class Solution:
    def numIslands(self, grid: List[List[str]]) -> int:
        if not grid:
            return 0

        rows, cols = len(grid), len(grid[0])

        def dfs(r,c):
            if r<0 or c< 0 or r>=rows or c>=cols or grid[r][c] == "0":
                return
            grid[r][c]="0"

            dfs(r+1,c)
            dfs(r-1,c)
            dfs(r,c+1)
            dfs(r,c-1)

        count = 0
        for r in range(rows):
            for c in range(cols):
                if grid[r][c]=="1":
                    dfs(r,c)
                    count += 1
        return count 
    
#====================================================================

#133. Clone Graph


# Definition for a Node.
class Node:
    def __init__(self, val = 0, neighbors = None):
        self.val = val
        self.neighbors = neighbors if neighbors is not None else []


from typing import Optional
class Solution:
    def cloneGraph(self, node: Optional['Node']) -> Optional['Node']:
        if not node:
            return None

        old_to_new = {}
        queue = deque([node])
        old_to_new[node] = Node(node.val)

        while queue:
            curr = queue.popleft()
            for neighbor in curr.neighbors:
                if neighbor not in old_to_new:
                    old_to_new[neighbor] = Node(neighbor.val)
                    queue.append(neighbor)
                old_to_new[curr].neighbors.append(old_to_new[neighbor])

        return old_to_new[node]

#=============================================================================

#LeetCode Problem #286: Walls and Gates

class Solution:
    def islandsAndTreasure(self, grid: List[List[int]]) -> None:
        ROWS, COLS = len(grid), len(grid[0])
        queue = deque()
        directions = [(-1,0),(1,0),(0,-1),(0,1)]

        for r in range(ROWS):
            for c in range(COLS):
                if grid[r][c]==0:
                    queue.append((r,c))

        while queue:
            r,c = queue.popleft()
            for dr,dc in directions:
                nr,nc = r+dr, c+dc

                if 0<=nr <ROWS and 0<=nc<COLS and grid[nr][nc] == 2147483647:
                    grid[nr][nc] = grid[r][c]+1
                    queue.append((nr,nc))

#======================================================================================

#994. Rotting Oranges

class Solution:
    def orangesRotting(self, grid: List[List[int]]) -> int:
        if not grid or not grid[0]:
            return 0

        rows, cols = len(grid), len(grid[0])
        queue = deque()
        fresh = 0
        for r in range(rows):
            for c in range(cols):
                if grid[r][c]==1:
                    fresh += 1
                elif grid[r][c] == 2:
                    queue.append((r,c))

        directions=[(-1,0),(1,0),(0,-1),(0,1)]
        time = 0

        while queue and fresh > 0:
            for _ in range(len(queue)):
                r,c = queue.popleft()
                for dr,dc in directions:
                    nr,nc = r+dr, c+dc

                    if 0<=nr<rows and 0<=nc<cols and grid[nr][nc]==1:
                        grid[nr][nc]=2

                        fresh -= 1
                        queue.append((nr,nc))
            time += 1
        return time if fresh == 0 else -1

#======================================================================================

#417. Pacific Atlantic Water Flow

class Solution:
    def pacificAtlantic(self, heights: List[List[int]]) -> List[List[int]]:
        if not heights or not heights[0]:
            return []
        rows, cols = len(heights), len(heights[0])
        directions = [(-1,0),(1,0),(0,-1),(0,1)]

        can_reach_pacific = [[False]*cols for _ in range(rows)]
        can_reach_atlantic = [[False]*cols for _ in range(rows)]

        def dfs(r,c,visited):
            if visited[r][c]:
                return 
            visited[r][c] = True

            for dr,dc in directions:
                nr,nc = r+dr, c+dc

                if 0<=nr<rows and 0<=nc<cols:
                    if heights[nr][nc] >= heights[r][c]:
                        dfs(nr,nc,visited)

        for r in range(rows):
            dfs(r,0,can_reach_pacific)
            dfs(r,cols-1,can_reach_atlantic)

        for c in range(cols):
            dfs(0,c,can_reach_pacific)
            dfs(rows-1, c, can_reach_atlantic)

        result = []

        for r in range(rows):
            for c in range(cols):
                if can_reach_pacific[r][c] and can_reach_atlantic[r][c]:
                    result.append([r,c])

        return result
    
#====================================================================================

#130. Surrounded Regions

class Solution:
    def solve(self, board: List[List[str]]) -> None:
        """
        Do not return anything, modify board in-place instead.
        """
        if not board or not board[0]:
            return

        rows, cols = len(board), len(board[0])

        def dfs(r,c):
            if r<0 or c<0 or r>= rows or c>= cols or board[r][c] != "O":
                return

            board[r][c] = "T"

            dfs(r+1,c)
            dfs(r-1,c)
            dfs(r,c+1)
            dfs(r,c-1)

        for r in range(rows):
            for c in range(cols):
                if board[r][c] == "O" and (r==0 or r==rows-1 or c==0 or c== cols-1):
                    dfs(r,c)
        
        for r in range(rows):
            for c in range(cols):
                if board[r][c]=="O":
                    board[r][c]="X"

        for r in range(rows):
            for c in range(cols):
                if board[r][c]=="T":
                    board[r][c]= "O"

#=====================================================================================

#207. Course Schedule

class Solution:
    def canFinish(self, numCourses: int, prerequisites: List[List[int]]) -> bool:
        graph = [[] for _ in range(numCourses)]
        in_degree = [0]*numCourses
        for dest, src in prerequisites:
            graph[src].append(dest)
            in_degree[dest] += 1

        queue= deque()
        for r in range(numCourses):
            if in_degree[r] == 0:
                queue.append(r)
        count = 0

        while queue:
            curr = queue.popleft()
            count += 1
            for neighbor in graph[curr]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor]==0:
                    queue.append(neighbor)

        return count == numCourses
    
#========================================================================================

#210. Course Schedule II

class Solution:
    def findOrder(self, numCourses: int, prerequisites: List[List[int]]) -> List[int]:
        graph = [[] for _ in range(numCourses)]
        in_degree = [0]*numCourses

        for dest, src in prerequisites:
            graph[src].append(dest)
            in_degree[dest] += 1
        queue = deque([i for i in range(numCourses) if in_degree[i] ==0])

        result = []
        while queue:
            curr = queue.popleft()
            result.append(curr)
            for neighbor in graph[curr]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)
            
        return result if len(result) == numCourses else []

#=============================================================================================

#LeetCode 261: Graph Valid Tree

class Solution:
    def validTree(self, n: int, edges: List[List[int]]) -> bool:
        # Edge case: no edges → empty or single node is always a tree
        if n == 0:
            return True

        parent = list(range(n))

        def find(x):
            if parent[x] != x:
                parent[x] = find(parent[x])
            return parent[x]

        def union(x, y):
            root_x = find(x)
            root_y = find(y)
            if root_x == root_y:
                return False  # Already connected → cycle!
            parent[root_x] = root_y
            return True

        for u, v in edges:
            if not union(u, v):
                return False  # Cycle found

        # Finally: should have exactly n - 1 unions
        return sum(find(i) == i for i in range(n)) == 1
    
#==============================================================================================

#LeetCode 547: Number of Connected Components in an Undirected Graph

from typing import List
from collections import defaultdict

class Solution:
    def countComponents(self, n: int, edges: List[List[int]]) -> int:
        # Step 1: Build adjacency list
        graph = defaultdict(list)
        for u, v in edges:
            graph[u].append(v)
            graph[v].append(u)

        visited = set()
        
        def dfs(node):
            for neighbor in graph[node]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    dfs(neighbor)

        components = 0
        for i in range(n):
            if i not in visited:
                visited.add(i)
                dfs(i)
                components += 1

        return components
    
#============================================================================

#684. Redundant Connection

class Solution:
    def findRedundantConnection(self, edges: List[List[int]]) -> List[int]:
        parent = list(range(1001))
        def find(x):
            if parent[x] != x:
                parent[x] = find(parent[x])
            return parent[x]
        
        def union(x,y):
            root_x = find(x)
            root_y = find(y)
            if root_x == root_y:
                return False
            
            parent[root_x]= root_y

            return True

        for u,v in edges:
            if not union(u,v):
                return [u,v]
            
        return []
    
#==================================================================================

#127. Word Ladder

class Solution:
    def ladderLength(self, beginWord: str, endWord: str, wordList: List[str]) -> int:
        if endWord not in wordList:
            return 0

        L = len(beginWord)
        pattern_map = defaultdict(list)
        for word in wordList:
            for i in range(L):
                pattern = word[:i] + "*" +word[i+1:]
                pattern_map[pattern].append(word)
        visited = set([beginWord])
        queue = deque([(beginWord,1)])

        while queue:
            current_word, level = queue.popleft()
            for i in range(L):
                pattern = current_word[:i]+"*"+current_word[i+1:]
                for neighbor in pattern_map[pattern]:
                    if neighbor == endWord:
                        return level +1
                    if neighbor not in visited:
                        visited.add(neighbor)
                        queue.append((neighbor, level+1))
                
        return 0

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

#=================================================================================

#647. Palindromic Substrings

class Solution:
    def countSubstrings(self, s: str) -> int:
        n = len(s)
        count = 0
        dp = [ [False]*n for _ in range(2)]  # Only keep 2 rows

        for i in range(n-1, -1, -1):  # Start from bottom
            for j in range(i, n):
                if i == j:
                    dp[i%2][j] = True
                    count += 1
                elif i + 1 == j:
                    if s[i] == s[j]:
                        dp[i%2][j] = True
                        count += 1
                else:
                    if s[i] == s[j] and dp[(i+1)%2][j-1]:
                        dp[i%2][j] = True
                        count += 1
                    else:
                        dp[i%2][j] = False

        return count

