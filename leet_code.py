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