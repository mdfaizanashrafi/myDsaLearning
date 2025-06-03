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

