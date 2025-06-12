

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
    
    