
"""

# 📚 Stack in Programming: Complete Guide

---

## 📌 1. Introduction to Stack

A **Stack** is a **linear data structure** that follows the **Last In, First Out (LIFO)** principle. This means the last element added to the stack is the first one to be removed.

> 🔍 Think of a stack like a pile of plates—new plates are added on top, and the last plate added is the first one you take off.

Stacks are commonly used in scenarios where **reversal**, **backtracking**, or **nested function execution** is involved.

---

### ✅ Key Characteristics:

* Operates with **push (insert)** and **pop (remove)** operations.
* Follows **LIFO (Last In, First Out)** order.
* Can be implemented using **arrays** or **linked lists**.
* Often supports **peek/top()** to view the top element without removing it.

---

## 🔁 2. When to Use a Stack

Stacks are ideal when:

* You need to reverse an order.
* The problem involves **nested or recursive structure**.
* You need to **undo** operations.
* Parsing or **expression evaluation** is involved.

---

## 🧩 3. Types of Stack Usage

| Type or Variation      | Description                                                        |
| ---------------------- | ------------------------------------------------------------------ |
| **Simple Stack**       | Standard push/pop functionality                                    |
| **Monotonic Stack**    | Stack that maintains elements in increasing or decreasing order    |
| **Call Stack**         | Used by programming languages to handle function calls and returns |
| **Two-Stack Queue**    | Technique to implement a queue using two stacks                    |
| **Stack with Min/Max** | Stack that tracks minimum or maximum in constant time              |

---

## 📊 4. Real-World Use Cases

| Use Case                         | Description                                                              |
| -------------------------------- | ------------------------------------------------------------------------ |
| **Undo/Redo Functionality**      | Track recent actions in applications like editors and IDEs               |
| **Expression Evaluation**        | Evaluate infix/postfix expressions using operator precedence             |
| **Backtracking Algorithms**      | Used in maze solving, recursion, DFS (Depth-First Search)                |
| **Web Browser Navigation**       | Go back to the previous page using a history stack                       |
| **Function Call Management**     | Programming languages use the call stack to manage active function calls |
| **Balanced Parentheses Checker** | Validate if brackets, braces, or parentheses are balanced                |
| **Compiler Syntax Parsing**      | Parsing nested tokens or grammar trees using stack-based parsing         |

---

## ⚡ 5. Advantages

* Simple and easy to implement.
* Very efficient for managing **nested or hierarchical data**.
* Helps in maintaining **execution state**, like recursion and function calls.

---

## ⚠️ 6. Limitations

* **Limited access**: Only the top element is accessible.
* May **overflow** if maximum capacity is fixed and exceeded.
* Less suitable for searching or accessing random elements.

---

## 🔚 7. Conclusion

Stacks are a foundational data structure that appear everywhere from **parsing expressions** to **managing function calls**. 
Their simplicity and power make them essential in both theory and real-world applications. Whether you're building an **undo system**, 
analyzing **syntax trees**, or implementing **depth-first search**, the stack is your go-to structure.

"""

#==============================================================================================
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
    
    