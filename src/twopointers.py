#Two pointers:

"""
# 👣 Two Pointers Technique in Programming: Complete Guide

---

## 📌 1. Introduction to Two Pointers

The **Two Pointers** technique is a powerful and intuitive approach used for solving array or string problems where two indices (or pointers) move through the data structure, typically from opposite ends or both from the start.

> 🔍 Think of it like two people walking along a path from different directions to meet in the middle, or one walking faster than the other to catch up.

This technique helps **reduce nested loops**, enabling **O(n)** solutions to problems that would otherwise be **O(n²)**.

---

### ✅ Key Characteristics:

* Uses two indices (`left` and `right`, or `slow` and `fast`) to iterate over data.
* Efficient for **sorted arrays**, **palindrome checks**, and **partitioning**.
* Helps in solving problems involving **pairs**, **subarrays**, and **movement-based logic**.

---

## 🧠 2. When to Use the Two Pointers Technique

Two Pointers is ideal when:

* You're working with **sorted arrays or strings**.
* You need to find **pairs or subarrays** matching a condition.
* You need to traverse an array **from both ends**.
* You’re trying to **compare or filter elements** while minimizing nested loops.

---

## 🔁 3. Types of Two Pointers Usage

| Pattern                    | Description                                                                         |
| -------------------------- | ----------------------------------------------------------------------------------- |
| **Opposite Direction**     | One pointer starts from the beginning, the other from the end (e.g., sum = target). |
| **Same Direction**         | Both pointers move from left to right (e.g., sliding window, longest substring).    |
| **Fast and Slow Pointers** | One pointer moves faster to detect cycles or intervals (used in linked lists).      |
| **Merge Pattern**          | Used in merging two sorted arrays/lists (like merge sort).                          |

---

## 📊 4. Real-World Use Cases

| Use Case                                  | Description                                                     |
| ----------------------------------------- | --------------------------------------------------------------- |
| **Finding a Pair with Given Sum**         | Two pointers move inward on a sorted array to find a sum match. |
| **Removing Duplicates from Sorted Array** | One pointer places unique elements; the other scans ahead.      |
| **Palindrome Check**                      | Compare characters from both ends moving inward.                |
| **Merging Sorted Arrays**                 | One pointer per array, merging as you go.                       |
| **Trapping Rain Water Problem**           | Two pointers to track left and right boundaries.                |
| **Linked List Cycle Detection**           | Floyd’s Tortoise and Hare algorithm using slow/fast pointers.   |
| **Partitioning Arrays**                   | Split array into sections using two-moving pointers.            |

---

## ⚡ 5. Advantages

* **Time-efficient**: Converts brute-force O(n²) solutions to O(n).
* **Memory-efficient**: Often requires no extra space (in-place).
* Highly versatile for a wide range of **array and string problems**.

---

## ⚠️ 6. Limitations

* Often requires **sorted input** for optimal usage.
* Logic can be **tricky to implement** for variable-length conditions.
* Needs careful boundary handling to avoid **off-by-one errors**.

---

## 🔚 7. Conclusion

The **Two Pointers** technique is one of the most elegant and efficient tools in a programmer's problem-solving arsenal. 
It simplifies and speeds up complex problems involving **pairs, intervals, or patterns** in linear structures. Mastering this 
technique opens doors to solving many **interview-level and competitive programming problems** effectively.
"""

#==============================================================================
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