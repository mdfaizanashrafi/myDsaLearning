#Two pointers:
#===================
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