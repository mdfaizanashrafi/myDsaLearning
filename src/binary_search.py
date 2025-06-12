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
