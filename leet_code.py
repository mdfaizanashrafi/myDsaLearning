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

print(concatenation_of_array([1,2,1,5,7]))

#Method 2:
def concatenation_of_array(nums):
    return nums + nums

print(concatenation_of_array([1,2,1,5,7]))

#------------------------------------------------------------

#Q:1920: Build array from permutation

#Method 1:
def build_array_from_permutation(nums):
    n=len(nums)
    ans=[0]*n
    for i in range (n):
        ans[i]=nums[nums[i]]
    return ans

print(build_array_from_permutation([0,2,1,5,3,4]))

#Method 2:
def build_array_from_permutation(nums):
    return [nums[nums[i]] for i in range(len(nums))]

print(build_array_from_permutation([0,2,1,5,3,4]))

#method 3:

def buildArray(self, nums: List[int]) -> List[int]:
    ans = [0] * len(nums)

    for idx, num in enumerate(nums):
        ans[idx] = nums[num]
    return ans

#------------------------------------------------------------





