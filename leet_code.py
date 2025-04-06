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







