#EASY LEVEL QUESTIONS:
#=====================================================

#reverse an array:

#method 1:
def reverse_an_array(reverse_array):
    reversed_array=[]
    for i in range(len(reverse_array)-1,-1,-1):
        reversed_array.append(reverse_array[i])
    return reversed_array

#method 2:
def reverse_using_slice(reverse_array):
    return reverse_array[::-1]

#find maximun element in an array

def max_element(array):
    temp=0
    for i in range(len(array)):
        if array[i]>temp:
            temp=array[i]
    return temp

#reverse an array
def reverse_arr(array):
    rev_arr=[]
    for i in range(len(array)-1,-1,-1):       
        rev_arr.append(array[i])
    return rev_arr

#check if array is a palindrome

def is_palindrome(array):
    rev_arr=reverse_arr(array)
    n=len(array)
    for i in range(n//2):
        if array[i]!=rev_arr[i]:
            return False
        
    return True

#method 2:
def is_palind(array):
    return array==array[::-1]

#move all zeros to the end of the array

def move_zeros(array):
    n=len(array)
    new_arr=[]
    count=0
    for i in range(n):
        if array[i]==0:
            count+=1
        else:
            new_arr.append(array[i])

    for i in range(count):
        new_arr.append(0)

    return new_arr

#find the missing numbebr in an array:

def missing_number(array):
    missing_num=[]
    n=len(array)
    for num in range(1,n+1):
        found= False
        for elements in array:
            if num==elements:
                found=True
                break
        if not found:
            missing_num.append(num)
        
    return missing_num

#method 2:
def missing_num_hash(array):
    n= len(array)
    missing_num=[]
    hash_set= set(array)
    for num in range(1,n+1):
        if num not in hash_set:
            missing_num.append(num)
    
    return missing_num

#find max and min element in an array:

#method 1
def max_min(array):
    max_num = max(array)
    min_num = min(array)
    return max_num,min_num

#method 2:
def max_min_logic(array):
    max=array[0]
    min=array[0]
    for num in array:
        if num>max:
            max=num
        if num<min:
            min=num

    return max,min

#count the number of even and odd elements:

def coint_even_odd(array):
    even,odd=0,0
    for num in array:
        if num%2==0:
            even=even+1
        else:
            odd=odd+1
    return even,odd

#frequency of each element in an araray:

def frequency(array):

    for num in set(array):
        count=0
        for i in range(len(array)):
            if num==array[i]:
                count+=1
        print(f"Total number of element {num} in array is {count}")

#remove duplicates from an array:
def remove_duplicates(array):
    return list(set(array))

#check if two arrays are equal: same elements and frequency
def tw_equal_array(array1,array2):
    found=[]
    if len(array1)!=len(array2):
        return "Arrays are not equal"
    
    elif len(array1)==len(array2):
        for num in array1:
            for nums in array2:
                if num==nums:
                    found.append(True)
        if len(found)==len(array1):
            return "Arrays are equal"
        else:
            return "Arrays are not equal"
    
#method 2:
def two_equal_array_hash(array1,array2):
    if len(array1)!=len(array2):
        return "Arrays are not equal"
    hash_set1= set(array1)
    hash_set2= set(array2)
    if hash_set1==hash_set2:
        return "Arrays are equal"
    else:
        return "Arrays are not equal"

#method 3:
def twoo_equal_array_sort(array1,array2):
    if len(array1)!=len(array2):
        return "Arrays are not equal"
    
    return "Arrays are equal" if sorted(array1)==sorted(array2) else "Arrays are not equal" 

#Find the second largest element in an array:

def second_lar(nums):
    max=0
    max2=0
    for num in nums:
        if num>max:
            max2=max
            max=num
    return max2

#method 2:
def second_largest(nums):
    return sorted(nums)[-2]

#method:3
def second_larg(nums):
    if len(nums)<2:
        return "Array should have at least 2 elements"
    
    max_val= float('-inf')
    max2_val=float('-inf')

    for num in nums:
        if num>max_val:
            max2_val=max_val
            max_val=num
        elif num>max2_val:
            max2_val=num
    
    if max2_val==float('-inf'):
        return "No valid second largest number"

    return max2_val
            

#method 4:
def second_large(nums):
    hash_set= set(nums)
    if len(nums)<2:
        return "Array should have at least 2 elements"
    return hash_set.sorted([-2])

#find all elements that appear more than once in an array:
def find_duplicates(array):
    hash_set=set(array)
    for num in hash_set:
        count =0
        for i in range(len(array)):
            if num==array[i]:
                count+=1
        print(f"Total number of element {num} in array is {count}")
    
#method 2:
def find_duplicates_using_dict(array):
    freq={}
    for num in array:
        freq[num]=freq.get(num,0)+1
    
    duplicates=[num for num,count in freq.items() if count>1]

    return duplicates
    

#Merge two sorted arrays:

def merge_arrays(arr1,arr2):
    arr3=arr1+arr2
    return sorted(arr3)


#=================================================================
#MEDIUM LEVEL QUESTIONS:
#=================================================================

#Move all zeros to the end of the array




        


    
    

   

                   