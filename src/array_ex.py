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

print(move_zeros([0,1,0,3,12]))

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




    
    

   

                   