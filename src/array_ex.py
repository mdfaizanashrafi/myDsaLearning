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

#find the second largest element in an array
def second_largest_element(array):
    temp=array[0]
    array_in_ascending_order=[]
    for i in range(len(array)):
        for j in range(i+1,len(array)):
            

                   