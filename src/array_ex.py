#🧮 Arrays in Programming: Introduction, Types, Examples & Real-World Uses

"""
📌 What is an Array?
An array is a data structure that stores a fixed-size sequential collection of elements of the same type . 
It allows you to store multiple values under a single variable name and access them using indexes (positions).

Arrays are one of the most fundamental and widely used data structures in programming.

Here’s a well-structured Google Docs-style content on **Arrays** — including an introduction, types, examples, and real-world use cases in programming. You can copy this content directly into a Google Doc.

---

# 📚 Arrays in Programming: Complete Guide

## 📌 1. Introduction to Arrays

An **array** is a **data structure** used to **store multiple values** of the **same type** in a **single variable**. Arrays are used when you want to work with many variables of the same kind more efficiently and logically.

> 🔍 Think of an array like a **row of mailboxes** — each box holds a value, and each has a number (index) so you can easily access it.

**Key Points:**

* Arrays hold **elements** that are **indexed** starting from 0.
* Arrays are **fixed in size** in many programming languages (like C/C++/Java), but **dynamic** in others (like Python or JavaScript using lists).
* Arrays allow **random access** using an index.

---

## 🧩 2. Types of Arrays

| Type                       | Description                                                                | Example (Syntax)                         |
| -------------------------- | -------------------------------------------------------------------------- | ---------------------------------------- |
| **1D Array**               | A simple array with a single row of elements                               | `int arr[5] = {1, 2, 3, 4, 5};`          |
| **2D Array**               | An array of arrays (like a matrix or table)                                | `int matrix[2][3] = {{1,2,3}, {4,5,6}};` |
| **Multidimensional Array** | Arrays with more than 2 dimensions (rarely used)                           | `int arr[2][2][2];`                      |
| **Dynamic Array**          | Grows/shrinks in size during runtime (like Python lists or Java ArrayList) | `arr.append(10)` in Python               |
| **Jagged Array**           | An array of arrays where inner arrays can have different lengths           | `int[][] jagged = new int[3][];`         |

---

## 💡 3. Code Examples in Different Languages

### ➤ C++ (1D Array)

```cpp
#include<iostream>
using namespace std;
int main() {
    int numbers[3] = {10, 20, 30};
    cout << numbers[1]; // Output: 20
}
```

### ➤ Python (Dynamic List)

```python
numbers = [10, 20, 30]
print(numbers[1])  # Output: 20
```

### ➤ JavaScript (Array)

```javascript
let fruits = ["apple", "banana", "cherry"];
console.log(fruits[2]); // Output: cherry
```

### ➤ Java (2D Array)

```java
int[][] matrix = {
    {1, 2},
    {3, 4}
};
System.out.println(matrix[1][0]); // Output: 3
```

---

## 🌍 4. Real-World Use Cases

| Use Case                | Description                                                          |
| ----------------------- | -------------------------------------------------------------------- |
| **Image Processing**    | Images are stored as 2D arrays of pixels (RGB values).               |
| **Games**               | Game boards (like chess) use 2D arrays to track player positions.    |
| **Data Tables**         | Represent rows and columns using arrays.                             |
| **Sensor Data Storage** | Store real-time sensor values like temperature or pressure readings. |
| **String Manipulation** | Strings are internally arrays of characters.                         |
| **Finance**             | Arrays store stock prices, time-series data, etc.                    |

---

## ✅ 5. Advantages of Arrays

* **Fast access** using index.
* **Efficient storage** of related items.
* Easy to **iterate** and **manipulate** using loops.

---

## ⚠️ 6. Limitations

* **Fixed size** in static arrays.
* **All elements must be of the same type**.
* **Insertion/deletion** can be costly (in terms of shifting elements).

---

## 🔚 Conclusion

Arrays are **fundamental building blocks** in programming. Mastering arrays allows you to handle large data sets efficiently and build more complex data structures like stacks, queues, and matrices. Whether you're building a simple to-do app or a machine learning model, arrays are always behind the scenes.

---

Would you like me to export this as a **Google Docs link** or generate it as a **PDF** for download?


"""

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

def move_all_zeros_at_the_end(nums):
    zero_count=0
    ans=[]
    for num in nums:
        if num!=0:
            ans.append(num)
        elif num==0:
            zero_count+=1
    ans.extend([0]*zero_count)
    return ans

#Find the Missing Number from 0 to n:

def missing_num(nums):
    n= len(nums)
    total_sum=n*(n+1)//2
    actual_sum= sum(nums)
    return (total_sum-actual_sum)

#=====================================





        


    
    

   

                   