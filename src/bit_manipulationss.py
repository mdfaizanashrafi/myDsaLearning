"""
# 🧠 Bit Manipulation in Python (DSA Revision Notes)

Bit Manipulation is the technique of performing operations on binary representations of numbers directly. It’s widely used in competitive programming and system-level optimization.

---

## 🧾 Binary Basics

| Operation | Binary |
| --------- | ------ |
| 1         | `0001` |
| 2         | `0010` |
| 3         | `0011` |
| 4         | `0100` |

```python
bin(5)     # '0b101'
int('101', 2)  # 5
```

---

## 🛠️ Common Bitwise Operators in Python

| Operator | Name        | Description              |                            |
| -------- | ----------- | ------------------------ | -------------------------- |
| `&`      | AND         | 1 if both bits are 1     |                            |
| \`       | \`          | OR                       | 1 if at least one bit is 1 |
| `^`      | XOR         | 1 if bits are different  |                            |
| `~`      | NOT         | Inverts all the bits     |                            |
| `<<`     | Left Shift  | Shifts bits to the left  |                            |
| `>>`     | Right Shift | Shifts bits to the right |                            |

### 🔍 Examples

```python
a = 5       # 0101
b = 3       # 0011

print(a & b)   # 1 -> 0001
print(a | b)   # 7 -> 0111
print(a ^ b)   # 6 -> 0110
print(~a)      # -6 (2's complement)
print(a << 1)  # 10 (shift left)
print(a >> 1)  # 2  (shift right)
```

---

## 🧮 Useful Bit Tricks (with Examples)

### ✅ Check if a number is even or odd

```python
def is_even(n):
    return (n & 1) == 0
```

### ✅ Get the ith bit (0-based)

```python
def get_bit(n, i):
    return (n >> i) & 1
```

### ✅ Set the ith bit

```python
def set_bit(n, i):
    return n | (1 << i)
```

### ✅ Clear the ith bit

```python
def clear_bit(n, i):
    return n & ~(1 << i)
```

### ✅ Toggle the ith bit

```python
def toggle_bit(n, i):
    return n ^ (1 << i)
```

### ✅ Count set bits (Brian Kernighan’s Algorithm)

```python
def count_set_bits(n):
    count = 0
    while n:
        n &= (n - 1)
        count += 1
    return count
```

### ✅ Check if a number is a power of two

```python
def is_power_of_two(n):
    return n > 0 and (n & (n - 1)) == 0
```

### ✅ Remove the lowest set bit

```python
n & (n - 1)
```

### ✅ Isolate the lowest set bit

```python
n & -n
```

---

## 🧠 Advanced Use Cases in DSA

### 1. **Finding the Unique Number in an Array**

Where every element appears twice except one:

```python
def find_unique(arr):
    xor = 0
    for num in arr:
        xor ^= num
    return xor
```

### 2. **Find 2 non-repeating numbers**

Where every element appears twice except two elements:

```python
def find_two_unique(arr):
    xor = 0
    for num in arr:
        xor ^= num
    
    set_bit = xor & -xor  # rightmost set bit
    x = y = 0
    for num in arr:
        if num & set_bit:
            x ^= num
        else:
            y ^= num
    return x, y
```

### 3. **Subset Generation using Bitmask**

```python
arr = [1, 2, 3]
n = len(arr)
for i in range(1 << n):
    subset = [arr[j] for j in range(n) if i & (1 << j)]
    print(subset)
```

### 4. **Count number of 1s from 1 to N**

```python
def count_ones(n):
    count = 0
    i = 1
    while i <= n:
        divider = i * 10
        count += (n // divider) * i + min(max(n % divider - i + 1, 0), i)
        i *= 10
    return count
```

---

## 🔄 Conversion Between Number Systems

### Decimal to Binary

```python
bin(10)  # '0b1010'
```

### Binary to Decimal

```python
int('1010', 2)  # 10
```

---

## 📦 Applications of Bit Manipulation

| Problem Type                | Use of Bit Manipulation           |
| --------------------------- | --------------------------------- |
| Optimization problems       | Fast operations, space efficiency |
| Counting subsets            | Bitmasking                        |
| Permissions & Flags         | Represent state as bits           |
| Graphics & Embedded Systems | Efficient storage                 |
| XOR-based problems          | Unique elements, parity check     |
| Power-of-Two checks         | Fast validation                   |
| Competitive programming     | Speed critical solutions          |

---

## 🧰 Common Python Gotchas

* `~x` is equal to `-x - 1`
* Python has **infinite-length integers**, so bit tricks for overflow won't work as in C++/Java.
* Bitmasking works the same way, but performance impact is negligible unless extremely large loops are involved.

---

## 🧪 Practice Problems (Leetcode)

| Problem             | Link                                                              |
| ------------------- | ----------------------------------------------------------------- |
| Single Number       | [Leetcode 136](https://leetcode.com/problems/single-number)       |
| Counting Bits       | [Leetcode 338](https://leetcode.com/problems/counting-bits)       |
| Subsets             | [Leetcode 78](https://leetcode.com/problems/subsets)              |
| Sum of Two Integers | [Leetcode 371](https://leetcode.com/problems/sum-of-two-integers) |
| Power of Two        | [Leetcode 231](https://leetcode.com/problems/power-of-two)        |

---

## ✅ Summary Cheat Sheet

| Task                  | Expression              |            |
| --------------------- | ----------------------- | ---------- |
| Get ith bit           | `(n >> i) & 1`          |            |
| Set ith bit           | \`n                     | (1 << i)\` |
| Clear ith bit         | `n & ~(1 << i)`         |            |
| Toggle ith bit        | `n ^ (1 << i)`          |            |
| Multiply by 2^k       | `n << k`                |            |
| Divide by 2^k         | `n >> k`                |            |
| Is n even?            | `n & 1 == 0`            |            |
| Count set bits        | `while n: n &= (n - 1)` |            |
| Is power of 2         | `n & (n - 1) == 0`      |            |
| Remove lowest set bit | `n & (n - 1)`           |            |
| Get lowest set bit    | `n & -n`                |            |
"""

#=======================================================================================

#136. Single Number

class Solution:
    """
    #BRUTE FORCE:

    def singleNumber(self, nums: List[int]) -> int:
        count = {}
        for num in nums:
            count[num]=count.get(num,0)+1
        
        for num in count:
            if count[num]==1:
                return num
        """
    def singleNumber(self, nums: List[int]) -> int:
        result = 0
        for num in nums:
            result ^= num
        return result

#=======================================================================

#191. Number of 1 Bits

class Solution:
    def hammingWeight(self, n: int) -> int:
       count = 0
       for _ in range(32):
          count += n&1
          n >>=1
       return count
    
#======================================================

#number of 1 bits #191

class Solution:
    def hammingWeight(self, n: int) -> int:
       count = 0
       for _ in range(32):
          count += n&1
          n >>=1
       return count

#============================================================

#338. Counting Bits

class Solution:
    def countBits(self, n: int) -> List[int]:
        dp = [0] * (n + 1)
        for i in range(1, n + 1):
            dp[i] = dp[i >> 1] + (i & 1)
        return dp
    
#=================================================================

#190. Reverse Bits

def reverseBits(n: int) -> int:
    result = 0
    for _ in range(32):
        result = (result << 1) | (n & 1)  # Add last bit of n to result
        n >>= 1  # Move to next bit
    return result

#================================================================

#268. Missing Number

class Solution:
    def missingNumber(self, nums: List[int]) -> int:
        xor = 0
        n = len(nums)
    
    # XOR all numbers from 0 to n
        for i in range(n + 1):
            xor ^= i

    # XOR all elements in the array
        for num in nums:
            xor ^= num

        return xor

#================================================================

#371. Sum of Two Integers

class Solution:
    def getSum(self, a: int, b: int) -> int:
        MASK = 0xFFFFFFFF
        MAX_INT = 0x7FFFFFFF  

        while b != 0:
            a, b = (a ^ b), (a & b) << 1
            a = (a & MASK)
        return a if a <= MAX_INT else ~(a ^ MASK)

#===============================================================

#7. Reverse Integer

def reverse(x: int) -> int:
    INT_MIN, INT_MAX = -2**31, 2**31 - 1
    sign = -1 if x < 0 else 1
    x_abs = abs(x)
    reversed_num = 0

    while x_abs != 0:
        digit = x_abs % 10
        x_abs //= 10
        if reversed_num > (INT_MAX - digit) // 10:
            return 0
        reversed_num = reversed_num * 10 + digit
    return sign * reversed_num

#================================================================

#