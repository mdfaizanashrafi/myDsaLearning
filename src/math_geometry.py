"""
# 📚 Math & Geometry Revision Notes for DSA

---

## ✅ 1. **Basic Math Concepts**

### 🔢 Integer Properties
- **Even/Odd**:  
  - Even: `n % 2 == 0`  
  - Odd: `n % 2 == 1`

- **Divisibility**:  
  - If `a % b == 0`, then `b` divides `a`.

- **Absolute Value**:  
  - `abs(x)` returns the non-negative value of `x`.

- **GCD (Greatest Common Divisor)**:
  ```python
  import math
  math.gcd(a, b)
  ```

- **LCM (Least Common Multiple)**:
  ```python
  lcm = abs(a * b) // gcd(a, b)
  ```

- **Prime Numbers**:
  - A number >1 that has no divisors other than 1 and itself.
  - Efficient check: loop up to √n.

### 🧮 Modular Arithmetic
- `(a + b) % m = ((a % m) + (b % m)) % m`
- `(a * b) % m = ((a % m) * (b % m)) % m`
- **Modular Inverse** (only exists if `a` and `m` are coprime):
  - Use Fermat’s Little Theorem if `m` is prime: `a^(m-2) % m`

---

## ✅ 2. **Number Theory in DSA**

### 🧮 Sieve of Eratosthenes (Find Primes up to n)

```python
def sieve(n):
    is_prime = [True] * (n+1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(n**0.5)+1):
        if is_prime[i]:
            for j in range(i*i, n+1, i):
                is_prime[j] = False
    return [i for i, val in enumerate(is_prime) if val]
```

### 🔁 Euclidean Algorithm (for GCD)

```python
def gcd(a, b):
    while b:
        a, b = b, a % b
    return a
```

### 📈 Fast Exponentiation (Power in Log n Time)

```python
def power(x, y):
    result = 1
    while y > 0:
        if y & 1:
            result *= x
        x *= x
        y >>= 1
    return result
```

---

## ✅ 3. **Bit Manipulation**

### 🔤 Basic Operations
| Operation       | Description                      |
|----------------|----------------------------------|
| `n << k`       | Multiply by 2^k                  |
| `n >> k`       | Divide by 2^k                    |
| `n & 1`        | Check if even/odd               |
| `n ^ n`        | Clears bits (result = 0)        |
| `n ^ 0`        | Returns n                         |
| `n & (n - 1)`  | Turns off the rightmost set bit |

### 🧠 Useful Tricks
- Count set bits (Hamming weight):  
  ```python
  bin(n).count('1')
  ```
- Swap two numbers without extra space:
  ```python
  a ^= b
  b ^= a
  a ^= b
  ```

---

## ✅ 4. **Geometry Basics**

### 📐 Distance Between Two Points

Given points $ P_1(x_1, y_1) $, $ P_2(x_2, y_2) $

- **Euclidean distance**:
  $$
  d = \sqrt{(x_2 - x_1)^2 + (y_2 - y_1)^2}
  $$

- **Manhattan distance**:
  $$
  d = |x_2 - x_1| + |y_2 - y_1|
  $$

### 🔺 Triangle Area (Heron’s Formula)

Given side lengths $ a, b, c $:

- Semi-perimeter: $ s = \frac{a+b+c}{2} $
- Area: $ A = \sqrt{s(s-a)(s-b)(s-c)} $

### 📏 Line Slope

For line through $ (x_1, y_1), (x_2, y_2) $:

$$
\text{slope} = \frac{y_2 - y_1}{x_2 - x_1},\quad x_2 \ne x_1
$$

### 📐 Collinearity Check

Three points $ A, B, C $ are collinear if:

- Slope of AB == Slope of AC

Or use determinant:
$$
\begin{vmatrix}
x_1 & y_1 & 1 \\
x_2 & y_2 & 1 \\
x_3 & y_3 & 1 \\
\end{vmatrix} = 0
$$

---

## ✅ 5. **Coordinate Geometry in DSA Problems**

### 📍 Common Problem Types:
- Find closest pair
- Overlapping intervals
- Convex hull
- Rotate coordinates
- Detect rectangle from points
- Find intersection between lines or shapes

---

## ✅ 6. **Math Functions in Python**

| Function         | Purpose                           |
|------------------|-----------------------------------|
| `math.sqrt(x)`   | Square root                       |
| `math.floor(x)`  | Round down                        |
| `math.ceil(x)`   | Round up                          |
| `math.log(x)`    | Natural log                       |
| `math.log(x, b)` | Log base `b`                      |
| `math.factorial` | Factorial function                |
| `math.isqrt(x)`  | Integer square root (Python 3.8+) |

---

## ✅ 7. **Important Series and Formulas**

### 📥 Arithmetic Progression (AP)
- Sum of first `n` terms:  
  $$
  S_n = \frac{n}{2}(2a + (n-1)d)
  $$

### 📈 Geometric Progression (GP)
- Sum of first `n` terms:  
  $$
  S_n = a \cdot \frac{1 - r^n}{1 - r},\quad r ≠ 1
  $$

### 🧮 Sum of First N Numbers
- $ \sum_{i=1}^{N} i = \frac{N(N+1)}{2} $
- $ \sum_{i=1}^{N} i^2 = \frac{N(N+1)(2N+1)}{6} $
- $ \sum_{i=1}^{N} i^3 = \left(\frac{N(N+1)}{2}\right)^2 $

---

## ✅ 8. **Common Math/Geometry Interview Problems**

| Problem Title                     | Approach                             |
|----------------------------------|--------------------------------------|
| Reverse Integer                   | Digit extraction using modulo/divide |
| Palindrome Number                 | Reverse half digits                  |
| Single Number                     | XOR trick                            |
| Missing Number                    | XOR or sum method                    |
| Counting Bits                     | DP + Bit shift                       |
| Reverse Bits                      | Bit manipulation                     |
| Add Two Integers Without '+'      | Bitwise XOR + AND                    |
| Valid Triangle                    | Triangle inequality                  |
| Max Points on a Line              | Hash map slope counting              |
| Rectangle Overlap                 | Compare bounds                       |
| K Closest Points to Origin        | Heap or sort with Euclidean distance |

---

## 📝 Bonus Tips

- Always handle **integer overflow** in problems involving large values.
- Use **modulo operations** to keep results within limits.
- Precompute factorials or powers if used multiple times.
- Avoid floating-point when possible (use cross product instead of division).
- Be cautious with **precision errors** in geometry (e.g., use `EPSILON` comparisons).
"""
#======================================================================================

#48. Rotate Image

class Solution:
    def rotate(self, matrix: List[List[int]]) -> None:
        """
        Do not return anything, modify matrix in-place instead.
        """
        n = len(matrix)
    
        for i in range(n // 2):
            for j in range(i, n - i - 1):
            # Save top element
                temp = matrix[i][j]
            
            # Move left to top
                matrix[i][j] = matrix[n - 1 - j][i]
            
            # Move bottom to left
                matrix[n - 1 - j][i] = matrix[n - 1 - i][n - 1 - j]
            
            # Move right to bottom
                matrix[n - 1 - i][n - 1 - j] = matrix[j][n - 1 - i]
            
            # Move saved top to right
                matrix[j][n - 1 - i] = temp
              
#==========================================================================

#54: Spiral Matrix:

class Solution:
    def spiralOrder(self, matrix: List[List[int]]) -> List[int]:
        if not matrix or not matrix[0]:
            return []

        m, n = len(matrix), len(matrix[0])
        result = []
        top, bottom = 0, m - 1
        left, right = 0, n - 1
    
        while len(result) < m * n:
            for col in range(left, right + 1):
                result.append(matrix[top][col])
            top += 1
        
            if len(result) == m * n:
                break
            for row in range(top, bottom + 1):
                result.append(matrix[row][right])
            right -= 1
        
            if len(result) == m * n:
                break
            for col in range(right, left - 1, -1):
                result.append(matrix[bottom][col])
            bottom -= 1
        
            if len(result) == m * n:
                break
        
            for row in range(bottom, top - 1, -1):
                result.append(matrix[row][left])
            left += 1

        return result
    
#======================================================================================

#73. Set Matrix Zeroes

class Solution:
    def setZeroes(self, matrix: List[List[int]]) -> None:
        """
        Do not return anything, modify matrix in-place instead.
        """
            
        if not matrix or not matrix[0]:
            return

        m, n = len(matrix), len(matrix[0])
        first_row_has_zero = any(matrix[0][j] == 0 for j in range(n))
        first_col_has_zero = any(matrix[i][0] == 0 for i in range(m))

        for i in range(1, m):
            for j in range(1, n):
                if matrix[i][j] == 0:
                    matrix[i][0] = 0
                    matrix[0][j] = 0

    
        for i in range(1, m):
            for j in range(1, n):
                if matrix[i][0] == 0 or matrix[0][j] == 0:
                    matrix[i][j] = 0

    
        if first_row_has_zero:
            for j in range(n):
                matrix[0][j] = 0

   
        if first_col_has_zero:
            for i in range(m):
                matrix[i][0] = 0
              
#========================================================================

#202. Happy Number

class Solution:
    def isHappy(self, n: int) -> bool:
        def get_next(num: int) -> int:
            return sum(int(digit) ** 2 for digit in str(num))
    
        slow = n
        fast = get_next(n)
    
        while fast != 1 and slow != fast:
            slow = get_next(slow)
            fast = get_next(get_next(fast))
    
        return fast == 1

#======================================================================

#66. Plus One

class Solution:
    def plusOne(self, digits: List[int]) -> List[int]:
        n = len(digits)

        for i in range(n - 1, -1, -1):
            if digits[i] < 9:
                digits[i] += 1
                return digits
            digits[i] = 0  

        return [1] + digits
    
#======================================================================

#50. Pow(x, n)

class Solution:
    def myPow(self, x: float, n: int) -> float:
        def helper(x, n):
            if n == 0:
                return 1
            half = helper(x, n // 2)
            if n % 2 == 0:
                return half * half
            else:
                return half * half * x

        N = abs(n)
        result = helper(x, N)
        return result if n >= 0 else 1 / result
  
#================================================================================

#43: Multiply Strings:

class Solution:
    def multiply(self, num1: str, num2: str) -> str:

        if num1 == "0" or num2 == "0":
            return "0"

        m, n = len(num1), len(num2)
        result = [0] * (m + n)

    # Reverse strings for easy processing
        num1 = num1[::-1]
        num2 = num2[::-1]

    # Multiply digit by digit
        for i in range(m):
            for j in range(n):
                digit1 = ord(num1[i]) - ord('0')
                digit2 = ord(num2[j]) - ord('0')
                product = digit1 * digit2

            # Position in result
                pos1 = i + j
                pos2 = i + j + 1

            # Add to result and handle carry
                result[pos1] += product
                result[pos2] += result[pos1] // 10
                result[pos1] %= 10

    # Remove trailing zeros and build final string
        while result[-1] == 0:
            result.pop()

        return ''.join(str(digit) for digit in reversed(result))

#==========================================================================

#2013. Detect Squares

from collections import defaultdict
from typing import List

class DetectSquares:

    def __init__(self):
        # Dictionary to store frequency of each point
        self.points = defaultdict(int)
        # List to keep unique points for fast iteration
        self.point_list = []

    def add(self, point: List[int]) -> None:
        x, y = point
        key = (x, y)
        self.points[key] += 1

        # Only add to point_list if it's a new unique point
        if key not in self.point_list:
            self.point_list.append(key)

    def count(self, point: List[int]) -> int:
        x, y = point
        count = 0

        for (x1, y1) in self.point_list:
            # Check if (x, y) and (x1, y1) can be diagonal corners of a square
            if abs(x1 - x) == abs(y1 - y) and x1 != x:
                # Check if the other two required points exist
                if (x1, y) in self.points and (x, y1) in self.points:
                    # Multiply frequencies
                    count += (
                        self.points[(x1, y1)] *
                        self.points[(x1, y)] *
                        self.points[(x, y1)]
                    )
        return count
    
