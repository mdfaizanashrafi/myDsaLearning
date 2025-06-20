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

