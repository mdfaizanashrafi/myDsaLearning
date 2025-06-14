
"""

# 🪟 Sliding Window Technique in Programming: Complete Guide

---

## 📌 1. Introduction to Sliding Window

The **Sliding Window** is a **computational technique** used to reduce the time complexity of algorithms that involve **linear data structures** (like arrays, strings, or lists). It’s especially effective when dealing with **subarrays** or **substrings** of a given size.

> 🔍 Imagine looking through a window that moves one step at a time across a row of buildings, and you only see the view from that window—this is the essence of the sliding window.

Instead of recalculating results for every new window from scratch, this technique **reuses information** from the previous window, resulting in **faster and more optimized solutions**.

---

### ✅ Key Characteristics:

* Reduces time complexity from **O(n²)** to **O(n)** in many problems.
* Typically uses two pointers to define a **current window**.
* Can be **fixed-size** or **variable-size**.
* Ideal for problems involving **contiguous segments**.

---

## 🧠 2. When to Use Sliding Window

Sliding Window is useful when:

* You're working with **contiguous data** (e.g., subarrays, substrings).
* You need to find **maximum**, **minimum**, **sum**, **average**, **count**, or **longest/shortest** segment under some constraint.
* You want to **optimize** brute-force nested loop solutions.

---

## 🔁 3. Types of Sliding Window

| Type                       | Description                                                                                  |
| -------------------------- | -------------------------------------------------------------------------------------------- |
| **Fixed-Size Window**      | The window size `k` is constant. The window slides one element at a time.                    |
| **Variable-Size Window**   | The window size expands and shrinks based on conditions (e.g., sum, character frequency).    |
| **Maximum/Minimum Window** | Specialized problems that track the max/min in the current window. Often solved with Deques. |

---

## 📊 4. Real-World Use Cases

| Use Case                                | Description                                                                |
| --------------------------------------- | -------------------------------------------------------------------------- |
| **Network Packet Analysis**             | Analyze a fixed-size window of packets over time for anomalies.            |
| **Time Series Data Processing**         | Compute rolling averages, trends, or moving sums.                          |
| **String Matching / Pattern Detection** | Check for anagrams, substrings, or palindromes within a window.            |
| **Stock Market Analysis**               | Find max profit, average price, or volatility over a given window of days. |
| **Data Stream Processing**              | Continuously monitor streams in real time (e.g., sensor data, logs).       |
| **Audio/Video Processing**              | Apply filters over chunks (windows) of signal data.                        |
| **Competitive Programming**             | Frequently used for optimal subarray or substring problems.                |

---

## ⚡ 5. Advantages

* **Time efficient**: Reduces nested loops to linear complexity.
* Uses **previous computations**, avoiding redundant work.
* Easily adaptable to many types of problems.

---

## ⚠️ 6. Limitations

* **Only works** with problems involving **contiguous segments**.
* Requires careful **pointer management**.
* May need **additional data structures** (e.g., hash maps, deques) for certain variants.

---

## 🔚 7. Conclusion

The **Sliding Window** technique is a must-have in a programmer’s toolbox. It provides an elegant and efficient way to solve problems 
involving **ranges or substructures** in linear time. Whether it’s processing streaming data, analyzing strings, or solving algorithmic 
challenges, sliding window methods turn brute-force into brilliance.


"""

#=================================================================================================
#Best time to buy and ssell stocks #121

class Solution:
    def maxProfit(self, prices: List[int]) -> int:
        min_price= float('inf')
        max_profit = 0

        for price in prices:
            if price<min_price:
                min_price = price
            else:
                max_profit = max(max_profit, price - min_price)
            
        return max_profit
    
#===========================================================
    #best time to buy stock II  #leetcode 122

    class Solution:
        def maxProfit(self, prices: List[int]) -> int:
            profit = 0

            for i in range(1, len(prices)):
                if prices[i]> prices[i-1]:
                    profit += prices[i] - prices[i-1]

            return profit
    
#=========================================================
#best time to buy stock III  #leetcode 123

class Solution:
    def maxProfit(self, prices: List[int]) -> int:
        n = len(prices)

        if n<=1:
            return 0
        
        left_profit = [0]*n
        min_price = prices[0]

        for i in range(1,n):
            min_price = min(min_price, prices[i])
            left_profit[i] = max(left_profit[i-1], prices[i]-min_price)

        max_price = prices[-1]
        right_profit = 0
        max_total = 0

        for i in  range(n-2, -1, -1):
            max_price = max(max_price, prices[i])
            right_profit = max_price - prices[i]
            max_total = max(max_total, left_profit[i] + right_profit)

        return max_total

#=================================================================

#3. Longest Substring Without Repeating Characters

class Solution:
    def lengthOfLongestSubstring(self, s: str) -> int:
        left = 0
        max_len =0
        seen ={}

        for right in range(len(s)):
            char= s[right]

            if char in seen and seen[char] >= left:
                left = seen[char] +1

            seen[char] = right

            max_len = max(max_len, right - left +1)

        return max_len
    
#==========================================================

#424. Longest Repeating Character Replacement
from collections import defaultdict
class Solution:
    def characterReplacement(self, s: str, k: int) -> int:
        left=0
        max_len = 0
        max_freq = 0
        count = defaultdict(int)

        for right in range (len(s)):
            char = s[right]
            count[char] +=1
            max_freq = max(max_freq, count[char])

            while (right-left+1) - max_freq > k:
                count[s[left]] -=1
                left +=1

            max_len = max(max_len, right-left+1)
        return max_len
    
#==============================================================

#567. Permutation in String

class Solution:
    def checkInclusion(self, s1: str, s2: str) -> bool:
        len_s1 = len(s1)
        len_s2 = len(s2)

        if len_s1 > len_s2:
            return False

        from collections import Counter

        s1_count = Counter(s1)
        window_count = Counter()
        for i in range(len_s1):
            window_count[s2[i]] += 1

        if window_count == s1_count:
            return True

        for i in range(len_s1, len_s2):
            left_char = s2[i-len_s1]
            right_char = s2[i]
            window_count[left_char] -= 1
            if window_count[left_char] == 0:
                del window_count[left_char]

            window_count[right_char] += 1

            if window_count == s1_count:
                return True
        
        return False
    
#==============================================================

#76. Minimum Window Substring

class Solution:
    def minWindow(self, s: str, t: str) -> str:
        from collections import defaultdict

        if not s or not t or len(s) < len(t):
            return ""
        
        t_count = defaultdict(int)
        for char in t:
            t_count[char] += 1

        window_count = defaultdict(int)
        required = len(t_count)
        formed =0
        left = 0
        min_len =float('inf')
        result = (0,0)

        for right in range(len(s)):
            char = s[right]
            window_count[char] += 1

            if char in t_count and window_count[char] == t_count[char]:
                formed +=1

            while formed == required:
                if (right - left +1) < min_len:
                    min_len = right-left+1
                    result = (left, right)
                    
                window_count[s[left]] -= 1
                if s[left] in t_count and window_count[s[left]] < t_count[s[left]]:
                    formed -= 1

                left +=1
            
        return s[result[0] : result[1]+1] if min_len != float('inf') else ""
    
#==============================================================================

