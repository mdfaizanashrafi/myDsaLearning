
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
