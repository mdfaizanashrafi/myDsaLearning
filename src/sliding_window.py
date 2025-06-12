
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
    
    #====================================================
