


#===========================================================================================

#703: Kth largest term in a stream:
import heapq

class KthLargest:

    def __init__(self, k: int, nums: List[int]):
        self.k = k
        self.min_heap = []
        for num in nums:
            self.add(num)

    def add(self, val: int) -> int:
        heapq.heappush(self.min_heap, val)
        if len(self.min_heap) > self.k:
            heapq.heappop(self.min_heap)
        return self.min_heap[0]

#=============================================================================

#1046. Last Stone Weight

class Solution:
    def lastStoneWeight(self, stones: List[int]) -> int:
        max_heap = [-stone for stone in stones]
        heapq.heapify(max_heap)
        while len(max_heap)>1:
            y = -heapq.heappop(max_heap)
            x= -heapq.heappop(max_heap)
            if y != x:
                new_stone = y-x
                heapq.heappush(max_heap, -new_stone)
            
        return -max_heap[0] if max_heap else 0

#===========================================================================================

#973. K Closest Points to Origin

class Solution:
    def kClosest(self, points: List[List[int]], k: int) -> List[List[int]]:
        max_heap = []
        for x,y in points:
            dist = -(x*x + y*y)
            heapq.heappush(max_heap, (dist, [x,y]))
            if len(max_heap) > k:
                heapq.heappop(max_heap)

        return [point for (_, point) in max_heap]

#================================================================================

#215. Kth Largest Element in an Array

class Solution:
    def findKthLargest(self, nums: List[int], k: int) -> int:
        min_heap = []
        for num in nums:
            heapq.heappush(min_heap, num)
            if len(min_heap) > k:
                heapq.heappop(min_heap)
        return min_heap[0]

#============================================================================

#621. Task Scheduler

from collections import Counter
from typing import List

class Solution:
    def leastInterval(self, tasks: List[str], n: int) -> int:
        if n == 0:
            return len(tasks)  # No cooldown needed

        freq = list(Counter(tasks).values())
        max_freq = max(freq)
        max_count = freq.count(max_freq)

        # Minimum time based on scheduling most frequent tasks
        units_needed = (max_freq - 1) * (n + 1) + max_count

        # Either fit into scheduled slots or just do all tasks sequentially
        return max(len(tasks), units_needed)

