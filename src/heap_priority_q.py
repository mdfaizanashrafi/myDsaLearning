


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

#=====================================================================================

#355: Design Twitter

from collections import defaultdict
from typing import List
import heapq

class Twitter:

    def __init__(self):
        self.follow_map = defaultdict(set)  # user -> {followees}
        self.tweets = defaultdict(list)      # user -> [(time, tweetId)]
        self.time = 0                       # Lower numbers = newer tweets

    def postTweet(self, userId: int, tweetId: int) -> None:
        self.tweets[userId].append((self.time, tweetId))
        self.time -= 1  # Decrement so newer tweets have lower time values

    def getNewsFeed(self, userId: int) -> List[int]:
        result = []
        max_heap = []

        # Make sure user follows themselves
        self.follow(userId, userId)

        # Push latest tweet from each followee into heap
        for followee in self.follow_map[userId]:
            if self.tweets[followee]:
                index = len(self.tweets[followee]) - 1
                time, tweetId = self.tweets[followee][index]
                heapq.heappush(max_heap, (time, tweetId, followee, index))

        while max_heap and len(result) < 10:
            time, tweetId, followee, index = heapq.heappop(max_heap)
            result.append(tweetId)

            if index > 0:
                new_index = index - 1
                new_time, new_tweetId = self.tweets[followee][new_index]
                heapq.heappush(max_heap, (new_time, new_tweetId, followee, new_index))

        return result

    def follow(self, followerId: int, followeeId: int) -> None:
        self.follow_map[followerId].add(followeeId)

    def unfollow(self, followerId: int, followeeId: int) -> None:
        if followerId in self.follow_map and followeeId in self.follow_map[followerId]:
            self.follow_map[followerId].remove(followeeId)
        
#========================================================================

#295. Find Median from Data Stream
from heapq import heappush, heappop

class MedianFinder:

    def __init__(self):
        self.max_heap = []
        self.min_heap = []

    def addNum(self, num: int) -> None:
        heappush(self.max_heap, -num)
        if self.max_heap and self.min_heap and (-self.max_heap[0] > self.min_heap[0]):
            val = - heappop(self.max_heap)
            heappush(self.min_heap, val)

        if len(self.max_heap) > len(self.min_heap) +1:
            val = -heappop(self.max_heap)
            heappush(self.min_heap, val)

        elif len(self.min_heap) > len(self.max_heap):
            val = heappop(self.min_heap)
            heappush(self.max_heap, -val)

    def findMedian(self) -> float:
        if len(self.max_heap)>len(self.min_heap):
            return -self.max_heap[0]

        else:
            return (-self.max_heap[0]+self.min_heap[0])/2


# Your MedianFinder object will be instantiated and called as such:
# obj = MedianFinder()
# obj.addNum(num)
# param_2 = obj.findMedian()

