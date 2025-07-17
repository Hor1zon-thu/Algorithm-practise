from typing import List
'''
class Solution:
    def maxSlidingWindow(self, nums: List[int], k: int) -> List[int]:
        from collections import defaultdict
        windows = defaultdict(int)
        right , left = 0,0
        n = len(nums)
        max_num = []

        while right < n:
            c = nums[right]
            right+=1
            windows[c]+=1
            while right - left >= k:
                max_num.append(max(windows.keys()))
                d = nums[left]
                windows[d]-=1


                left+=1
        return max_num
'''
class Solution:
    def maxSlidingWindow(self, nums: List[int], k: int) -> List[int]:
        from collections import deque
        n = len(nums)
        if n==0 or k == 0:
            return []
        if k == 1:
            return nums
        result = []
        window = deque()

        for i in  range(n):
            while window and window[0] <= i -k:
                window.popleft()

            while window and nums[window[-1]]<nums[i]:
                window.pop()

            window.append(i)

            if i >= k -1 :
                result.append(nums[window[0]])

        return result


Test = Solution()
nums = [1,3,-1,-3,5,3,6,7]
k = 3
output = Test.maxSlidingWindow(nums,k)
print(output)