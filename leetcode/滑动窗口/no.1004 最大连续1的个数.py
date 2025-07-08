from typing import List
class Solution:
    def longestOnes(self, nums: List[int], k: int) -> int:
        left = 0
        max_length = 0
        n = len(nums)
        zero_count = 0
        for right in range(n):
            if nums[right] == 0:
                zero_count+=1
            while zero_count > k:
                if nums[left] == 0:
                    zero_count -= 1
                left += 1
            max_length = max(max_length, right - left + 1)
        return max_length
    #时间复杂度O(N)


Test = Solution()
nums = [1,1,1,0,0,0,1,1,1,1,0]
K = 2
output = Test.longestOnes(nums,K)
print (output)