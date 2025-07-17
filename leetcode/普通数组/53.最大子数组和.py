from typing import List
class Solution:
    def maxSubArray(self, nums: List[int]) -> int:
        current_sum = 0

        max_sum = float('-inf')
        min_sum = 0
        for num in nums:
            current_sum += num
            max_sum = max(max_sum, current_sum - min_sum)
            min_sum = min(min_sum, current_sum)

        return max_sum
Test = Solution()
nums = [-2,1,-3,4,-1,2,1,-5,4]
output = Test.maxSubArray(nums)
print(output)