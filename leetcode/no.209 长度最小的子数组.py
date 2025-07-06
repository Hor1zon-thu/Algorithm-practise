
from typing import List
'''
#遍历，时间复杂度O(N²)，超出运行时间限制
class Solution:
    def minSubArrayLen(self, target: int, nums: List[int]) -> int:
        n = len(nums)
        output  = n+1
        for start in range(n):
            sum = 0
            for end in range (start,n):
                sum += nums[end]
                if sum >= target:
                    output =min(output, end - start + 1)
                    break
        if output > n:
            output = 0
        return output
'''

class Solution:
    def minSubArrayLen(self,target:int,nums:List[int])-> int:
        n = len(nums)
        right,left = 0,0
        current_sum = 0
        min_length = float('inf')
        for right in range (n):
            current_sum += nums[right]
            while current_sum >= target:
                min_length = min(min_length,right - left+1)
                current_sum -= nums[left]
                left += 1

        return 0 if min_length == float('inf') else min_length

#时间复杂度O(N)





s =10
num =[2,3,1,2,4,3]
SOL = Solution()
output = SOL.minSubArrayLen(target=s,nums=num)

print(output)
