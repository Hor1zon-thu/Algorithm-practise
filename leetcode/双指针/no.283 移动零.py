from typing import List
'''
class Solution:
    def moveZeroes(self, nums: List[int]) -> None:
        """
        Do not return anything, modify nums in-place instead.
        """
        zero_num = nums.count(0)
        while 0 in nums:
            nums.remove(0)
        nums.extend([0] * zero_num)
'''

class Solution:
    def moveZeroes(self, nums: List[int]) -> None:
        """
        Do not return anything, modify nums in-place instead.
        """
        i = 0
        for j in range(len(nums)):
            if nums[j]!=0:
                nums[i],nums[j] = nums[j],nums[i]
                i += 1




Test = Solution()
nums = [0,1,0,3,12]
Test.moveZeroes(nums)
print(nums)
