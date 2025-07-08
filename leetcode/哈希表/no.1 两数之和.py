from typing import List
class Solution:
    def twoSum(self, nums: List[int], target: int) -> List[int]:
        num2index ={}
        for i ,num in enumerate(nums):
            comlement = target - num
            if comlement in num2index:
                return [num2index[comlement],i]
            num2index[num] = i

        return []
    #时间复杂度O(N)


Test = Solution()
nums = [2, 7, 11, 15]
target = 9
output = Test.twoSum(nums, target)
print(output)


'''
第一次暴力求解，时间复杂度O(N²)
class Solution(object):
    def twoSum(self, nums, target):
        n = len(nums)
        for i in range(n):
            for j in range(i+1,n):
                if nums[i] + nums[j]  == target :
                    return [i, j]
        return []

'''