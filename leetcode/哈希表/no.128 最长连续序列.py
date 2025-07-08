from typing import List
'''
class Solution:
    def longestConsecutive(self, nums: List[int]) -> int:
        set_num = set(nums)
        sorted_num = sorted(set_num)
        left = 0
        max_length = 0
        for right in range(len(sorted_num)):
            if sorted_num[right] == sorted_num[right-1] +1 :
                max_length = max(max_length,right -left+1)
            else:
                left = right
                max_length = max(max_length,right -left+1)
        return  max_length
    '''
class Solution:
    def longestConsecutive(self, nums: List[int]) -> int:
        if not nums:
            return  0
        sorted_num = sorted(set(nums))
        n = len(sorted_num)
        max_length = 1
        current_length = 1
        for i in range (1,n):
           if sorted_num[i] == sorted_num[i-1] + 1:
               current_length +=1
           else:
               current_length = 1
           max_length = max(max_length,current_length)
        return  max_length
#时间复杂度 O(n log n)


class Best_Solution:
    def longestConsecutive(self, nums: List[int]) -> int:
        num_set = set(nums)
        max_length = 0
        for num in num_set:
            if num - 1 not in num_set:
                current_num = num
                current_length = 1

                while current_num +1 in num_set:
                    current_num += 1
                    current_length += 1
                max_length = max(max_length,current_length)
        return  max_length



Test = Solution()
nums = [100,4,200,1,3,2]
output = Test.longestConsecutive(nums)
print(output)

