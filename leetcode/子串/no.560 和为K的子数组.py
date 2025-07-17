from typing import List
'''
class Solution:
    def subarraySum(self, nums: List[int], k: int) -> int:
        if k == 0 :
            return 0
        left,right=0,0
        n = len(nums)
        nums_sum = 0
        all = 0
        while right < n :
            nums_sum += nums[right]
            right += 1
            while nums_sum >= k:
                if nums_sum == k:
                    all +=1
                nums_sum -= nums[left]
                left += 1


        return all
'''
class Solution:
    def subarraySum(self, nums: List[int], k: int) -> int:
        from collections import defaultdict
        sum_dict =defaultdict(int)
        count = 0
        current_sum = 0
        sum_dict[0] = 1

        for i ,num in enumerate(nums):
            current_sum += num
            count += sum_dict.get(current_sum-k,0)
            sum_dict[current_sum]+=1
        return count


Test = Solution()
nums = [1,2,3]
k = 5
output = Test.subarraySum(nums,k)
print(output)
