from typing import  List
'''
class Solution:
    def threeSum(self, nums: List[int]) -> List[List[int]]:
        num2index = {}
        n= len(nums)
        output = []
        for i in range(n):
            for j in range(i,n):
                complement = -(nums[i]+nums[j])
                if complement in num2index:
                    output.append([complement,nums[i],nums[j]])
                num2index[nums[i]] = i
                num2index[nums[j]] = j
            return output
        return []
'''

'''
class Solution:
    def threeSum(self, nums: List[int]) -> List[List[int]]:
        output = set()  
        nums.sort()  

        for i in range(len(nums)):
            if i > 0 and nums[i] == nums[i - 1]:
                continue

            target = -nums[i]
            seen = {}  

            for j in range(i + 1, len(nums)):
                complement = target - nums[j]

                if complement in seen:
                    triplet = (nums[i], complement, nums[j])
                    output.add(triplet)

                seen[nums[j]] = j

        return [list(t) for t in output]
'''
class Solution:
    def threeSum(self, nums: List[int]) -> List[List[int]]:
        nums.sort()
        result = []
        n = len(nums)
        for i in range(n-2):
            if i >0 and nums[i]==nums[i-1]:
                continue
            target = - nums[i]
            left,right =i+1,n-1
            while left < right :
                current_sum =nums[left]+nums[right]
                if current_sum == target:
                    result.append([nums[i],nums[left],nums[right]])
                    while left< right and nums[left] == nums[left + 1]:
                        left += 1
                    while left < right and nums[right] == nums[right - 1]:
                        right -= 1
                    left += 1
                    right -= 1
                elif current_sum < target:
                    left += 1
                else:
                    right-=1

        return result


Test = Solution()
nums = [-1,0,1,2,-1,-4]
output = Test.threeSum(nums)
print(output)