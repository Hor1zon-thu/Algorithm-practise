from typing import List
class Solution:
    def maxArea(self, height: List[int]) -> int:
        maxarea = 0
        i = 0
        j = len(height)-1
        while i !=j:
            area = (j-i)*min(height[i],height[j])
            maxarea = max(area,maxarea)
            if height[i] <= height[j]:
                i+=1
            else:
                j-=1

        return maxarea

height = [1,8,6,2,5,4,8,3,7]
Test = Solution()
maxarea = Test.maxArea(height)
print(maxarea)