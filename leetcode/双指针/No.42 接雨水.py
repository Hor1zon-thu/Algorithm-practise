from typing import List
class Solution:
    def trap(self, height: List[int]):
        if not height:
            return 0
        n = len(height)
        left,right = 0,n-1
        area = 0
        max_left = height[left]
        max_right = height[right]
        while left< right:
            if max_left<max_right:
                left += 1
                max_left = max(max_left,height[left])
                area += max_left-height[left]

            else:
                right -=1
                max_right = max(max_right,height[right])
                area += max_right -height[right]
        return area



Test = Solution()
height = [0,1,0,2,1,0,1,3,2,1,2,1]
output = Test.trap(height)
print(output)