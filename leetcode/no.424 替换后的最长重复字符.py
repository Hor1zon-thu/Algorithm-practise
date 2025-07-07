from typing import List
class Solution:
    def characterReplacement(self, s: str, k: int) -> int:
        from collections import Counter
        left = 0
        max_length = 0
        n = len(s)
        count = Counter()
        freq = 0
        for right in range(n):
            count[s[right]] += 1
            freq = max(freq,count[s[right]])
            window_size = right-left+1
            if window_size - freq > k :#窗口大小减去出现次数最多的字幕大于k
                count[s[left]] -= 1
                left += 1
            else:
                max_length = max(max_length,right-left+1)

        return  max_length

Test = Solution()
s = "AABABBA"
k = 1
output = Test.characterReplacement(s,k)
print (output)