from typing import List
class Solution:
    def findAnagrams(self, s: str, p: str) -> List[int]:
        from collections import defaultdict
        need = defaultdict(int)
        for c in p:
            need[c] += 1
        need_len = len(need)
        out = []
        windows = defaultdict(int)
        valid = 0
        left,right = 0,0
        while right < len(s):
            c = s[right]
            right+=1
            if c in need:
                windows[c]+=1
                if windows[c] == need[c]:
                    valid += 1
            while right - left >= len(p):
                if valid == need_len:
                    out.append(left)
                d = s[left]
                left +=1
                if d in need:
                    if windows[d]==need[d]:
                        valid -=1
                    windows[d]-=1


        return out

Test = Solution()
s = "cbaebabacd"
p = "abc"
output= Test.findAnagrams(s =s,p =p)
print(output)