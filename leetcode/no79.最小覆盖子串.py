class Solution:
    def minWindow(self,s:str,t:str)-> str:
        from collections import  defaultdict
        need = defaultdict(int)
        for c in t :
            need[c] += 1
        need_len = len(need)

        window = defaultdict(int)
        valid = 0

        left,right =0,0
        min_len = float('inf')
        start = 0
        while right < len(s):
            c = s[right]
            right += 1
            if c in need:
                window[c]+=1
                if window[c]==need[c]:
                    valid +=1

            while valid == need_len:
                current_len = right -left
                if current_len <min_len:
                    min_len = current_len
                    start = left

                d= s[left]
                left += 1
                if d in need:
                    if window[d] == need [d]:
                        valid -= 1
                    window[d]-= 1
        return "" if min_len == float('inf') else s[start:start+min_len]