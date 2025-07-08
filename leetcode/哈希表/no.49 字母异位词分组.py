from typing import List
class Solution:
    def groupAnagrams(self, strs: List[str]) -> List[List[str]]:
        group = {}
        for s in strs:
            sorted_s = ''.join(sorted(s))
            if sorted_s not in group:
                group[sorted_s] = []
            group[sorted_s].append(s)

        return list(group.values())

Test = Solution()
strs = ["eat", "tea", "tan", "ate", "nat", "bat"]
output = Test.groupAnagrams(strs)
print(output)

