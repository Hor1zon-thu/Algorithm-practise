class Solution:
    def lengthOfLongestSubstring(self, s: str) -> int:
        n = len(s)
        left = 0
        max_length = 0
        char_set = set()  # 使用集合来快速检查字符是否存在

        for right in range(n):
            while s[right] in char_set:
                char_set.remove(s[left])
                left += 1

            char_set.add(s[right])

            current_length = right - left + 1
            max_length = max(max_length, current_length)

        return max_length



test = Solution()
s = ("abca")
output = test.lengthOfLongestSubstring(s)
print(output)
