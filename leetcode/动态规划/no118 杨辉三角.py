from typing import List
class Solution:
    def generate(self, numRows: int) -> List[List[int]]:
        if numRows == 0 :
            return []

        result  = [[1]]
        for i in range(1,numRows):
            prev_row = result[i-1]
            curr_row = [1]
            for j in range (1,i):
                curr_row.append(prev_row[j-1] + prev_row[j])
            curr_row.append(1)
            result.append(curr_row)
        return result
