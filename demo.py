class Solution:
    def smallestNumber(self, n: int, t: int) -> int:
        while True:
            product = 1
            for digit in str(n):
                print(digit)
                product *= int(digit)
                print(product)
            if product%t == 0:
                print(n)
                return n
            else:
                n += 1
s = Solution()
s.smallestNumber(15,3)
