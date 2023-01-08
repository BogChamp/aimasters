n, k = input().split()
k = int(k)
res = 0
for i in range(1, k + 1):
    res += int(i * n)
print(res)
