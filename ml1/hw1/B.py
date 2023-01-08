n = int(input())
array = list(map(int, input().split()))
array = sorted(array, key=lambda x: (sum(map(int, str(x))), x))
for i in array:
    print(i, end=" ")
