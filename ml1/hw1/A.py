n = int(input())
array = list(map(int, input().split()))
dups = set()
num_dups = 0
for i in array:
    if i not in dups:
        print(i, end=" ")
        dups.add(i)
    else:
        num_dups += 1
print()
print(num_dups)
