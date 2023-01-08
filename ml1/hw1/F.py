one = tuple(map(lambda x: tuple(x.lower()), input().split()))
one = tuple(map(lambda x: tuple(sorted(x)), one))
one = tuple(sorted(one))
two = tuple(map(lambda x: tuple(x.lower()), input().split()))
two = tuple(map(lambda x: tuple(sorted(x)), two))
two = tuple(sorted(two))

if set(two).issubset(set(one)):
    print("YES")
else:
    print("NO")
