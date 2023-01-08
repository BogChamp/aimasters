one = tuple(map(lambda x: tuple(x.lower()), input().split()))
one = tuple(map(lambda x: tuple(sorted(x)), one))

two = tuple(map(lambda x: tuple(x.lower()), input().split()))
two = tuple(map(lambda x: tuple(sorted(x)), two))

d_one = {}
d_two = {}

for i in one:
    if i in d_one:
        d_one[i] += 1
    else:
        d_one[i] = 1

for i in two:
    if i in d_two:
        d_two[i] += 1
    else:
        d_two[i] = 1
msg = ""
if set(two).issubset(set(one)):
    for j in d_two:
        if d_two[j] > d_one[j]:
            msg = "NO"
            break
    if msg == "":
        msg = "YES"
else:
    msg = "NO"
print(msg)
