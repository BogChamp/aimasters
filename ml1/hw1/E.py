n = int(input())
d = {}
for i in range(n):
    word = input()
    s = tuple(word)
    s = tuple(sorted(s))
    if s in d:
        d[s].append(word)
    else:
        d[s] = [word]

for words in d.values():
    for w in words:
        print(w, end=" ")
    print()
