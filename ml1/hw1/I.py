def five_part(left, r, size):
    step = (r - left + 1) // (size + 1)
    four = [0 for _ in range(size)]
    for i in range(left + step, r, step):
        print("? " + str(i), flush=True)
    if size == 3:
        print("?", r, flush=True)
    print("+", flush=True)
    for i in range(size):
        four[i] = int(input())
    for i in four:
        if i == 0:
            left += step
        else:
            r -= step
    return left, r


left = 1
r = 100000
left, r = five_part(left, r, 4)
left, r = five_part(left, r, 4)
left, r = five_part(left, r, 4)
left, r = five_part(left, r, 4)
left, r = five_part(left, r, 4)
left, r = five_part(left, r, 7)
left, r = five_part(left, r, 3)
print("!", left, flush=True)
