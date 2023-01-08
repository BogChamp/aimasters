n = float(input())
array = [0, 0, 0, 0, 0, 0, 0, 0]
d = {0: 10, 1: 5, 2: 2, 3: 1, 4: 0.5, 5: 0.1, 6: 0.05, 7: 0.01}
array[0] = n // 10
n -= array[0] * 10
array[1] = n // 5
n -= array[1] * 5
array[2] = n // 2
n -= array[2] * 2
array[3] = n // 1
n -= array[3]
n *= 100
array[4] = n // 50
n -= array[4] * 50
array[5] = n // 10
n -= array[5] * 10
array[6] = n // 5
n -= array[6] * 5
array[7] = n // 1
for i in range(8):
    if array[i] > 0:
        print(str(format(d[i], '5.2f')) + "\t" + str(int(array[i])))
