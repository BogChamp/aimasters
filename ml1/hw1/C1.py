n = float(input())
array = [0,0,0,0,0,0,0,0]
d = {0:10, 1:5, 2:2, 3:1, 4:0.5, 5:0.1, 6:0.05, 7:0.01}
while n > 10:
    array[0] += 1
    n -= 10
while n > 5:
    array[1] += 1
    n -= 5
while n > 2:
    array[2] += 1
    n -= 2
while n > 1:
    array[3] += 1
    n -= 1
while n > 0.5:
    array[4] += 1
    n -= 0.5
while n > 0.1:
    array[5] += 1
    n -= 0.1
while n > 0.05:
    array[6] += 1
    n -= 0.05
while n > 0.01:
    array[7] += 1
    n -= 0.01

for i in range(8):
    if array[i] > 0:
        print(str(format(d[i], '5.2f'))+"\t"+str(int(array[i])))