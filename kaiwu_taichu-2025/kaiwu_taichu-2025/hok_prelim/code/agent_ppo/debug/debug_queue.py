from collections import deque

q = deque(maxlen=10)
for i in range(20):
    q.append(i)
    print(q[0])
    print(list(q))

