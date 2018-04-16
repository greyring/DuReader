import re
import matplotlib.pyplot as plt

x = []
y = []
length = 0
with open('nohup.out', 'r') as fi:
    for line in fi:
        los = re.findall(r'Average loss from batch \d* to \d* is .*', line)
        if len(los):
            loss = float(los[0].strip().split()[-1])
            x.append(length + 1)
            y.append(loss)
            length += 1


plt.figure()
plt.plot(x, y)
plt.show() 
