import numpy as np
import matplotlib.pyplot as plt

red = '#EA4335'
blue = '#4285F4'
green = '#34A853'
yellow = '#FBBC05'
redc = '#35EA6E'
bluec = '#F48942'
greenc = '#A83C34'
yellowc = '#7C05FB'
fig = plt.figure()

fig.add_subplot(121)
x = range(1,11)
y1 = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
y2 = [100, 90, 80, 70, 60, 50, 40, 30, 20, 10]
y3 = [50, 50, 50, 50, 50, 50, 50, 50, 50, 50]
y4 = [60, 70, 60, 50, 40, 50, 60, 70, 60, 50]
plt.plot(x, y1, marker = 'o', markerfacecolor = redc, markersize = 5, color = red, linewidth = 2)
plt.plot(x, y2, marker = 'o', markerfacecolor = bluec, markersize = 5, color = blue, linewidth = 2)
plt.plot(x, y3, marker = 'o', markerfacecolor = greenc, markersize = 5, color = green, linewidth = 2)
plt.plot(x, y4, marker = 'o', markerfacecolor = yellowc, markersize = 5, color = yellow, linewidth = 2)

fig.add_subplot(122)
x = range(1,11)
y1.reverse()
y2.reverse()
y3.reverse()
y4.reverse()
plt.plot(x, y1, marker = 'o', markerfacecolor = redc, markersize = 5, color = red, linewidth = 2)
plt.plot(x, y2, marker = 'o', markerfacecolor = bluec, markersize = 5, color = blue, linewidth = 2)
plt.plot(x, y3, marker = 'o', markerfacecolor = greenc, markersize = 5, color = green, linewidth = 2)
plt.plot(x, y4, marker = 'o', markerfacecolor = yellowc, markersize = 5, color = yellow, linewidth = 2)

plt.show()
