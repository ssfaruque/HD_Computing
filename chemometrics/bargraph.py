import numpy as np
import matplotlib.pyplot as plt

N = 4
ind = np.arange(N)  # the x locations for the groups
width = 0.2       # the width of the bars
red = '#AF0404'
green = '#2B580C'
blue = '#0A3569'
capsize = 3.5
fig = plt.figure()

ax = fig.add_subplot(151)

Pvals = [55.56, 47.00, 53.22, 49.22]
Perror = [8.69, 3.55, 4.20, 3.67]
rects1 = ax.bar(ind, Pvals, width, color=red, yerr=Perror, capsize=capsize)
Mvals = [71.11, 70.44, 71.68, 72.11]
Merror = [2.82, 3.86, 1.98, 1.77]
rects2 = ax.bar(ind+width, Mvals, width, color=green, yerr=Merror, capsize=capsize)
Cvals = [79.22, 80.11, 81.61, 81.56]
Cerror = [5.10, 3.97, 2.52, 3.11]
rects3 = ax.bar(ind+width*2, Cvals, width, color=blue, yerr=Cerror, capsize=capsize)
#rects4 = ax.bar(ind+width*3, kvals, width, color='g')

ax.set_xlabel('Number of splits')
ax.set_ylabel('Accuracy (%)')
ax.set_xticks(ind+width)
ax.set_xticklabels( ('2', '3', '4', '5') )
plt.ylim(top=100)
plt.title("DNA ECOLI")

ax.legend( (rects1[0], rects2[0], rects3[0]), ('Permutation', 'Multiplication', 'Convolution') )

ax = fig.add_subplot(152)

Pvals = [49.22, 52.44, 54.38, 52.44]
Perror = [3.92, 3.70, 1.88, 2.21]
rects1 = ax.bar(ind, Pvals, width, color=red, yerr=Perror, capsize=capsize)
Mvals = [55.33, 56.56, 57.74, 58.56]
Merror = [4.34, 3.16, 3.26, 3.06]
rects2 = ax.bar(ind+width, Mvals, width, color=green, yerr=Merror, capsize=capsize)
Cvals = [63.22, 65.89, 67.92, 67.22]
Cerror = [5.84, 2.46, 3.46, 1.91]
rects3 = ax.bar(ind+width*2, Cvals, width, color=blue, yerr=Cerror, capsize=capsize)
#rects4 = ax.bar(ind+width*3, kvals, width, color='g')

ax.set_xlabel('Number of splits')
ax.set_ylabel('Accuracy (%)')
ax.set_xticks(ind+width)
ax.set_xticklabels( ('2', '3', '4', '5') )
plt.ylim(top=100)
plt.title("DNA Anodisc")

ax.legend( (rects1[0], rects2[0], rects3[0]), ('Permutation', 'Multiplication', 'Convolution') )

ax = fig.add_subplot(153)

Pvals = [86.78, 86.56, 86.37, 87.00]
Perror = [1.52, 2.25, 1.47, 1.17]
rects1 = ax.bar(ind, Pvals, width, color=red, yerr=Perror, capsize=capsize)
Mvals = [84.33, 83.11, 85.22, 84.56]
Merror = [1.77, 3.88, 1.26, 1.84]
rects2 = ax.bar(ind+width, Mvals, width, color=green, yerr=Merror, capsize=capsize)
Cvals = [83.11, 86.22, 84.67, 86.00]
Cerror = [4.08, 2.11, 1.55, 2.04]
rects3 = ax.bar(ind+width*2, Cvals, width, color=blue, yerr=Cerror, capsize=capsize)
#rects4 = ax.bar(ind+width*3, kvals, width, color='g')

ax.set_xlabel('Number of splits')
ax.set_ylabel('Accuracy (%)')
ax.set_xticks(ind+width)
ax.set_xticklabels( ('2', '3', '4', '5') )
plt.ylim(top=100)
plt.title("DNA In-Liquid DNA")

ax.legend( (rects1[0], rects2[0], rects3[0]), ('Permutation', 'Multiplication', 'Convolution') )

ax = fig.add_subplot(154)

Pvals = [42.56, 38.33, 40.34, 37.44]
Perror = [4.48, 3.36, 2.51, 3.23]
rects1 = ax.bar(ind, Pvals, width, color=red, yerr=Perror, capsize=capsize)
Mvals = [48.67, 52.67, 53.19, 54.78]
Merror = [6.50, 5.06, 3.95, 4.48]
rects2 = ax.bar(ind+width, Mvals, width, color=green, yerr=Merror, capsize=capsize)
Cvals = [49.78, 46.11, 44.82, 45.00]
Cerror = [5.56, 3.20, 3.50, 1.31]
rects3 = ax.bar(ind+width*2, Cvals, width, color=blue, yerr=Cerror, capsize=capsize)
#rects4 = ax.bar(ind+width*3, kvals, width, color='g')

ax.set_xlabel('Number of splits')
ax.set_ylabel('Accuracy (%)')
ax.set_xticks(ind+width)
ax.set_xticklabels( ('2', '3', '4', '5') )
plt.ylim(top=100)
plt.title("Yeast In-Liquid HK")

ax.legend( (rects1[0], rects2[0], rects3[0]), ('Permutation', 'Multiplication', 'Convolution') )

ax = fig.add_subplot(155)

Pvals = [56.22, 56.44, 61.14, 54.56]
Perror = [4.45, 1.72, 3.02, 2.19]
rects1 = ax.bar(ind, Pvals, width, color=red, yerr=Perror, capsize=capsize)
Mvals = [71.44, 72.00, 72.31, 72.22]
Merror = [1.74, 0.87, 0.60, 0.74]
rects2 = ax.bar(ind+width, Mvals, width, color=green, yerr=Merror, capsize=capsize)
Cvals = [74.44, 72.11, 72.60, 72.11]
Cerror = [2.57, 0.97, 0.71, 1.10]
rects3 = ax.bar(ind+width*2, Cvals, width, color=blue, yerr=Cerror, capsize=capsize)
#rects4 = ax.bar(ind+width*3, kvals, width, color='g')

ax.set_xlabel('Number of splits')
ax.set_ylabel('Accuracy (%)')
ax.set_xticks(ind+width)
ax.set_xticklabels( ('2', '3', '4', '5') )
plt.ylim(top=100)
plt.title("Yeast In-Liquid Live")

ax.legend( (rects1[0], rects2[0], rects3[0]), ('Permutation', 'Multiplication', 'Convolution') )

plt.subplots_adjust(left = 0.035, right = 0.985, top = 0.920, bottom = 0.395, hspace = 0.200, wspace = 0.200)
fig.suptitle('Splits Accuracies')
plt.show()
