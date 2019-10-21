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

ax.set_xlabel('Splits')
ax.set_ylabel('Accuracy (%)')
ax.set_xticks(ind+width)
ax.set_xticklabels( ('2', '3', '4', '5') )
plt.ylim(top=100)
plt.title("In Liquid-DNA (split variation)")

ax.legend( (rects1[0], rects2[0], rects3[0]), ('Permutation', 'Multiplication', 'Convolution') )

ax = fig.add_subplot(152)

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

ax.set_xlabel('Splits')
ax.set_ylabel('Accuracy (%)')
ax.set_xticks(ind+width)
ax.set_xticklabels( ('2', '3', '4', '5') )
plt.ylim(top=100)
plt.title("In Liquid-DNA (split variation)")

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

ax.set_xlabel('Splits')
ax.set_ylabel('Accuracy (%)')
ax.set_xticks(ind+width)
ax.set_xticklabels( ('2', '3', '4', '5') )
plt.ylim(top=100)
plt.title("In Liquid-DNA (split variation)")

ax.legend( (rects1[0], rects2[0], rects3[0]), ('Permutation', 'Multiplication', 'Convolution') )

ax = fig.add_subplot(154)

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

ax.set_xlabel('Splits')
ax.set_ylabel('Accuracy (%)')
ax.set_xticks(ind+width)
ax.set_xticklabels( ('2', '3', '4', '5') )
plt.ylim(top=100)
plt.title("In Liquid-DNA (split variation)")

ax.legend( (rects1[0], rects2[0], rects3[0]), ('Permutation', 'Multiplication', 'Convolution') )

ax = fig.add_subplot(155)

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

ax.set_xlabel('Splits')
ax.set_ylabel('Accuracy (%)')
ax.set_xticks(ind+width)
ax.set_xticklabels( ('2', '3', '4', '5') )
plt.ylim(top=100)
plt.title("In Liquid-DNA (split variation)")

ax.legend( (rects1[0], rects2[0], rects3[0]), ('Permutation', 'Multiplication', 'Convolution') )




plt.subplots_adjust(left = 0.035, right = 0.985, top = 0.580, bottom = 0.055, hspace = 0.200, wspace = 0.200)
plt.show()
