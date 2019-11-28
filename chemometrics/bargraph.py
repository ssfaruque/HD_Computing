import numpy as np
import matplotlib.pyplot as plt

red = '#AF0404'
green = '#2B580C'
blue = '#0A3569'
capsize = 3.5
width = 0.2       # the width of the bars

#---------------Limited Data F1s---------------

N = 3
ind = np.arange(N)  # the x locations for the groups

fig0 = plt.figure()

ax = fig0.add_subplot(151)

Pvals = [0.71, 0.65, 0.61]
Perror = [0.07, 0.07, 0.06]
rects1 = ax.bar(ind, Pvals, width, color=red, yerr=Perror, capsize=capsize)
Mvals = [0.79, 0.74, 0.81]
Merror = [0.08, 0.08, 0.04]
rects2 = ax.bar(ind+width, Mvals, width, color=green, yerr=Merror, capsize=capsize)
Cvals = [0.78, 0.87, 0.84]
Cerror = [0.05, 0.08, 0.04]
rects3 = ax.bar(ind+width*2, Cvals, width, color=blue, yerr=Cerror, capsize=capsize)

ax.set_xlabel('Number of samples')
ax.set_ylabel('F1-score')
ax.set_xticks(ind+width)
ax.set_xticklabels( ('1', '3', '5') )
plt.ylim(top=1)
plt.title("DNA E.coli")

ax.legend( (rects1[0], rects2[0], rects3[0]), ('Permutation', 'Multiplication', 'Convolution') )

ax = fig0.add_subplot(152)

Pvals = [0.83, 0.87, 0.83]
Perror = [0.10, 0.04, 0.05]
rects1 = ax.bar(ind, Pvals, width, color=red, yerr=Perror, capsize=capsize)
Mvals = [0.85, 0.83, 0.86]
Merror = [0.07, 0.04, 0.06]
rects2 = ax.bar(ind+width, Mvals, width, color=green, yerr=Merror, capsize=capsize)
Cvals = [0.87, 0.87, 0.87]
Cerror = [0.08, 0.07, 0.04]
rects3 = ax.bar(ind+width*2, Cvals, width, color=blue, yerr=Cerror, capsize=capsize)

ax.set_xlabel('Number of samples')
ax.set_ylabel('F1-score')
ax.set_xticks(ind+width)
ax.set_xticklabels( ('1', '3', '5') )
plt.ylim(top=1)
plt.title("DNA Anodisc")

ax.legend( (rects1[0], rects2[0], rects3[0]), ('Permutation', 'Multiplication', 'Convolution') )

ax = fig0.add_subplot(153)

Pvals = [0.93, 0.98, 0.96]
Perror = [0.06, 0.02, 0.02]
rects1 = ax.bar(ind, Pvals, width, color=red, yerr=Perror, capsize=capsize)
Mvals = [0.90, 0.90, 0.91]
Merror = [0.05, 0.05, 0.04]
rects2 = ax.bar(ind+width, Mvals, width, color=green, yerr=Merror, capsize=capsize)
Cvals = [0.91, 0.95, 0.95]
Cerror = [0.07, 0.02, 0.04]
rects3 = ax.bar(ind+width*2, Cvals, width, color=blue, yerr=Cerror, capsize=capsize)

ax.set_xlabel('Number of samples')
ax.set_ylabel('F1-score')
ax.set_xticks(ind+width)
ax.set_xticklabels( ('1', '3', '5') )
plt.ylim(top=1)
plt.title("DNA In-Liquid DNA")

ax.legend( (rects1[0], rects2[0], rects3[0]), ('Permutation', 'Multiplication', 'Convolution') )

ax = fig0.add_subplot(154)

Pvals = [0.76, 0.79, 0.78]
Perror = [0.09, 0.02, 0.03]
rects1 = ax.bar(ind, Pvals, width, color=red, yerr=Perror, capsize=capsize)
Mvals = [0.79, 0.79, 0.81]
Merror = [0.06, 0.05, 0.02]
rects2 = ax.bar(ind+width, Mvals, width, color=green, yerr=Merror, capsize=capsize)
Cvals = [0.82, 0.81, 0.79]
Cerror = [0.04, 0.04, 0.06]
rects3 = ax.bar(ind+width*2, Cvals, width, color=blue, yerr=Cerror, capsize=capsize)

ax.set_xlabel('Number of samples')
ax.set_ylabel('F1-score')
ax.set_xticks(ind+width)
ax.set_xticklabels( ('1', '3', '5') )
plt.ylim(top=1)
plt.title("Yeast In-Liquid HK")

ax.legend( (rects1[0], rects2[0], rects3[0]), ('Permutation', 'Multiplication', 'Convolution') )

ax = fig0.add_subplot(155)

Pvals = [0.73, 0.77, 0.81]
Perror = [0.08, 0.09, 0.10]
rects1 = ax.bar(ind, Pvals, width, color=red, yerr=Perror, capsize=capsize)
Mvals = [0.81, 0.82, 0.86]
Merror = [0.08, 0.05, 0.04]
rects2 = ax.bar(ind+width, Mvals, width, color=green, yerr=Merror, capsize=capsize)
Cvals = [0.83, 0.84, 0.84]
Cerror = [0.08, 0.06, 0.08]
rects3 = ax.bar(ind+width*2, Cvals, width, color=blue, yerr=Cerror, capsize=capsize)

ax.set_xlabel('Number of samples')
ax.set_ylabel('F1-score')
ax.set_xticks(ind+width)
ax.set_xticklabels( ('1', '3', '5') )
plt.ylim(top=1)
plt.title("Yeast In-Liquid Live")

ax.legend( (rects1[0], rects2[0], rects3[0]), ('Permutation', 'Multiplication', 'Convolution') )

plt.subplots_adjust(left = 0.035, right = 0.985, top = 0.920, bottom = 0.395, hspace = 0.200, wspace = 0.200)
fig0.suptitle('Limited Data F1s')

#---------------Limited Data Accuracies---------------

fig1 = plt.figure()

ax = fig1.add_subplot(151)

Pvals = [48.47, 41.73, 50.77]
Perror = [10.42, 10.98, 7.00]
rects1 = ax.bar(ind, Pvals, width, color=red, yerr=Perror, capsize=capsize)
Mvals = [61.64, 62.53, 72.46]
Merror = [9.01, 6.48, 4.84]
rects2 = ax.bar(ind+width, Mvals, width, color=green, yerr=Merror, capsize=capsize)
Cvals = [62.23, 75.20, 75.38]
Cerror = [1.00, 8.62, 7.32]
rects3 = ax.bar(ind+width*2, Cvals, width, color=blue, yerr=Cerror, capsize=capsize)

ax.set_xlabel('Number of samples')
ax.set_ylabel('Accuracy (%)')
ax.set_xticks(ind+width)
ax.set_xticklabels( ('1', '3', '5') )
plt.ylim(top=100)
plt.title("DNA E.coli")

ax.legend( (rects1[0], rects2[0], rects3[0]), ('Permutation', 'Multiplication', 'Convolution') )

ax = fig1.add_subplot(152)

Pvals = [46.35, 52.13, 54.46]
Perror = [11.71, 7.99, 6.20]
rects1 = ax.bar(ind, Pvals, width, color=red, yerr=Perror, capsize=capsize)
Mvals = [48.11, 55.20, 55.23]
Merror = [11.26, 6.41, 6.63]
rects2 = ax.bar(ind+width, Mvals, width, color=green, yerr=Merror, capsize=capsize)
Cvals = [53.65, 56.93, 66.15]
Cerror = [10.15, 5.34, 3.55]
rects3 = ax.bar(ind+width*2, Cvals, width, color=blue, yerr=Cerror, capsize=capsize)

ax.set_xlabel('Number of samples')
ax.set_ylabel('Accuracy (%)')
ax.set_xticks(ind+width)
ax.set_xticklabels( ('1', '3', '5') )
plt.ylim(top=100)
plt.title("DNA Anodisc")

ax.legend( (rects1[0], rects2[0], rects3[0]), ('Permutation', 'Multiplication', 'Convolution') )

ax = fig1.add_subplot(153)

Pvals = [73.76, 83.87, 82.62]
Perror = [7.94, 6.11, 2.06]
rects1 = ax.bar(ind, Pvals, width, color=red, yerr=Perror, capsize=capsize)
Mvals = [75.53, 71.87, 80.15]
Merror = [6.12, 11.02, 6.09]
rects2 = ax.bar(ind+width, Mvals, width, color=green, yerr=Merror, capsize=capsize)
Cvals = [69.53, 83.20, 80.92]
Cerror = [13.26, 3.40, 5.72]
rects3 = ax.bar(ind+width*2, Cvals, width, color=blue, yerr=Cerror, capsize=capsize)

ax.set_xlabel('Number of samples')
ax.set_ylabel('Accuracy (%)')
ax.set_xticks(ind+width)
ax.set_xticklabels( ('1', '3', '5') )
plt.ylim(top=100)
plt.title("DNA In-Liquid DNA")

ax.legend( (rects1[0], rects2[0], rects3[0]), ('Permutation', 'Multiplication', 'Convolution') )

ax = fig1.add_subplot(154)

Pvals = [56.71, 49.73, 47.08]
Perror = [11.96, 5.62, 8.52]
rects1 = ax.bar(ind, Pvals, width, color=red, yerr=Perror, capsize=capsize)
Mvals = [58.82, 54.80, 48.77]
Merror = [7.21, 10.01, 13.33]
rects2 = ax.bar(ind+width, Mvals, width, color=green, yerr=Merror, capsize=capsize)
Cvals = [64.11, 57.73, 54.62]
Cerror = [5.36, 7.51, 10.42]
rects3 = ax.bar(ind+width*2, Cvals, width, color=blue, yerr=Cerror, capsize=capsize)

ax.set_xlabel('Number of samples')
ax.set_ylabel('Accuracy (%)')
ax.set_xticks(ind+width)
ax.set_xticklabels( ('1', '3', '5') )
plt.ylim(top=100)
plt.title("Yeast In-Liquid HK")

ax.legend( (rects1[0], rects2[0], rects3[0]), ('Permutation', 'Multiplication', 'Convolution') )

ax = fig1.add_subplot(155)

Pvals = [50.11, 57.20, 62.30]
Perror = [14.21, 12.72, 12.11]
rects1 = ax.bar(ind, Pvals, width, color=red, yerr=Perror, capsize=capsize)
Mvals = [65.76, 71.47, 75.38]
Merror = [12.14, 5.49, 4.17]
rects2 = ax.bar(ind+width, Mvals, width, color=green, yerr=Merror, capsize=capsize)
Cvals = [70.82, 72.13, 71.85]
Cerror = [8.48, 6.45, 6.65]
rects3 = ax.bar(ind+width*2, Cvals, width, color=blue, yerr=Cerror, capsize=capsize)

ax.set_xlabel('Number of samples')
ax.set_ylabel('Accuracy (%)')
ax.set_xticks(ind+width)
ax.set_xticklabels( ('1', '3', '5') )
plt.ylim(top=100)
plt.title("Yeast In-Liquid Live")

ax.legend( (rects1[0], rects2[0], rects3[0]), ('Permutation', 'Multiplication', 'Convolution') )

plt.subplots_adjust(left = 0.035, right = 0.985, top = 0.920, bottom = 0.395, hspace = 0.200, wspace = 0.200)
fig1.suptitle('Limited Data Accuracies')

#---------------White Gaussian Noise F1s---------------

fig2 = plt.figure()

ax = fig2.add_subplot(151)

Pvals = [0.72, 0.58, 0.72]
Perror = [0.03, 0.07, 0.03]
rects1 = ax.bar(ind, Pvals, width, color=red, yerr=Perror, capsize=capsize)
Mvals = [0.53, 0.53, 0.61]
Merror = [0.04, 0.03, 0.03]
rects2 = ax.bar(ind+width, Mvals, width, color=green, yerr=Merror, capsize=capsize)
Cvals = [0.84, 0.74, 0.72]
Cerror = [0.02, 0.03, 0.03]
rects3 = ax.bar(ind+width*2, Cvals, width, color=blue, yerr=Cerror, capsize=capsize)

ax.set_xlabel('Standard deviation (\u03C3)')
ax.set_ylabel('F1-score')
ax.set_xticks(ind+width)
ax.set_xticklabels( ('0.05', '0.10', '0.15') )
plt.ylim(top=1)
plt.title("DNA E.coli")

ax.legend( (rects1[0], rects2[0], rects3[0]), ('Permutation', 'Multiplication', 'Convolution') )

ax = fig2.add_subplot(152)

Pvals = [0.80, 0.82, 0.74]
Perror = [0.02, 0.03, 0.02]
rects1 = ax.bar(ind, Pvals, width, color=red, yerr=Perror, capsize=capsize)
Mvals = [0.70, 0.83, 0.79]
Merror = [0.04, 0.01, 0.02]
rects2 = ax.bar(ind+width, Mvals, width, color=green, yerr=Merror, capsize=capsize)
Cvals = [0.90, 0.88, 0.90]
Cerror = [0.02, 0.02, 0.02]
rects3 = ax.bar(ind+width*2, Cvals, width, color=blue, yerr=Cerror, capsize=capsize)

ax.set_xlabel('Standard deviation (\u03C3)')
ax.set_ylabel('F1-score')
ax.set_xticks(ind+width)
ax.set_xticklabels( ('0.05', '0.10', '0.15') )
plt.ylim(top=1)
plt.title("DNA Anodisc")

ax.legend( (rects1[0], rects2[0], rects3[0]), ('Permutation', 'Multiplication', 'Convolution') )

ax = fig2.add_subplot(153)

Pvals = [0.71, 0.73, 0.60]
Perror = [0.03, 0.03, 0.04]
rects1 = ax.bar(ind, Pvals, width, color=red, yerr=Perror, capsize=capsize)
Mvals = [0.78, 0.73, 0.61]
Merror = [0.02, 0.03, 0.03]
rects2 = ax.bar(ind+width, Mvals, width, color=green, yerr=Merror, capsize=capsize)
Cvals = [0.83, 0.84, 0.67]
Cerror = [0.02, 0.01, 0.04]
rects3 = ax.bar(ind+width*2, Cvals, width, color=blue, yerr=Cerror, capsize=capsize)

ax.set_xlabel('Standard deviation (\u03C3)')
ax.set_ylabel('F1-score')
ax.set_xticks(ind+width)
ax.set_xticklabels( ('0.05', '0.10', '0.15') )
plt.ylim(top=1)
plt.title("DNA In-Liquid DNA")

ax.legend( (rects1[0], rects2[0], rects3[0]), ('Permutation', 'Multiplication', 'Convolution') )

ax = fig2.add_subplot(154)

Pvals = [0.71, 0.69, 0.69]
Perror = [0.03, 0.01, 0.01]
rects1 = ax.bar(ind, Pvals, width, color=red, yerr=Perror, capsize=capsize)
Mvals = [0.75, 0.59, 0.60]
Merror = [0.01, 0.04, 0.03]
rects2 = ax.bar(ind+width, Mvals, width, color=green, yerr=Merror, capsize=capsize)
Cvals = [0.78, 0.75, 0.65]
Cerror = [0.02, 0.02, 0.03]
rects3 = ax.bar(ind+width*2, Cvals, width, color=blue, yerr=Cerror, capsize=capsize)

ax.set_xlabel('Standard deviation (\u03C3)')
ax.set_ylabel('F1-score')
ax.set_xticks(ind+width)
ax.set_xticklabels( ('0.05', '0.10', '0.15') )
plt.ylim(top=1)
plt.title("Yeast In-Liquid HK")

ax.legend( (rects1[0], rects2[0], rects3[0]), ('Permutation', 'Multiplication', 'Convolution') )

ax = fig2.add_subplot(155)

Pvals = [0.73, 0.75, 0.69]
Perror = [0.02, 0.02, 0.03]
rects1 = ax.bar(ind, Pvals, width, color=red, yerr=Perror, capsize=capsize)
Mvals = [0.70, 0.60, 0.59]
Merror = [0.02, 0.03, 0.05]
rects2 = ax.bar(ind+width, Mvals, width, color=green, yerr=Merror, capsize=capsize)
Cvals = [0.77, 0.73, 0.72]
Cerror = [0.03, 0.03, 0.04]
rects3 = ax.bar(ind+width*2, Cvals, width, color=blue, yerr=Cerror, capsize=capsize)

ax.set_xlabel('Standard deviation (\u03C3)')
ax.set_ylabel('F1-score')
ax.set_xticks(ind+width)
ax.set_xticklabels( ('0.05', '0.10', '0.15') )
plt.ylim(top=1)
plt.title("Yeast In-Liquid Live")

ax.legend( (rects1[0], rects2[0], rects3[0]), ('Permutation', 'Multiplication', 'Convolution') )

plt.subplots_adjust(left = 0.035, right = 0.985, top = 0.920, bottom = 0.395, hspace = 0.200, wspace = 0.200)
fig2.suptitle('White Gaussian Noise F1s')

#---------------White Gaussian Noise Accuracies---------------

fig3 = plt.figure()

ax = fig3.add_subplot(151)

Pvals = [39.66, 31.55, 36.47]
Perror = [3.74, 2.80, 3.01]
rects1 = ax.bar(ind, Pvals, width, color=red, yerr=Perror, capsize=capsize)
Mvals = [39.96, 39.03, 34.31]
Merror = [4.17, 1.08, 1.85]
rects2 = ax.bar(ind+width, Mvals, width, color=green, yerr=Merror, capsize=capsize)
Cvals = [71.71, 57.91, 44.56]
Cerror = [3.41, 1.93, 5.40]
rects3 = ax.bar(ind+width*2, Cvals, width, color=blue, yerr=Cerror, capsize=capsize)

ax.set_xlabel('Standard deviation (\u03C3)')
ax.set_ylabel('Accuracy (%)')
ax.set_xticks(ind+width)
ax.set_xticklabels( ('0.05', '0.10', '0.15') )
plt.ylim(top=100)
plt.title("DNA E.coli")

ax.legend( (rects1[0], rects2[0], rects3[0]), ('Permutation', 'Multiplication', 'Convolution') )

ax = fig3.add_subplot(152)

Pvals = [55.71, 54.68, 44.17]
Perror = [2.93, 2.56, 3.06]
rects1 = ax.bar(ind, Pvals, width, color=red, yerr=Perror, capsize=capsize)
Mvals = [52.09, 56.24, 51.84]
Merror = [1.22, 2.02, 2.50]
rects2 = ax.bar(ind+width, Mvals, width, color=green, yerr=Merror, capsize=capsize)
Cvals = [68.06, 64.66, 60.18]
Cerror = [1.14, 3.18, 3.45]
rects3 = ax.bar(ind+width*2, Cvals, width, color=blue, yerr=Cerror, capsize=capsize)

ax.set_xlabel('Standard deviation (\u03C3)')
ax.set_ylabel('Accuracy (%)')
ax.set_xticks(ind+width)
ax.set_xticklabels( ('0.05', '0.10', '0.15') )
plt.ylim(top=100)
plt.title("DNA Anodisc")

ax.legend( (rects1[0], rects2[0], rects3[0]), ('Permutation', 'Multiplication', 'Convolution') )

ax = fig3.add_subplot(153)

Pvals = [50.65, 53.54, 39.68]
Perror = [2.94, 4.28, 3.05]
rects1 = ax.bar(ind, Pvals, width, color=red, yerr=Perror, capsize=capsize)
Mvals = [55.14, 48.99, 32.90]
Merror = [3.19, 1.96, 2.76]
rects2 = ax.bar(ind+width, Mvals, width, color=green, yerr=Merror, capsize=capsize)
Cvals = [66.16, 66.89, 47.10]
Cerror = [2.52, 1.72, 3.96]
rects3 = ax.bar(ind+width*2, Cvals, width, color=blue, yerr=Cerror, capsize=capsize)

ax.set_xlabel('Standard deviation (\u03C3)')
ax.set_ylabel('Accuracy (%)')
ax.set_xticks(ind+width)
ax.set_xticklabels( ('0.05', '0.10', '0.15') )
plt.ylim(top=100)
plt.title("DNA In-Liquid DNA")

ax.legend( (rects1[0], rects2[0], rects3[0]), ('Permutation', 'Multiplication', 'Convolution') )

ax = fig3.add_subplot(154)

Pvals = [44.58, 43.99, 41.24]
Perror = [2.79, 1.83, 2.73]
rects1 = ax.bar(ind, Pvals, width, color=red, yerr=Perror, capsize=capsize)
Mvals = [53.22, 34.52, 24.39]
Merror = [1.38, 2.82, 3.75]
rects2 = ax.bar(ind+width, Mvals, width, color=green, yerr=Merror, capsize=capsize)
Cvals = [49.58, 45.89, 36.22]
Cerror = [4.26, 2.82, 4.31]
rects3 = ax.bar(ind+width*2, Cvals, width, color=blue, yerr=Cerror, capsize=capsize)

ax.set_xlabel('Standard deviation (\u03C3)')
ax.set_ylabel('Accuracy (%)')
ax.set_xticks(ind+width)
ax.set_xticklabels( ('0.05', '0.10', '0.15') )
plt.ylim(top=100)
plt.title("Yeast In-Liquid HK")

ax.legend( (rects1[0], rects2[0], rects3[0]), ('Permutation', 'Multiplication', 'Convolution') )

ax = fig3.add_subplot(155)

Pvals = [43.61, 50.10, 44.23]
Perror = [3.26, 1.71, 2.04]
rects1 = ax.bar(ind, Pvals, width, color=red, yerr=Perror, capsize=capsize)
Mvals = [62.20, 42.80, 37.39]
Merror = [1.58, 1.98, 3.75]
rects2 = ax.bar(ind+width, Mvals, width, color=green, yerr=Merror, capsize=capsize)
Cvals = [66.54, 62.56, 58.22]
Cerror = [2.97, 2.24, 2.92]
rects3 = ax.bar(ind+width*2, Cvals, width, color=blue, yerr=Cerror, capsize=capsize)

ax.set_xlabel('Standard deviation (\u03C3)')
ax.set_ylabel('Accuracy (%)')
ax.set_xticks(ind+width)
ax.set_xticklabels( ('0.05', '0.10', '0.15') )
plt.ylim(top=100)
plt.title("Yeast In-Liquid Live")

ax.legend( (rects1[0], rects2[0], rects3[0]), ('Permutation', 'Multiplication', 'Convolution') )

plt.subplots_adjust(left = 0.035, right = 0.985, top = 0.920, bottom = 0.395, hspace = 0.200, wspace = 0.200)
fig3.suptitle('White Gaussian Noise Accuracies')

#---------------Additive Noise F1s---------------

fig4 = plt.figure()

ax = fig4.add_subplot(151)

Pvals = [0.61, 0.58, 0.75]
Perror = [0.03, 0.04, 0.02]
rects1 = ax.bar(ind, Pvals, width, color=red, yerr=Perror, capsize=capsize)
Mvals = [0.49, 0.63, 0.88]
Merror = [0.04, 0.05, 0.02]
rects2 = ax.bar(ind+width, Mvals, width, color=green, yerr=Merror, capsize=capsize)
Cvals = [0.84, 0.90, 0.88]
Cerror = [0.02, 0.03, 0.03]
rects3 = ax.bar(ind+width*2, Cvals, width, color=blue, yerr=Cerror, capsize=capsize)

ax.set_xlabel('Additive constant')
ax.set_ylabel('F1-score')
ax.set_xticks(ind+width)
ax.set_xticklabels( ('0.025', '0.050', '0.100') )
plt.ylim(top=1)
plt.title("DNA E.coli")

ax.legend( (rects1[0], rects2[0], rects3[0]), ('Permutation', 'Multiplication', 'Convolution') )

ax = fig4.add_subplot(152)

Pvals = [0.59, 0.86, 0.92]
Perror = [0.03, 0.01, 0.003]
rects1 = ax.bar(ind, Pvals, width, color=red, yerr=Perror, capsize=capsize)
Mvals = [0.84, 0.84, 0.88]
Merror = [0.04, 0.02, 0.02]
rects2 = ax.bar(ind+width, Mvals, width, color=green, yerr=Merror, capsize=capsize)
Cvals = [0.92, 0.93, 0.95]
Cerror = [0.01, 0.01, 0.01]
rects3 = ax.bar(ind+width*2, Cvals, width, color=blue, yerr=Cerror, capsize=capsize)

ax.set_xlabel('Additive constant')
ax.set_ylabel('F1-score')
ax.set_xticks(ind+width)
ax.set_xticklabels( ('0.025', '0.050', '0.100') )
plt.ylim(top=1)
plt.title("DNA Anodisc")

ax.legend( (rects1[0], rects2[0], rects3[0]), ('Permutation', 'Multiplication', 'Convolution') )

ax = fig4.add_subplot(153)

Pvals = [0.89, 0.98, 0.77]
Perror = [0.03, 0.002, 0.02]
rects1 = ax.bar(ind, Pvals, width, color=red, yerr=Perror, capsize=capsize)
Mvals = [0.82, 0.96, 0.76]
Merror = [0.02, 0.01, 0.03]
rects2 = ax.bar(ind+width, Mvals, width, color=green, yerr=Merror, capsize=capsize)
Cvals = [0.92, 0.98, 0.89]
Cerror = [0.02, 0.002, 0.02]
rects3 = ax.bar(ind+width*2, Cvals, width, color=blue, yerr=Cerror, capsize=capsize)

ax.set_xlabel('Additive constant')
ax.set_ylabel('F1-score')
ax.set_xticks(ind+width)
ax.set_xticklabels( ('0.025', '0.050', '0.100') )
plt.ylim(top=1)
plt.title("DNA In-Liquid DNA")

ax.legend( (rects1[0], rects2[0], rects3[0]), ('Permutation', 'Multiplication', 'Convolution') )

ax = fig4.add_subplot(154)

Pvals = [0.82, 0.61, 0.75]
Perror = [0.01, 0.02, 0.02]
rects1 = ax.bar(ind, Pvals, width, color=red, yerr=Perror, capsize=capsize)
Mvals = [0.75, 0.56, 0.73]
Merror = [0.03, 0.04, 0.02]
rects2 = ax.bar(ind+width, Mvals, width, color=green, yerr=Merror, capsize=capsize)
Cvals = [0.84, 0.90, 0.82]
Cerror = [0.03, 0.02, 0.03]
rects3 = ax.bar(ind+width*2, Cvals, width, color=blue, yerr=Cerror, capsize=capsize)

ax.set_xlabel('Additive constant')
ax.set_ylabel('F1-score')
ax.set_xticks(ind+width)
ax.set_xticklabels( ('0.025', '0.050', '0.100') )
plt.ylim(top=1)
plt.title("Yeast In-Liquid HK")

ax.legend( (rects1[0], rects2[0], rects3[0]), ('Permutation', 'Multiplication', 'Convolution') )

ax = fig4.add_subplot(155)

Pvals = [0.75, 0.71, 0.78]
Perror = [0.01, 0.02, 0.02]
rects1 = ax.bar(ind, Pvals, width, color=red, yerr=Perror, capsize=capsize)
Mvals = [0.66, 0.68, 0.64]
Merror = [0.01, 0.02, 0.01]
rects2 = ax.bar(ind+width, Mvals, width, color=green, yerr=Merror, capsize=capsize)
Cvals = [0.73, 0.75, 0.78]
Cerror = [0.04, 0.03, 0.03]
rects3 = ax.bar(ind+width*2, Cvals, width, color=blue, yerr=Cerror, capsize=capsize)

ax.set_xlabel('Additive constant')
ax.set_ylabel('F1-score')
ax.set_xticks(ind+width)
ax.set_xticklabels( ('0.025', '0.050', '0.100') )
plt.ylim(top=1)
plt.title("Yeast In-Liquid Live")

ax.legend( (rects1[0], rects2[0], rects3[0]), ('Permutation', 'Multiplication', 'Convolution') )

plt.subplots_adjust(left = 0.035, right = 0.985, top = 0.920, bottom = 0.395, hspace = 0.200, wspace = 0.200)
fig4.suptitle('Additive Noise F1s')

#---------------Additive Noise Accuracies---------------

fig5 = plt.figure()

ax = fig5.add_subplot(151)

Pvals = [44.15, 45.56, 69.80]
Perror = [3.54, 3.62, 2.97]
rects1 = ax.bar(ind, Pvals, width, color=red, yerr=Perror, capsize=capsize)
Mvals = [44.13, 51.15, 63.41]
Merror = [4.21, 3.22, 2.84]
rects2 = ax.bar(ind+width, Mvals, width, color=green, yerr=Merror, capsize=capsize)
Cvals = [78.61, 86.33, 81.92]
Cerror = [1.77, 3.50, 3.22]
rects3 = ax.bar(ind+width*2, Cvals, width, color=blue, yerr=Cerror, capsize=capsize)

ax.set_xlabel('Additive constant')
ax.set_ylabel('Accuracy (%)')
ax.set_xticks(ind+width)
ax.set_xticklabels( ('0.025', '0.050', '0.100') )
plt.ylim(top=100)
plt.title("DNA E.coli")

ax.legend( (rects1[0], rects2[0], rects3[0]), ('Permutation', 'Multiplication', 'Convolution') )

ax = fig5.add_subplot(152)

Pvals = [48.27, 57.86, 70.73]
Perror = [1.87, 3.16, 1.57]
rects1 = ax.bar(ind, Pvals, width, color=red, yerr=Perror, capsize=capsize)
Mvals = [58.07, 55.13, 59.34]
Merror = [6.08, 2.89, 3.79]
rects2 = ax.bar(ind+width, Mvals, width, color=green, yerr=Merror, capsize=capsize)
Cvals = [71.85, 74.31, 75.59]
Cerror = [2.40, 2.53, 3.33]
rects3 = ax.bar(ind+width*2, Cvals, width, color=blue, yerr=Cerror, capsize=capsize)

ax.set_xlabel('Additive constant')
ax.set_ylabel('Accuracy (%)')
ax.set_xticks(ind+width)
ax.set_xticklabels( ('0.025', '0.050', '0.100') )
plt.ylim(top=100)
plt.title("DNA Anodisc")

ax.legend( (rects1[0], rects2[0], rects3[0]), ('Permutation', 'Multiplication', 'Convolution') )

ax = fig5.add_subplot(153)

Pvals = [81.32, 77.56, 57.27]
Perror = [2.65, 3.33, 2.87]
rects1 = ax.bar(ind, Pvals, width, color=red, yerr=Perror, capsize=capsize)
Mvals = [64.58, 67.62, 53.68]
Merror = [2.18, 1.69, 2.01]
rects2 = ax.bar(ind+width, Mvals, width, color=green, yerr=Merror, capsize=capsize)
Cvals = [76.22, 79.22, 80.04]
Cerror = [2.28, 1.66, 1.93]
rects3 = ax.bar(ind+width*2, Cvals, width, color=blue, yerr=Cerror, capsize=capsize)

ax.set_xlabel('Additive constant')
ax.set_ylabel('Accuracy (%)')
ax.set_xticks(ind+width)
ax.set_xticklabels( ('0.025', '0.050', '0.100') )
plt.ylim(top=100)
plt.title("DNA In-Liquid DNA")

ax.legend( (rects1[0], rects2[0], rects3[0]), ('Permutation', 'Multiplication', 'Convolution') )

ax = fig5.add_subplot(154)

Pvals = [50.86, 37.03, 34.28]
Perror = [1.11, 2.26, 3.50]
rects1 = ax.bar(ind, Pvals, width, color=red, yerr=Perror, capsize=capsize)
Mvals = [47.00, 40.34, 56.34]
Merror = [3.12, 2.46, 1.29]
rects2 = ax.bar(ind+width, Mvals, width, color=green, yerr=Merror, capsize=capsize)
Cvals = [48.75, 80.26, 59.98]
Cerror = [4.12, 2.01, 5.25]
rects3 = ax.bar(ind+width*2, Cvals, width, color=blue, yerr=Cerror, capsize=capsize)

ax.set_xlabel('Additive constant')
ax.set_ylabel('Accuracy (%)')
ax.set_xticks(ind+width)
ax.set_xticklabels( ('0.025', '0.050', '0.100') )
plt.ylim(top=100)
plt.title("Yeast In-Liquid HK")

ax.legend( (rects1[0], rects2[0], rects3[0]), ('Permutation', 'Multiplication', 'Convolution') )

ax = fig5.add_subplot(155)

Pvals = [64.16, 61.53, 68.05]
Perror = [1.07, 1.64, 2.55]
rects1 = ax.bar(ind, Pvals, width, color=red, yerr=Perror, capsize=capsize)
Mvals = [57.59, 59.25, 58.33]
Merror = [1.52, 1.40, 1.11]
rects2 = ax.bar(ind+width, Mvals, width, color=green, yerr=Merror, capsize=capsize)
Cvals = [63.33, 65.40, 67.85]
Cerror = [3.16, 2.12, 2.53]
rects3 = ax.bar(ind+width*2, Cvals, width, color=blue, yerr=Cerror, capsize=capsize)

ax.set_xlabel('Additive constant')
ax.set_ylabel('Accuracy (%)')
ax.set_xticks(ind+width)
ax.set_xticklabels( ('0.025', '0.050', '0.100') )
plt.ylim(top=100)
plt.title("Yeast In-Liquid Live")

ax.legend( (rects1[0], rects2[0], rects3[0]), ('Permutation', 'Multiplication', 'Convolution') )

plt.subplots_adjust(left = 0.035, right = 0.985, top = 0.920, bottom = 0.395, hspace = 0.200, wspace = 0.200)
fig5.suptitle('Additive Noise Accuracies')

#---------------Splits F1s---------------

N = 4
ind = np.arange(N)  # the x locations for the groups

fig6 = plt.figure()

ax = fig6.add_subplot(151)

Pvals = [0.64, 0.60, 0.61, 0.60]
Perror = [0.07, 0.05, 0.05, 0.05]
rects1 = ax.bar(ind, Pvals, width, color=red, yerr=Perror, capsize=capsize)
Mvals = [0.80, 0.78, 0.80, 0.80]
Merror = [0.02, 0.04, 0.02, 0.02]
rects2 = ax.bar(ind+width, Mvals, width, color=green, yerr=Merror, capsize=capsize)
Cvals = [0.87, 0.86, 0.87, 0.87]
Cerror = [0.04, 0.03, 0.02, 0.03]
rects3 = ax.bar(ind+width*2, Cvals, width, color=blue, yerr=Cerror, capsize=capsize)

ax.set_xlabel('Number of splits')
ax.set_ylabel('F1-score')
ax.set_xticks(ind+width)
ax.set_xticklabels( ('2', '3', '4', '5') )
plt.ylim(top=1)
plt.title("DNA E.coli")

ax.legend( (rects1[0], rects2[0], rects3[0]), ('Permutation', 'Multiplication', 'Convolution') )

ax = fig6.add_subplot(152)

Pvals = [0.83, 0.82, 0.85, 0.78]
Perror = [0.02, 0.03, 0.03, 0.02]
rects1 = ax.bar(ind, Pvals, width, color=red, yerr=Perror, capsize=capsize)
Mvals = [0.87, 0.89, 0.90, 0.90]
Merror = [0.05, 0.01, 0.01, 0.01]
rects2 = ax.bar(ind+width, Mvals, width, color=green, yerr=Merror, capsize=capsize)
Cvals = [0.86, 0.87, 0.86, 0.87]
Cerror = [0.03, 0.02, 0.03, 0.02]
rects3 = ax.bar(ind+width*2, Cvals, width, color=blue, yerr=Cerror, capsize=capsize)

ax.set_xlabel('Number of splits')
ax.set_ylabel('F1-score')
ax.set_xticks(ind+width)
ax.set_xticklabels( ('2', '3', '4', '5') )
plt.ylim(top=1)
plt.title("DNA Anodisc")

ax.legend( (rects1[0], rects2[0], rects3[0]), ('Permutation', 'Multiplication', 'Convolution') )

ax = fig6.add_subplot(153)

Pvals = [0.98, 0.98, 0.98, 0.99]
Perror = [0.01, 0.02, 0.01, 0.01]
rects1 = ax.bar(ind, Pvals, width, color=red, yerr=Perror, capsize=capsize)
Mvals = [0.94, 0.93, 0.95, 0.95]
Merror = [0.01, 0.03, 0.01, 0.01]
rects2 = ax.bar(ind+width, Mvals, width, color=green, yerr=Merror, capsize=capsize)
Cvals = [0.96, 0.98, 0.97, 0.98]
Cerror = [0.03, 0.02, 0.02, 0.01]
rects3 = ax.bar(ind+width*2, Cvals, width, color=blue, yerr=Cerror, capsize=capsize)

ax.set_xlabel('Number of splits')
ax.set_ylabel('F1-score')
ax.set_xticks(ind+width)
ax.set_xticklabels( ('2', '3', '4', '5') )
plt.ylim(top=1)
plt.title("DNA In-Liquid DNA")

ax.legend( (rects1[0], rects2[0], rects3[0]), ('Permutation', 'Multiplication', 'Convolution') )

ax = fig6.add_subplot(154)

Pvals = [0.77, 0.78, 0.78, 0.76]
Perror = [0.01, 0.02, 0.02, 0.02]
rects1 = ax.bar(ind, Pvals, width, color=red, yerr=Perror, capsize=capsize)
Mvals = [0.83, 0.82, 0.81, 0.82]
Merror = [0.02, 0.02, 0.02, 0.02]
rects2 = ax.bar(ind+width, Mvals, width, color=green, yerr=Merror, capsize=capsize)
Cvals = [0.82, 0.85, 0.85, 0.84]
Cerror = [0.05, 0.02, 0.02, 0.02]
rects3 = ax.bar(ind+width*2, Cvals, width, color=blue, yerr=Cerror, capsize=capsize)

ax.set_xlabel('Number of splits')
ax.set_ylabel('F1-score')
ax.set_xticks(ind+width)
ax.set_xticklabels( ('2', '3', '4', '5') )
plt.ylim(top=1)
plt.title("Yeast In-Liquid HK")

ax.legend( (rects1[0], rects2[0], rects3[0]), ('Permutation', 'Multiplication', 'Convolution') )

ax = fig6.add_subplot(155)

Pvals = [0.76, 0.75, 0.79, 0.73]
Perror = [0.02, 0.01, 0.02, 0.02]
rects1 = ax.bar(ind, Pvals, width, color=red, yerr=Perror, capsize=capsize)
Mvals = [0.82, 0.82, 0.81, 0.81]
Merror = [0.02, 0.01, 0.01, 0.01]
rects2 = ax.bar(ind+width, Mvals, width, color=green, yerr=Merror, capsize=capsize)
Cvals = [0.86, 0.84, 0.85, 0.83]
Cerror = [0.02, 0.01, 0.01, 0.01]
rects3 = ax.bar(ind+width*2, Cvals, width, color=blue, yerr=Cerror, capsize=capsize)

ax.set_xlabel('Number of splits')
ax.set_ylabel('F1-score')
ax.set_xticks(ind+width)
ax.set_xticklabels( ('2', '3', '4', '5') )
plt.ylim(top=1)
plt.title("Yeast In-Liquid Live")

ax.legend( (rects1[0], rects2[0], rects3[0]), ('Permutation', 'Multiplication', 'Convolution') )

plt.subplots_adjust(left = 0.035, right = 0.985, top = 0.920, bottom = 0.395, hspace = 0.200, wspace = 0.200)
fig6.suptitle('Splits F1s')

#---------------Splits Accuracies---------------

fig7 = plt.figure()

ax = fig7.add_subplot(151)

Pvals = [55.56, 47.00, 53.22, 49.22]
Perror = [8.69, 3.55, 4.20, 3.67]
rects1 = ax.bar(ind, Pvals, width, color=red, yerr=Perror, capsize=capsize)
Mvals = [71.11, 70.44, 71.68, 72.11]
Merror = [2.82, 3.86, 1.98, 1.77]
rects2 = ax.bar(ind+width, Mvals, width, color=green, yerr=Merror, capsize=capsize)
Cvals = [79.22, 80.11, 81.61, 81.56]
Cerror = [5.10, 3.97, 2.52, 3.11]
rects3 = ax.bar(ind+width*2, Cvals, width, color=blue, yerr=Cerror, capsize=capsize)

ax.set_xlabel('Number of splits')
ax.set_ylabel('Accuracy (%)')
ax.set_xticks(ind+width)
ax.set_xticklabels( ('2', '3', '4', '5') )
plt.ylim(top=100)
plt.title("DNA E.coli")

ax.legend( (rects1[0], rects2[0], rects3[0]), ('Permutation', 'Multiplication', 'Convolution') )

ax = fig7.add_subplot(152)

Pvals = [49.22, 52.44, 54.38, 52.44]
Perror = [3.92, 3.70, 1.88, 2.21]
rects1 = ax.bar(ind, Pvals, width, color=red, yerr=Perror, capsize=capsize)
Mvals = [55.33, 56.56, 57.74, 58.56]
Merror = [4.34, 3.16, 3.26, 3.06]
rects2 = ax.bar(ind+width, Mvals, width, color=green, yerr=Merror, capsize=capsize)
Cvals = [63.22, 65.89, 67.92, 67.22]
Cerror = [5.84, 2.46, 3.46, 1.91]
rects3 = ax.bar(ind+width*2, Cvals, width, color=blue, yerr=Cerror, capsize=capsize)

ax.set_xlabel('Number of splits')
ax.set_ylabel('Accuracy (%)')
ax.set_xticks(ind+width)
ax.set_xticklabels( ('2', '3', '4', '5') )
plt.ylim(top=100)
plt.title("DNA Anodisc")

ax.legend( (rects1[0], rects2[0], rects3[0]), ('Permutation', 'Multiplication', 'Convolution') )

ax = fig7.add_subplot(153)

Pvals = [86.78, 86.56, 86.37, 87.00]
Perror = [1.52, 2.25, 1.47, 1.17]
rects1 = ax.bar(ind, Pvals, width, color=red, yerr=Perror, capsize=capsize)
Mvals = [84.33, 83.11, 85.22, 84.56]
Merror = [1.77, 3.88, 1.26, 1.84]
rects2 = ax.bar(ind+width, Mvals, width, color=green, yerr=Merror, capsize=capsize)
Cvals = [83.11, 86.22, 84.67, 86.00]
Cerror = [4.08, 2.11, 1.55, 2.04]
rects3 = ax.bar(ind+width*2, Cvals, width, color=blue, yerr=Cerror, capsize=capsize)

ax.set_xlabel('Number of splits')
ax.set_ylabel('Accuracy (%)')
ax.set_xticks(ind+width)
ax.set_xticklabels( ('2', '3', '4', '5') )
plt.ylim(top=100)
plt.title("DNA In-Liquid DNA")

ax.legend( (rects1[0], rects2[0], rects3[0]), ('Permutation', 'Multiplication', 'Convolution') )

ax = fig7.add_subplot(154)

Pvals = [42.56, 38.33, 40.34, 37.44]
Perror = [4.48, 3.36, 2.51, 3.23]
rects1 = ax.bar(ind, Pvals, width, color=red, yerr=Perror, capsize=capsize)
Mvals = [48.67, 52.67, 53.19, 54.78]
Merror = [6.50, 5.06, 3.95, 4.48]
rects2 = ax.bar(ind+width, Mvals, width, color=green, yerr=Merror, capsize=capsize)
Cvals = [49.78, 46.11, 44.82, 45.00]
Cerror = [5.56, 3.20, 3.50, 1.31]
rects3 = ax.bar(ind+width*2, Cvals, width, color=blue, yerr=Cerror, capsize=capsize)

ax.set_xlabel('Number of splits')
ax.set_ylabel('Accuracy (%)')
ax.set_xticks(ind+width)
ax.set_xticklabels( ('2', '3', '4', '5') )
plt.ylim(top=100)
plt.title("Yeast In-Liquid HK")

ax.legend( (rects1[0], rects2[0], rects3[0]), ('Permutation', 'Multiplication', 'Convolution') )

ax = fig7.add_subplot(155)

Pvals = [56.22, 56.44, 61.14, 54.56]
Perror = [4.45, 1.72, 3.02, 2.19]
rects1 = ax.bar(ind, Pvals, width, color=red, yerr=Perror, capsize=capsize)
Mvals = [71.44, 72.00, 72.31, 72.22]
Merror = [1.74, 0.87, 0.60, 0.74]
rects2 = ax.bar(ind+width, Mvals, width, color=green, yerr=Merror, capsize=capsize)
Cvals = [74.44, 72.11, 72.60, 72.11]
Cerror = [2.57, 0.97, 0.71, 1.10]
rects3 = ax.bar(ind+width*2, Cvals, width, color=blue, yerr=Cerror, capsize=capsize)

ax.set_xlabel('Number of splits')
ax.set_ylabel('Accuracy (%)')
ax.set_xticks(ind+width)
ax.set_xticklabels( ('2', '3', '4', '5') )
plt.ylim(top=100)
plt.title("Yeast In-Liquid Live")

ax.legend( (rects1[0], rects2[0], rects3[0]), ('Permutation', 'Multiplication', 'Convolution') )

plt.subplots_adjust(left = 0.035, right = 0.985, top = 0.920, bottom = 0.395, hspace = 0.200, wspace = 0.200)
fig7.suptitle('Splits Accuracies')

#---------------Multiplicative Noise F1s---------------

fig8 = plt.figure()

ax = fig8.add_subplot(151)

Pvals = [0.69, 0.73, 0.62, 0.64]
Perror = [0.03, 0.01, 0.04, 0.05]
rects1 = ax.bar(ind, Pvals, width, color=red, yerr=Perror, capsize=capsize)
Mvals = [0.74, 0.66, 0.70, 0.56]
Merror = [0.02, 0.02, 0.03, 0.03]
rects2 = ax.bar(ind+width, Mvals, width, color=green, yerr=Merror, capsize=capsize)
Cvals = [0.81, 0.84, 0.85, 0.83]
Cerror = [0.02, 0.02, 0.03, 0.02]
rects3 = ax.bar(ind+width*2, Cvals, width, color=blue, yerr=Cerror, capsize=capsize)

ax.set_xlabel('Multiplier')
ax.set_ylabel('F1-score')
ax.set_xticks(ind+width)
ax.set_xticklabels( ('0.75', '0.90', '1.10', '1.25') )
plt.ylim(top=1)
plt.title("DNA E.coli")

ax.legend( (rects1[0], rects2[0], rects3[0]), ('Permutation', 'Multiplication', 'Convolution') )

ax = fig8.add_subplot(152)

Pvals = [0.87, 0.74, 0.91, 0.91]
Perror = [0.01, 0.02, 0.01, 0.01]
rects1 = ax.bar(ind, Pvals, width, color=red, yerr=Perror, capsize=capsize)
Mvals = [0.79, 0.71, 0.90, 0.79]
Merror = [0.05, 0.02, 0.05, 0.04]
rects2 = ax.bar(ind+width, Mvals, width, color=green, yerr=Merror, capsize=capsize)
Cvals = [0.78, 0.74, 0.93, 0.94]
Cerror = [0.03, 0.03, 0.01, 0.01]
rects3 = ax.bar(ind+width*2, Cvals, width, color=blue, yerr=Cerror, capsize=capsize)

ax.set_xlabel('Multiplier')
ax.set_ylabel('F1-score')
ax.set_xticks(ind+width)
ax.set_xticklabels( ('0.75', '0.90', '1.10', '1.25') )
plt.ylim(top=1)
plt.title("DNA Anodisc")

ax.legend( (rects1[0], rects2[0], rects3[0]), ('Permutation', 'Multiplication', 'Convolution') )

ax = fig8.add_subplot(153)

Pvals = [0.74, 0.96, 0.94, 0.90]
Perror = [0.04, 0.01, 0.01, 0.03]
rects1 = ax.bar(ind, Pvals, width, color=red, yerr=Perror, capsize=capsize)
Mvals = [0.67, 0.97, 0.81, 0.75]
Merror = [0.03, 0.003, 0.03, 0.01]
rects2 = ax.bar(ind+width, Mvals, width, color=green, yerr=Merror, capsize=capsize)
Cvals = [0.82, 0.96, 0.95, 0.91]
Cerror = [0.02, 0.02, 0.02, 0.03]
rects3 = ax.bar(ind+width*2, Cvals, width, color=blue, yerr=Cerror, capsize=capsize)

ax.set_xlabel('Multiplier')
ax.set_ylabel('F1-score')
ax.set_xticks(ind+width)
ax.set_xticklabels( ('0.75', '0.90', '1.10', '1.25') )
plt.ylim(top=1)
plt.title("DNA In-Liquid DNA")

ax.legend( (rects1[0], rects2[0], rects3[0]), ('Permutation', 'Multiplication', 'Convolution') )

ax = fig8.add_subplot(154)

Pvals = [0.77, 0.78, 0.83, 0.83]
Perror = [0.03, 0.01, 0.01, 0.02]
rects1 = ax.bar(ind, Pvals, width, color=red, yerr=Perror, capsize=capsize)
Mvals = [0.74, 0.79, 0.80, 0.78]
Merror = [0.02, 0.03, 0.01, 0.02]
rects2 = ax.bar(ind+width, Mvals, width, color=green, yerr=Merror, capsize=capsize)
Cvals = [0.82, 0.81, 0.82, 0.83]
Cerror = [0.02, 0.02, 0.02, 0.04]
rects3 = ax.bar(ind+width*2, Cvals, width, color=blue, yerr=Cerror, capsize=capsize)

ax.set_xlabel('Multiplier')
ax.set_ylabel('F1-score')
ax.set_xticks(ind+width)
ax.set_xticklabels( ('0.75', '0.90', '1.10', '1.25') )
plt.ylim(top=1)
plt.title("Yeast In-Liquid HK")

ax.legend( (rects1[0], rects2[0], rects3[0]), ('Permutation', 'Multiplication', 'Convolution') )

ax = fig8.add_subplot(155)

Pvals = [0.84, 0.70, 0.74, 0.68]
Perror = [0.01, 0.02, 0.02, 0.01]
rects1 = ax.bar(ind, Pvals, width, color=red, yerr=Perror, capsize=capsize)
Mvals = [0.65, 0.79, 0.78, 0.69]
Merror = [0.01, 0.03, 0.02, 0.02]
rects2 = ax.bar(ind+width, Mvals, width, color=green, yerr=Merror, capsize=capsize)
Cvals = [0.79, 0.78, 0.83, 0.81]
Cerror = [0.02, 0.03, 0.01, 0.03]
rects3 = ax.bar(ind+width*2, Cvals, width, color=blue, yerr=Cerror, capsize=capsize)

ax.set_xlabel('Multiplier')
ax.set_ylabel('F1-score')
ax.set_xticks(ind+width)
ax.set_xticklabels( ('0.75', '0.90', '1.10', '1.25') )
plt.ylim(top=1)
plt.title("Yeast In-Liquid Live")

ax.legend( (rects1[0], rects2[0], rects3[0]), ('Permutation', 'Multiplication', 'Convolution') )

plt.subplots_adjust(left = 0.035, right = 0.985, top = 0.920, bottom = 0.395, hspace = 0.200, wspace = 0.200)
fig8.suptitle('Multiplicative Noise F1s')

#---------------Multiplicative Noise Accuracies---------------

fig9 = plt.figure()

ax = fig9.add_subplot(151)

Pvals = [60.49, 58.75, 48.67, 51.27]
Perror = [2.69, 1.46, 6.37, 3.30]
rects1 = ax.bar(ind, Pvals, width, color=red, yerr=Perror, capsize=capsize)
Mvals = [50.86, 59.00, 56.80, 48.89]
Merror = [2.24, 1.84, 3.39, 2.49]
rects2 = ax.bar(ind+width, Mvals, width, color=green, yerr=Merror, capsize=capsize)
Cvals = [75.51, 78.82, 80.27, 77.20]
Cerror = [1.99, 3.15, 2.37, 2.50]
rects3 = ax.bar(ind+width*2, Cvals, width, color=blue, yerr=Cerror, capsize=capsize)

ax.set_xlabel('Multiplier')
ax.set_ylabel('Accuracy (%)')
ax.set_xticks(ind+width)
ax.set_xticklabels( ('0.75', '0.90', '1.10', '1.25') )
plt.ylim(top=100)
plt.title("DNA E.coli")

ax.legend( (rects1[0], rects2[0], rects3[0]), ('Permutation', 'Multiplication', 'Convolution') )

ax = fig9.add_subplot(152)

Pvals = [54.16, 52.68, 57.24, 66.93]
Perror = [4.94, 1.98, 2.90, 3.38]
rects1 = ax.bar(ind, Pvals, width, color=red, yerr=Perror, capsize=capsize)
Mvals = [50.38, 47.96, 58.80, 53.66]
Merror = [3.63, 2.61, 2.79, 3.61]
rects2 = ax.bar(ind+width, Mvals, width, color=green, yerr=Merror, capsize=capsize)
Cvals = [61.24, 62.48, 70.35, 73.05]
Cerror = [1.73, 1.48, 4.20, 3.41]
rects3 = ax.bar(ind+width*2, Cvals, width, color=blue, yerr=Cerror, capsize=capsize)

ax.set_xlabel('Multiplier')
ax.set_ylabel('Accuracy (%)')
ax.set_xticks(ind+width)
ax.set_xticklabels( ('0.75', '0.90', '1.10', '1.25') )
plt.ylim(top=100)
plt.title("DNA Anodisc")

ax.legend( (rects1[0], rects2[0], rects3[0]), ('Permutation', 'Multiplication', 'Convolution') )

ax = fig9.add_subplot(153)

Pvals = [58.03, 84.68, 82.23, 75.69]
Perror = [3.14, 3.60, 1.74, 2.98]
rects1 = ax.bar(ind, Pvals, width, color=red, yerr=Perror, capsize=capsize)
Mvals = [49.11, 83.45, 74.25, 64.39]
Merror = [3.28, 2.49, 2.96, 1.62]
rects2 = ax.bar(ind+width, Mvals, width, color=green, yerr=Merror, capsize=capsize)
Cvals = [66.36, 81.10, 80.20, 75.32]
Cerror = [2.10, 2.21, 2.67, 2.69]
rects3 = ax.bar(ind+width*2, Cvals, width, color=blue, yerr=Cerror, capsize=capsize)

ax.set_xlabel('Multiplier')
ax.set_ylabel('Accuracy (%)')
ax.set_xticks(ind+width)
ax.set_xticklabels( ('0.75', '0.90', '1.10', '1.25') )
plt.ylim(top=100)
plt.title("DNA In-Liquid DNA")

ax.legend( (rects1[0], rects2[0], rects3[0]), ('Permutation', 'Multiplication', 'Convolution') )

ax = fig9.add_subplot(154)

Pvals = [55.25, 40.07, 43.61, 48.77]
Perror = [2.29, 2.79, 2.03, 2.05]
rects1 = ax.bar(ind, Pvals, width, color=red, yerr=Perror, capsize=capsize)
Mvals = [36.07, 48.19, 37.64, 35.26]
Merror = [3.09, 3.43, 3.32, 4.04]
rects2 = ax.bar(ind+width, Mvals, width, color=green, yerr=Merror, capsize=capsize)
Cvals = [42.59, 42.93, 43.89, 44.45]
Cerror = [2.69, 3.60, 3.18, 4.84]
rects3 = ax.bar(ind+width*2, Cvals, width, color=blue, yerr=Cerror, capsize=capsize)

ax.set_xlabel('Multiplier')
ax.set_ylabel('Accuracy (%)')
ax.set_xticks(ind+width)
ax.set_xticklabels( ('0.75', '0.90', '1.10', '1.25') )
plt.ylim(top=100)
plt.title("Yeast In-Liquid HK")

ax.legend( (rects1[0], rects2[0], rects3[0]), ('Permutation', 'Multiplication', 'Convolution') )

ax = fig9.add_subplot(155)

Pvals = [60.93, 52.06, 55.60, 53.81]
Perror = [1.00, 3.19, 1.70, 4.39]
rects1 = ax.bar(ind, Pvals, width, color=red, yerr=Perror, capsize=capsize)
Mvals = [57.20, 70.29, 68.35, 60.04]
Merror = [0.77, 2.56, 2.05, 1.17]
rects2 = ax.bar(ind+width, Mvals, width, color=green, yerr=Merror, capsize=capsize)
Cvals = [68.25, 68.35, 72.01, 70.12]
Cerror = [2.07, 2.38, 1.78, 2.18]
rects3 = ax.bar(ind+width*2, Cvals, width, color=blue, yerr=Cerror, capsize=capsize)

ax.set_xlabel('Multiplier')
ax.set_ylabel('Accuracy (%)')
ax.set_xticks(ind+width)
ax.set_xticklabels( ('0.75', '0.90', '1.10', '1.25') )
plt.ylim(top=100)
plt.title("Yeast In-Liquid Live")

ax.legend( (rects1[0], rects2[0], rects3[0]), ('Permutation', 'Multiplication', 'Convolution') )

plt.subplots_adjust(left = 0.035, right = 0.985, top = 0.920, bottom = 0.395, hspace = 0.200, wspace = 0.200)
fig9.suptitle('Multiplicative Noise Accuracies')

#----------------------------------------
plt.show()
