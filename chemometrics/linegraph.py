import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


red = '#EA4335'
blue = '#4285F4'
green = '#34A853'
yellow = '#FBBC05'

redc = '#35EA6E'
bluec = '#F48942'
greenc = '#A83C34'
yellowc = '#7C05FB'

colorc = {red:redc, blue:bluec, green:greenc, yellow:yellowc}


def plot_line_graph(fig, subplot, x, hdc, svm, dt, gbdt, title, xlabel, ylabel, ylimtop, legend_names):
	ax = fig.add_subplot(subplot)

	ax1 = ax.plot(x, hdc, marker = 'o', markerfacecolor = redc, markersize = 5, color = red, linewidth = 2)
	ax2 = ax.plot(x, svm, marker = 'o', markerfacecolor = bluec, markersize = 5, color = blue, linewidth = 2)
	ax3 = ax.plot(x, dt, marker = 'o', markerfacecolor = greenc, markersize = 5, color = green, linewidth = 2)
	ax4 = ax.plot(x, gbdt, marker = 'o', markerfacecolor = yellowc, markersize = 5, color = yellow, linewidth = 2)
	handles = [ax1[0], ax2[0], ax3[0], ax4[0]]

	plt.title(title)
	ax.set_xlabel(xlabel)
	ax.set_ylabel(ylabel)
	plt.ylim(bottom=0, top=ylimtop)
	ax.xaxis.set_major_locator(MaxNLocator(integer=True))
	ax.legend( (handles), (legend_names) )

	return ax




subplots = [151, 152, 153, 154, 155]
categories = ['DNA ECOLI', 'DNA@Anod', 'Inliquid DNA', 'Yeast inliquid HK', 'Yeast inliquid Live']
data = {category : {} for category in categories}
xtitle = 0.5
ytitle = 0.6






######################################### Splits (ACCURACY) ########################################################
fig0 = plt.figure()
fig0.suptitle('Splits Accuracies', fontsize=16, x=xtitle, y=ytitle)

xlabel = 'Number of splits'
ylabel = 'Accuracy (%)'
legend_names = ['HDC (convolution)', 'SVM', 'Decision Tree', 'Gradient Boosted Decision Tree']

data['x'] = [2, 3, 4, 5]

# DNA ECOLI
data[categories[0]]['hdc'] = [79.22, 80.11, 81.61, 81.56]
data[categories[0]]['svm'] = [99.00, 98.78, 99.67, 99.44]
data[categories[0]]['dt'] = [83.44, 89.00, 90.38, 80.22]
data[categories[0]]['gbdt'] = [89.56, 89.78, 91.77, 91.22]

# DNA@Anod
data[categories[1]]['hdc'] = [63.22, 65.89, 67.92, 67.22]
data[categories[1]]['svm'] = [92.78, 92.56, 94.80, 94.22]
data[categories[1]]['dt'] = [81.11, 85.89, 86.57, 87.56]
data[categories[1]]['gbdt'] = [81.11, 88.11, 89.34, 89.11]

# Inliquid DNA
data[categories[2]]['hdc'] = [83.11, 86.22, 84.67, 86.00]
data[categories[2]]['svm'] = [89.44, 89.44, 89.30, 88.89]
data[categories[2]]['dt'] = [79.33, 82.56, 82.49, 84.22]
data[categories[2]]['gbdt'] = [82.00, 82.44, 82.94, 83.22]

# Yeast inliquid HK
data[categories[3]]['hdc'] = [49.78, 46.11, 44.82, 45.00]
data[categories[3]]['svm'] = [97.11, 98.33, 98.87, 98.89]
data[categories[3]]['dt'] = [86.33, 88.67, 89.21, 90.00]
data[categories[3]]['gbdt'] = [80.56, 83.33, 84.58, 86.00]

# Yeast inliquid Live
data[categories[4]]['hdc'] = [74.44, 72.11, 72.60, 72.11]
data[categories[4]]['svm'] = [94.00, 96.44, 97.34, 97.22]
data[categories[4]]['dt'] = [86.22, 88.44, 88.25, 88.78]
data[categories[4]]['gbdt'] = [90.00, 84.11, 84.03, 96.33]


for i in range(0, len(categories)):
	ax = plot_line_graph(fig0, subplots[i], data['x'], data[categories[i]]['hdc'], data[categories[i]]['svm'], data[categories[i]]['dt'], data[categories[i]]['gbdt'], categories[i], xlabel, ylabel, 100, legend_names)
	ax.set_xticks(data['x'])
	ax.set_xticklabels( ('2', '3', '4', '5') )


plt.subplots_adjust(left = 0.05, right = 0.98, top = 0.53, bottom = 0.06, hspace = 0.200, wspace = 0.200)





######################################### LIMITED DATA (ACCURACY) ########################################################
fig1 = plt.figure()
fig1.suptitle('Limited Data Accuracies', fontsize=16, x=xtitle, y=ytitle)

xlabel = 'Number of samples'
ylabel = 'Accuracy (%)'
legend_names = ['HDC (convolution)', 'SVM', 'Decision Tree', 'Gradient Boosted Decision Tree']

data['x'] = [1, 3, 5]

# DNA ECOLI
data[categories[0]]['hdc'] = [62.23, 75.20, 75.38]
data[categories[0]]['svm'] = [71.18, 75.18, 76.12]
data[categories[0]]['dt'] = [52.12, 71.20, 79.85]
data[categories[0]]['gbdt'] = [56.71, 70.00, 84.31]

# DNA@Anod
data[categories[1]]['hdc'] = [53.65, 56.93, 66.15]
data[categories[1]]['svm'] = [62.71, 54.94, 57.18]
data[categories[1]]['dt'] = [53.76, 82.40, 79.08]
data[categories[1]]['gbdt'] = [54.00, 76.13, 77.23]

# Inliquid DNA
data[categories[2]]['hdc'] = [69.53, 83.20, 80.92]
data[categories[2]]['svm'] = [70.35, 65.65, 69.41]
data[categories[2]]['dt'] = [52.24, 71.07, 80.46]
data[categories[2]]['gbdt'] = [53.76, 67.20, 76.77]

# Yeast inliquid HK
data[categories[3]]['hdc'] = [64.11, 57.73, 54.62]
data[categories[3]]['svm'] = [64.82, 62.47, 64.82]
data[categories[3]]['dt'] = [57.18, 71.73, 81.54]
data[categories[3]]['gbdt'] = [63.76, 76.67, 77.54]

# Yeast inliquid Live
data[categories[4]]['hdc'] = [70.82, 72.13, 71.85]
data[categories[4]]['svm'] = [68.00, 68.59, 63.88]
data[categories[4]]['dt'] = [68.24, 77.47, 81.68]
data[categories[4]]['gbdt'] = [62.24, 77.73, 82.46]



for i in range(0, len(categories)):
	ax = plot_line_graph(fig1, subplots[i], data['x'], data[categories[i]]['hdc'], data[categories[i]]['svm'], data[categories[i]]['dt'], data[categories[i]]['gbdt'], categories[i], xlabel, ylabel, 100, legend_names)
	ax.set_xticks(data['x'])
	ax.set_xticklabels( ('1', '3', '5') )


plt.subplots_adjust(left = 0.05, right = 0.98, top = 0.53, bottom = 0.06, hspace = 0.200, wspace = 0.200)




######################################### Gaussian Noise (ACCURACY) ########################################################

fig2 = plt.figure()
fig2.suptitle('White Gaussian Noise Accuracies', fontsize=16, x=xtitle, y=ytitle)

xlabel = 'Standard deviation (σ)'
ylabel = 'Accuracy (%)'
legend_names = ['HDC (convolution)', 'SVM', 'Decision Tree', 'Gradient Boosted Decision Tree']

data['x'] = [0.05, 0.10, 0.15]

# DNA ECOLI
data[categories[0]]['hdc'] = [71.71, 57.91, 44.56]
data[categories[0]]['svm'] = [69.90, 36.90, 21.85]
data[categories[0]]['dt'] = [31.55, 21.23, 20.03]
data[categories[0]]['gbdt'] = [30.82, 18.02, 13.66]

# DNA@Anod
data[categories[1]]['hdc'] = [68.06, 64.66, 60.18]
data[categories[1]]['svm'] = [74.78, 59.53, 52.09]
data[categories[1]]['dt'] = [49.22, 33.43, 26.46]
data[categories[1]]['gbdt'] = [56.55, 36.21, 22.75]

# Inliquid DNA
data[categories[2]]['hdc'] = [66.16, 66.89, 47.10]
data[categories[2]]['svm'] = [59.73, 46.26, 29.22]
data[categories[2]]['dt'] = [42.48, 26.57, 23.90]
data[categories[2]]['gbdt'] = [43.50, 25.62, 25.61]

# Yeast inliquid HK
data[categories[3]]['hdc'] = [49.58, 45.89, 36.22]
data[categories[3]]['svm'] = [54.04, 47.38, 26.44]
data[categories[3]]['dt'] = [31.47, 22.08, 20.56]
data[categories[3]]['gbdt'] = [33.92, 19.48, 15.99]

# Yeast inliquid Live
data[categories[4]]['hdc'] = [66.54, 62.56, 58.22]
data[categories[4]]['svm'] = [62.64, 61.24, 51.85]
data[categories[4]]['dt'] = [46.58, 25.20, 25.28]
data[categories[4]]['gbdt'] = [47.60, 29.75, 23.68]



for i in range(0, len(categories)):
	ax = plot_line_graph(fig2, subplots[i], data['x'], data[categories[i]]['hdc'], data[categories[i]]['svm'], data[categories[i]]['dt'], data[categories[i]]['gbdt'], categories[i], xlabel, ylabel, 100, legend_names)
	ax.set_xticks(data['x'])
	ax.set_xticklabels( ('0.05', '0.10', '0.15') )


plt.subplots_adjust(left = 0.05, right = 0.98, top = 0.53, bottom = 0.06, hspace = 0.200, wspace = 0.200)





######################################### Multiplicative Noise (ACCURACY) ########################################################

fig3 = plt.figure()
fig3.suptitle('Multiplicative Noise Accuracies', fontsize=16, x=xtitle, y=ytitle)

xlabel = 'Multiplier'
ylabel = 'Accuracy (%)'
legend_names = ['HDC (convolution)', 'SVM', 'Decision Tree', 'Gradient Boosted Decision Tree']

data['x'] = [0.75, 0.90, 1.10, 1.25]

# DNA ECOLI
data[categories[0]]['hdc'] = [75.51, 78.82, 80.27, 77.20]
data[categories[0]]['svm'] = [99.11, 100.0, 98.78, 99.89]
data[categories[0]]['dt'] = [90.89, 91.11, 90.54, 89.87]
data[categories[0]]['gbdt'] = [91.56, 91.14, 92.46, 90.86]

# DNA@Anod
data[categories[1]]['hdc'] = [61.24, 62.48, 70.35, 73.05]
data[categories[1]]['svm'] = [92.98, 93.01, 94.36, 95.34]
data[categories[1]]['dt'] = [86.09, 87.20, 84.95, 88.14]
data[categories[1]]['gbdt'] = [87.01, 87.80, 88.77, 89.58]

# Inliquid DNA
data[categories[2]]['hdc'] = [66.36, 81.10, 80.20, 75.32]
data[categories[2]]['svm'] = [77.24, 90.88, 91.89, 90.66]
data[categories[2]]['dt'] = [69.44, 86.29, 85.92, 85.86]
data[categories[2]]['gbdt'] = [65.81, 83.10, 84.78, 83.77]

# Yeast inliquid HK
data[categories[3]]['hdc'] = [42.59, 42.93, 43.89, 44.45]
data[categories[3]]['svm'] = [98.78, 98.78, 98.46, 98.89]
data[categories[3]]['dt'] = [88.22, 88.68, 89.32, 91.65]
data[categories[3]]['gbdt'] = [80.63, 85.12, 83.78, 82.63]

# Yeast inliquid Live
data[categories[4]]['hdc'] = [68.25, 68.35, 72.01, 70.12]
data[categories[4]]['svm'] = [96.43, 96.90, 98.33, 97.34]
data[categories[4]]['dt'] = [88.98, 90.35, 89.17, 85.79]
data[categories[4]]['gbdt'] = [86.44, 86.75, 86.71, 85.23]


for i in range(0, len(categories)):
	ax = plot_line_graph(fig3, subplots[i], data['x'], data[categories[i]]['hdc'], data[categories[i]]['svm'], data[categories[i]]['dt'], data[categories[i]]['gbdt'], categories[i], xlabel, ylabel, 100, legend_names)
	ax.set_xticks(data['x'])
	ax.set_xticklabels( ('0.75', '0.90', '1.10', '1.25') )


plt.subplots_adjust(left = 0.05, right = 0.98, top = 0.53, bottom = 0.06, hspace = 0.200, wspace = 0.200)




######################################### Additive Noise (ACCURACY) ########################################################

fig4 = plt.figure()
fig4.suptitle('Additive Noise Accuracies', fontsize=16, x=xtitle, y=ytitle)

xlabel = 'Additive constant'
ylabel = 'Accuracy (%)'
legend_names = ['HDC (convolution)', 'SVM', 'Decision Tree', 'Gradient Boosted Decision Tree']

data['x'] = [0.025, 0.050, 0.10]

# DNA ECOLI
data[categories[0]]['hdc'] = [78.61, 86.33, 81.92]
data[categories[0]]['svm'] = [99.00, 100.0, 99.89]
data[categories[0]]['dt'] = [91.74, 92.01, 91.48]
data[categories[0]]['gbdt'] = [93.34, 92.09, 92.32]

# DNA@Anod
data[categories[1]]['hdc'] = [71.85, 74.31, 75.59]
data[categories[1]]['svm'] = [94.02, 94.13, 97.33]
data[categories[1]]['dt'] = [90.01, 86.98, 91.21]
data[categories[1]]['gbdt'] = [88.82, 86.10, 87.76]

# Inliquid DNA
data[categories[2]]['hdc'] = [76.22, 79.22, 80.04]
data[categories[2]]['svm'] = [91.79, 90.11, 94.34]
data[categories[2]]['dt'] = [86.33, 86.62, 92.56]
data[categories[2]]['gbdt'] = [84.58, 86.92, 89.58]

# Yeast inliquid HK
data[categories[3]]['hdc'] = [48.75, 80.26, 59.98]
data[categories[3]]['svm'] = [98.89, 97.46, 98.89]
data[categories[3]]['dt'] = [95.08, 94.44, 94.44]
data[categories[3]]['gbdt'] = [87.79, 96.53, 96.97]

# Yeast inliquid Live
data[categories[4]]['hdc'] = [63.33, 65.40, 67.85]
data[categories[4]]['svm'] = [98.88, 97.88, 99.10]
data[categories[4]]['dt'] = [85.67, 91.22, 92.15]
data[categories[4]]['gbdt'] = [84.90, 85.50, 87.34]


for i in range(0, len(categories)):
	ax = plot_line_graph(fig4, subplots[i], data['x'], data[categories[i]]['hdc'], data[categories[i]]['svm'], data[categories[i]]['dt'], data[categories[i]]['gbdt'], categories[i], xlabel, ylabel, 100, legend_names)
	ax.set_xticks(data['x'])
	ax.set_xticklabels( ('0.025', '0.050', '0.10') )


plt.subplots_adjust(left = 0.05, right = 0.98, top = 0.53, bottom = 0.06, hspace = 0.200, wspace = 0.200)



































######################################### Splits (F1) ########################################################

fig5 = plt.figure()
fig5.suptitle('Splits F1s', fontsize=16, x=xtitle, y=ytitle)

xlabel = 'Number of splits'
ylabel = 'F1 Score'
legend_names = ['HDC (convolution)', 'SVM', 'Decision Tree', 'Gradient Boosted Decision Tree']

data['x'] = [2, 3, 4, 5]

# DNA ECOLI
data[categories[0]]['hdc'] = [0.87, 0.86, 0.87, 0.87]
data[categories[0]]['svm'] = [0.99, 0.99, 1.0, 1.0]
data[categories[0]]['dt'] = [0.90, 0.93, 0.94, 0.93]
data[categories[0]]['gbdt'] = [0.94, 0.94, 0.95, 0.95]

# DNA@Anod
data[categories[1]]['hdc'] = [0.86, 0.87, 0.86, 0.87]
data[categories[1]]['svm'] = [0.97, 0.97, 0.98, 0.98]
data[categories[1]]['dt'] = [0.91, 0.92, 0.93, 0.93]
data[categories[1]]['gbdt'] = [0.92, 0.95, 0.95, 0.94]

# Inliquid DNA
data[categories[2]]['hdc'] = [0.96, 0.98, 0.97, 0.98]
data[categories[2]]['svm'] = [0.99, 0.99, 0.99, 0.99]
data[categories[2]]['dt'] = [0.97, 0.97, 0.96, 0.96]
data[categories[2]]['gbdt'] = [0.95, 0.96, 0.95, 0.96]

# Yeast inliquid HK
data[categories[3]]['hdc'] = [0.82, 0.85, 0.85, 0.84]
data[categories[3]]['svm'] = [0.98, 0.99, 0.99, 0.99]
data[categories[3]]['dt'] = [0.97, 0.97, 0.97, 0.98]
data[categories[3]]['gbdt'] = [0.92, 0.93, 0.93, 0.94]

# Yeast inliquid Live
data[categories[4]]['hdc'] = [0.86, 0.84, 0.85, 0.83]
data[categories[4]]['svm'] = [0.99, 0.99, 0.99, 0.99]
data[categories[4]]['dt'] = [0.93, 0.94, 0.93, 0.93]
data[categories[4]]['gbdt'] = [0.95, 0.91, 0.92, 0.93]



for i in range(0, len(categories)):
	ax = plot_line_graph(fig5, subplots[i], data['x'], data[categories[i]]['hdc'], data[categories[i]]['svm'], data[categories[i]]['dt'], data[categories[i]]['gbdt'], categories[i], xlabel, ylabel, 1.0, legend_names)
	ax.set_xticks(data['x'])
	ax.set_xticklabels( ('2', '3', '4', '5') )


plt.subplots_adjust(left = 0.05, right = 0.98, top = 0.53, bottom = 0.06, hspace = 0.200, wspace = 0.200)






######################################### LIMITED DATA (F1) ########################################################
fig6 = plt.figure()
fig6.suptitle('Limited Data F1s', fontsize=16, x=xtitle, y=ytitle)

xlabel = 'Number of samples'
ylabel = 'F1 Score'
legend_names = ['HDC (convolution)', 'SVM', 'Decision Tree', 'Gradient Boosted Decision Tree']

data['x'] = [1, 3, 5]

# DNA ECOLI
data[categories[0]]['hdc'] = [0.78, 0.87, 0.84]
data[categories[0]]['svm'] = [0.84, 0.86, 0.86]
data[categories[0]]['dt'] = [0.75, 0.82, 0.87]
data[categories[0]]['gbdt'] = [0.76, 0.83, 0.92]

# DNA@Anod
data[categories[1]]['hdc'] = [0.87, 0.87, 0.87]
data[categories[1]]['svm'] = [0.89, 0.84, 0.88]
data[categories[1]]['dt'] = [0.74, 0.94, 0.91]
data[categories[1]]['gbdt'] = [0.80, 0.90, 0.91]

# Inliquid DNA
data[categories[2]]['hdc'] = [0.91, 0.95, 0.95]
data[categories[2]]['svm'] = [0.86, 0.89, 0.89]
data[categories[2]]['dt'] = [0.69, 0.93, 0.97]
data[categories[2]]['gbdt'] = [0.75, 0.88, 0.91]

# Yeast inliquid HK
data[categories[3]]['hdc'] = [0.82, 0.81, 0.79]
data[categories[3]]['svm'] = [0.81, 0.80, 0.80]
data[categories[3]]['dt'] = [0.80, 0.87, 0.95]
data[categories[3]]['gbdt'] = [0.80, 0.91, 0.89]

# Yeast inliquid Live
data[categories[4]]['hdc'] = [0.83, 0.84, 0.84]
data[categories[4]]['svm'] = [0.81, 0.83, 0.78]
data[categories[4]]['dt'] = [0.82, 0.87, 0.89]
data[categories[4]]['gbdt'] = [0.76, 0.87, 0.91]


for i in range(0, len(categories)):
	ax = plot_line_graph(fig6, subplots[i], data['x'], data[categories[i]]['hdc'], data[categories[i]]['svm'], data[categories[i]]['dt'], data[categories[i]]['gbdt'], categories[i], xlabel, ylabel, 1.0, legend_names)
	ax.set_xticks(data['x'])
	ax.set_xticklabels( ('1', '3', '5') )


plt.subplots_adjust(left = 0.05, right = 0.98, top = 0.53, bottom = 0.06, hspace = 0.200, wspace = 0.200)




######################################### Gaussian Noise (F1) ########################################################

fig7 = plt.figure()
fig7.suptitle('White Gaussian Noise F1s', fontsize=16, x=xtitle, y=ytitle)

xlabel = 'Standard deviation (σ)'
ylabel = 'F1 Score'
legend_names = ['HDC (convolution)', 'SVM', 'Decision Tree', 'Gradient Boosted Decision Tree']

data['x'] = [0.05, 0.10, 0.15]

# DNA ECOLI
data[categories[0]]['hdc'] = [0.84, 0.74, 0.72]
data[categories[0]]['svm'] = [0.80, 0.60, 0.57]
data[categories[0]]['dt'] = [0.60, 0.54, 0.57]
data[categories[0]]['gbdt'] = [0.60, 0.48, 0.47]

# DNA@Anod
data[categories[1]]['hdc'] = [0.90, 0.88, 0.90]
data[categories[1]]['svm'] = [0.94, 0.92, 0.87]
data[categories[1]]['dt'] = [0.77, 0.68, 0.57]
data[categories[1]]['gbdt'] = [0.83, 0.67, 0.51]

# Inliquid DNA
data[categories[2]]['hdc'] = [0.83, 0.84, 0.67]
data[categories[2]]['svm'] = [0.79, 0.69, 0.58]
data[categories[2]]['dt'] = [0.66, 0.59, 0.57]
data[categories[2]]['gbdt'] = [0.61, 0.52, 0.53]

# Yeast inliquid HK
data[categories[3]]['hdc'] = [0.78, 0.75, 0.65]
data[categories[3]]['svm'] = [0.79, 0.75, 0.57]
data[categories[3]]['dt'] = [0.58, 0.54, 0.50]
data[categories[3]]['gbdt'] = [0.61, 0.53, 0.47]

# Yeast inliquid Live
data[categories[4]]['hdc'] = [0.77, 0.73, 0.72]
data[categories[4]]['svm'] = [0.73, 0.74, 0.64]
data[categories[4]]['dt'] = [0.67, 0.50, 0.54]
data[categories[4]]['gbdt'] = [0.69, 0.58, 0.54]



for i in range(0, len(categories)):
	ax = plot_line_graph(fig7, subplots[i], data['x'], data[categories[i]]['hdc'], data[categories[i]]['svm'], data[categories[i]]['dt'], data[categories[i]]['gbdt'], categories[i], xlabel, ylabel, 1.0, legend_names)
	ax.set_xticks(data['x'])
	ax.set_xticklabels( ('0.05', '0.10', '0.15') )


plt.subplots_adjust(left = 0.05, right = 0.98, top = 0.53, bottom = 0.06, hspace = 0.200, wspace = 0.200)





######################################### Multiplicative Noise (F1) ########################################################

fig8 = plt.figure()
fig8.suptitle('Multiplicative Noise F1s', fontsize=16, x=xtitle, y=ytitle)

xlabel = 'Multiplier'
ylabel = 'F1 Score'
legend_names = ['HDC (convolution)', 'SVM', 'Decision Tree', 'Gradient Boosted Decision Tree']

data['x'] = [0.75, 0.90, 1.10, 1.25]

# DNA ECOLI
data[categories[0]]['hdc'] = [0.81, 0.84, 0.85, 0.83]
data[categories[0]]['svm'] = [1.0, 1.0, 1.0, 1.0]
data[categories[0]]['dt'] = [0.94, 0.95, 0.94, 0.94]
data[categories[0]]['gbdt'] = [0.95, 0.95, 0.96, 0.94]

# DNA@Anod
data[categories[1]]['hdc'] = [0.78, 0.74, 0.93, 0.94]
data[categories[1]]['svm'] = [0.96, 0.96, 0.96, 0.97]
data[categories[1]]['dt'] = [0.92, 0.92, 0.90, 0.93]
data[categories[1]]['gbdt'] = [0.94, 0.94, 0.94, 0.95]

# Inliquid DNA
data[categories[2]]['hdc'] = [0.83, 0.84, 0.67, 0.67]
data[categories[2]]['svm'] = [0.79, 0.69, 0.58, 0.59]
data[categories[2]]['dt'] = [0.66, 0.59, 0.57, 0.62]
data[categories[2]]['gbdt'] = [0.61, 0.52, 0.53, 0.41]

# Yeast inliquid HK
data[categories[3]]['hdc'] = [0.78, 0.75, 0.65, 0.61]
data[categories[3]]['svm'] = [0.79, 0.75, 0.57, 0.52]
data[categories[3]]['dt'] = [0.58, 0.54, 0.50, 0.56]
data[categories[3]]['gbdt'] = [0.61, 0.53, 0.47, 0.49]

# Yeast inliquid Live
data[categories[4]]['hdc'] = [0.79 ,0.78, 0.83, 0.81]
data[categories[4]]['svm'] = [0.98, 0.99, 1.0, 0.99]
data[categories[4]]['dt'] = [0.94, 0.95, 0.94, 0.90]
data[categories[4]]['gbdt'] = [0.93, 0.94, 0.94, 0.93]


for i in range(0, len(categories)):
	ax = plot_line_graph(fig8, subplots[i], data['x'], data[categories[i]]['hdc'], data[categories[i]]['svm'], data[categories[i]]['dt'], data[categories[i]]['gbdt'], categories[i], xlabel, ylabel, 1.0, legend_names)
	ax.set_xticks(data['x'])
	ax.set_xticklabels( ('0.75', '0.90', '1.10', '1.25') )


plt.subplots_adjust(left = 0.05, right = 0.98, top = 0.53, bottom = 0.06, hspace = 0.200, wspace = 0.200)




######################################### Additive Noise (F1) ########################################################

fig9 = plt.figure()
fig9.suptitle('Additive Noise F1s', fontsize=16, x=xtitle, y=ytitle)

xlabel = 'Additive constant'
ylabel = 'F1 Score'
legend_names = ['HDC (convolution)', 'SVM', 'Decision Tree', 'Gradient Boosted Decision Tree']

data['x'] = [0.025, 0.050, 0.10]

# DNA ECOLI
data[categories[0]]['hdc'] = [0.84, 0.90, 0.88]
data[categories[0]]['svm'] = [1.0, 1.0, 1.0]
data[categories[0]]['dt'] = [0.95, 0.96, 0.96]
data[categories[0]]['gbdt'] = [0.96, 0.96, 0.97]

# DNA@Anod
data[categories[1]]['hdc'] = [0.92, 0.93, 0.95]
data[categories[1]]['svm'] = [0.96, 0.98, 0.99]
data[categories[1]]['dt'] = [0.94, 0.92, 1.0]
data[categories[1]]['gbdt'] = [0.94, 0.94, 0.99]

# Inliquid DNA
data[categories[2]]['hdc'] = [0.92, 0.98, 0.89]
data[categories[2]]['svm'] = [0.99, 0.98, 0.99]
data[categories[2]]['dt'] = [0.98, 0.97, 0.98]
data[categories[2]]['gbdt'] = [0.97, 0.98, 0.98]

# Yeast inliquid HK
data[categories[3]]['hdc'] = [0.84, 0.90, 0.82]
data[categories[3]]['svm'] = [0.99, 0.98, 0.99]
data[categories[3]]['dt'] = [0.97, 0.97, 0.98]
data[categories[3]]['gbdt'] = [0.95, 0.99, 0.98]

# Yeast inliquid Live
data[categories[4]]['hdc'] = [0.73, 0.75, 0.78]
data[categories[4]]['svm'] = [1.0, 1.0, 1.0]
data[categories[4]]['dt'] = [0.91, 0.95, 0.94]
data[categories[4]]['gbdt'] = [0.93, 0.93, 0.94]


for i in range(0, len(categories)):
	ax = plot_line_graph(fig9, subplots[i], data['x'], data[categories[i]]['hdc'], data[categories[i]]['svm'], data[categories[i]]['dt'], data[categories[i]]['gbdt'], categories[i], xlabel, ylabel, 1.0, legend_names)
	ax.set_xticks(data['x'])
	ax.set_xticklabels( ('0.025', '0.050', '0.10') )


plt.subplots_adjust(left = 0.05, right = 0.98, top = 0.53, bottom = 0.06, hspace = 0.200, wspace = 0.200)







plt.show()