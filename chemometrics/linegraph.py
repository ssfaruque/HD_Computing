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


def plot_line_graph(fig, subplot, x, hdc, svm, dt, gbdt, title, xlabel, ylabel, legend_names):
	ax = fig.add_subplot(subplot)

	ax1 = ax.plot(x, hdc, marker = 'o', markerfacecolor = redc, markersize = 5, color = red, linewidth = 2)
	ax2 = ax.plot(x, svm, marker = 'o', markerfacecolor = bluec, markersize = 5, color = blue, linewidth = 2)
	ax3 = ax.plot(x, dt, marker = 'o', markerfacecolor = greenc, markersize = 5, color = green, linewidth = 2)
	ax4 = ax.plot(x, gbdt, marker = 'o', markerfacecolor = yellowc, markersize = 5, color = yellow, linewidth = 2)
	handles = [ax1[0], ax2[0], ax3[0], ax4[0]]

	plt.title(title)
	ax.set_xlabel(xlabel)
	ax.set_ylabel(ylabel)
	plt.ylim(top=100)
	ax.xaxis.set_major_locator(MaxNLocator(integer=True))
	ax.legend( (handles), (legend_names) )

	return ax




subplots = [151, 152, 153, 154, 155]
categories = ['DNA ECOLI', 'Yeast inliquid HK', 'Yeast inliquid Live', 'DNA@Anod', 'Inliquid DNA']
data = {category : {} for category in categories}
xtitle = 0.5
ytitle = 0.6



######################################### LIMITED DATA (ACCURACY) ########################################################
fig1 = plt.figure()
fig1.suptitle('Limited Data Accuracies', fontsize=16, x=xtitle, y=ytitle)

xlabel = 'Number of Samples'
ylabel = 'Accuracy (%)'
legend_names = ['HDC (convolution)', 'SVM', 'Decision Tree', 'Gradient Boosted Decision Tree']

data['x'] = [1, 3, 5]

# DNA ECOLI
data[categories[0]]['hdc'] = [62.23, 75.20, 75.38]
data[categories[0]]['svm'] = [71.18, 75.18, 76.12]
data[categories[0]]['dt'] = [52.12, 71.20, 79.85]
data[categories[0]]['gbdt'] = [56.71, 70.00, 84.31]

# Yeast inliquid HK
data[categories[1]]['hdc'] = [64.11, 57.73, 54.62]
data[categories[1]]['svm'] = [64.82, 62.47, 64.82]
data[categories[1]]['dt'] = [57.18, 71.73, 81.54]
data[categories[1]]['gbdt'] = [63.76, 76.67, 77.54]

# Yeast inliquid Live
data[categories[2]]['hdc'] = [70.82, 72.13, 71.85]
data[categories[2]]['svm'] = [68.00, 68.59, 63.88]
data[categories[2]]['dt'] = [68.24, 77.47, 81.68]
data[categories[2]]['gbdt'] = [62.24, 77.73, 82.46]

# DNA@Anod
data[categories[3]]['hdc'] = [53.65, 56.93, 66.15]
data[categories[3]]['svm'] = [62.71, 54.94, 57.18]
data[categories[3]]['dt'] = [53.76, 82.40, 79.08]
data[categories[3]]['gbdt'] = [54.00, 76.13, 77.23]

# Inliquid DNA
data[categories[4]]['hdc'] = [69.53, 83.20, 80.92]
data[categories[4]]['svm'] = [70.35, 65.65, 69.41]
data[categories[4]]['dt'] = [52.24, 71.07, 80.46]
data[categories[4]]['gbdt'] = [53.76, 67.20, 76.77]


for i in range(0, len(categories)):
	ax = plot_line_graph(fig1, subplots[i], data['x'], data[categories[i]]['hdc'], data[categories[i]]['svm'], data[categories[i]]['dt'], data[categories[i]]['gbdt'], categories[i], xlabel, ylabel, legend_names)
	ax.set_xticks(data['x'])
	ax.set_xticklabels( ('1', '3', '5') )


plt.subplots_adjust(left = 0.05, right = 0.98, top = 0.53, bottom = 0.06, hspace = 0.200, wspace = 0.200)




######################################### Gaussian Noise (ACCURACY) ########################################################

fig2 = plt.figure()
fig2.suptitle('Gaussian Noise Accuracies', fontsize=16, x=xtitle, y=ytitle)

xlabel = 'Gaussian noise'
ylabel = 'Accuracy (%)'
legend_names = ['HDC (convolution)', 'SVM', 'Decision Tree', 'Gradient Boosted Decision Tree']

data['x'] = [0.05, 0.10, 0.15]

# DNA ECOLI
data[categories[0]]['hdc'] = [71.71, 57.91, 44.56]
data[categories[0]]['svm'] = [69.90, 36.90, 21.85]
data[categories[0]]['dt'] = [31.55, 21.23, 20.03]
data[categories[0]]['gbdt'] = [30.82, 18.02, 13.66]

# Yeast inliquid HK
data[categories[1]]['hdc'] = [49.58, 45.89, 36.22]
data[categories[1]]['svm'] = [54.04, 47.38, 26.44]
data[categories[1]]['dt'] = [31.47, 22.08, 20.56]
data[categories[1]]['gbdt'] = [33.92, 19.48, 15.99]

# Yeast inliquid Live
data[categories[2]]['hdc'] = [66.54, 62.56, 58.22]
data[categories[2]]['svm'] = [62.64, 61.24, 51.85]
data[categories[2]]['dt'] = [46.58, 25.20, 25.28]
data[categories[2]]['gbdt'] = [47.60, 29.75, 23.68]

# DNA@Anod
data[categories[3]]['hdc'] = [68.06, 64.66, 60.18]
data[categories[3]]['svm'] = [74.78, 59.53, 52.09]
data[categories[3]]['dt'] = [49.22, 33.43, 26.46]
data[categories[3]]['gbdt'] = [56.55, 36.21, 22.75]

# Inliquid DNA
data[categories[4]]['hdc'] = [66.16, 66.89, 47.10]
data[categories[4]]['svm'] = [59.73, 46.26, 29.22]
data[categories[4]]['dt'] = [42.48, 26.57, 23.90]
data[categories[4]]['gbdt'] = [43.50, 25.62, 25.61]


for i in range(0, len(categories)):
	ax = plot_line_graph(fig2, subplots[i], data['x'], data[categories[i]]['hdc'], data[categories[i]]['svm'], data[categories[i]]['dt'], data[categories[i]]['gbdt'], categories[i], xlabel, ylabel, legend_names)
	ax.set_xticks(data['x'])
	ax.set_xticklabels( ('0.05', '0.10', '0.15') )


plt.subplots_adjust(left = 0.05, right = 0.98, top = 0.53, bottom = 0.06, hspace = 0.200, wspace = 0.200)





######################################### Multiplicative Noise (ACCURACY) ########################################################

fig3 = plt.figure()
fig3.suptitle('Multiplicative Noise Accuracies', fontsize=16, x=xtitle, y=ytitle)

xlabel = 'Multiplicative noise'
ylabel = 'Accuracy (%)'
legend_names = ['HDC (convolution)', 'SVM', 'Decision Tree', 'Gradient Boosted Decision Tree']

data['x'] = [0.75, 0.90, 1.10, 1.25]

# DNA ECOLI
data[categories[0]]['hdc'] = [75.51, 78.82, 80.27, 77.20]
data[categories[0]]['svm'] = [99.11, 100.0, 98.78, 99.89]
data[categories[0]]['dt'] = [90.89, 91.11, 90.54, 89.87]
data[categories[0]]['gbdt'] = [91.56, 91.14, 92.46, 90.86]

# Yeast inliquid HK
data[categories[1]]['hdc'] = [42.59, 42.93, 43.89, 44.45]
data[categories[1]]['svm'] = [98.78, 98.78, 98.46, 98.89]
data[categories[1]]['dt'] = [88.22, 88.68, 89.32, 91.65]
data[categories[1]]['gbdt'] = [80.63, 85.12, 83.78, 82.63]

# Yeast inliquid Live
data[categories[2]]['hdc'] = [68.25, 68.35, 72.01, 70.12]
data[categories[2]]['svm'] = [96.43, 96.90, 98.33, 97.34]
data[categories[2]]['dt'] = [88.98, 90.35, 89.17, 85.79]
data[categories[2]]['gbdt'] = [86.44, 86.75, 86.71, 85.23]

# DNA@Anod
data[categories[3]]['hdc'] = [61.24, 62.48, 70.35, 73.05]
data[categories[3]]['svm'] = [92.98, 93.01, 94.36, 95.34]
data[categories[3]]['dt'] = [86.09, 87.20, 84.95, 88.14]
data[categories[3]]['gbdt'] = [87.01, 87.80, 88.77, 89.58]

# Inliquid DNA
data[categories[4]]['hdc'] = [66.36, 81.10, 80.20, 75.32]
data[categories[4]]['svm'] = [77.24, 90.88, 91.89, 90.66]
data[categories[4]]['dt'] = [69.44, 86.29, 85.92, 85.86]
data[categories[4]]['gbdt'] = [65.81, 83.10, 84.78, 83.77]


for i in range(0, len(categories)):
	ax = plot_line_graph(fig3, subplots[i], data['x'], data[categories[i]]['hdc'], data[categories[i]]['svm'], data[categories[i]]['dt'], data[categories[i]]['gbdt'], categories[i], xlabel, ylabel, legend_names)
	ax.set_xticks(data['x'])
	ax.set_xticklabels( ('0.75', '0.90', '1.10', '1.25') )


plt.subplots_adjust(left = 0.05, right = 0.98, top = 0.53, bottom = 0.06, hspace = 0.200, wspace = 0.200)




######################################### Additive Noise (ACCURACY) ########################################################

fig4 = plt.figure()
fig4.suptitle('Additive Noise Accuracies', fontsize=16, x=xtitle, y=ytitle)

xlabel = 'Additive noise'
ylabel = 'Accuracy (%)'
legend_names = ['HDC (convolution)', 'SVM', 'Decision Tree', 'Gradient Boosted Decision Tree']

data['x'] = [0.025, 0.050, 0.10]

# DNA ECOLI
data[categories[0]]['hdc'] = [78.61, 86.33, 81.92]
data[categories[0]]['svm'] = [99.00, 100.0, 99.89]
data[categories[0]]['dt'] = [91.74, 92.01, 91.48]
data[categories[0]]['gbdt'] = [93.34, 92.09, 92.32]

# Yeast inliquid HK
data[categories[1]]['hdc'] = [48.75, 80.26, 59.98]
data[categories[1]]['svm'] = [98.89, 97.46, 98.89]
data[categories[1]]['dt'] = [95.08, 94.44, 94.44]
data[categories[1]]['gbdt'] = [87.79, 96.53, 96.97]

# Yeast inliquid Live
data[categories[2]]['hdc'] = [63.33, 65.40, 67.85]
data[categories[2]]['svm'] = [98.88, 97.88, 99.10]
data[categories[2]]['dt'] = [85.67, 91.22, 92.15]
data[categories[2]]['gbdt'] = [84.90, 85.50, 87.34]

# DNA@Anod
data[categories[3]]['hdc'] = [71.85, 74.31, 75.59]
data[categories[3]]['svm'] = [94.02, 94.13, 97.33]
data[categories[3]]['dt'] = [90.01, 86.98, 91.21]
data[categories[3]]['gbdt'] = [88.82, 86.10, 87.76]

# Inliquid DNA
data[categories[4]]['hdc'] = [76.22, 79.22, 80.04]
data[categories[4]]['svm'] = [91.79, 90.11, 94.34]
data[categories[4]]['dt'] = [86.33, 86.62, 92.56]
data[categories[4]]['gbdt'] = [84.58, 86.92, 89.58]


for i in range(0, len(categories)):
	ax = plot_line_graph(fig4, subplots[i], data['x'], data[categories[i]]['hdc'], data[categories[i]]['svm'], data[categories[i]]['dt'], data[categories[i]]['gbdt'], categories[i], xlabel, ylabel, legend_names)
	ax.set_xticks(data['x'])
	ax.set_xticklabels( ('0.025', '0.050', '0.10') )


plt.subplots_adjust(left = 0.05, right = 0.98, top = 0.53, bottom = 0.06, hspace = 0.200, wspace = 0.200)





plt.show()


