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
	plt.ticklabel_format(style='plain',axis='x',useOffset=False)
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

	#ax.spines["top"].set_visible(False)
	#ax.spines["right"].set_visible(False)
	#ax.spines["bottom"].set_visible(False)

	return ax




fig = plt.figure()
subplots = [151, 152, 153, 154, 155]
categories = ['DNA ECOLI', 'Yeast inliquid HK', 'Inliquid DNA', 'DNA@Anod', 'Yeast inliquid Live']
xlabel = 'Number of Samples'
ylabel = 'Accuracy (%)'
legend_names = ['HDC (convolution)', 'SVM', 'Decision Tree', 'Gradient Boosted Decision Tree']

data = {category : {} for category in categories}



######################################### LIMITED DATA (ACCURACY) ########################################################

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

# Inliquid DNA
data[categories[2]]['hdc'] = [70.82, 72.13, 71.85]
data[categories[2]]['svm'] = [68.00, 68.59, 63.88]
data[categories[2]]['dt'] = [68.24, 77.47, 81.68]
data[categories[2]]['gbdt'] = [62.24, 77.73, 82.46]

# DNA@Anod
data[categories[3]]['hdc'] = [53.65, 56.93, 66.15]
data[categories[3]]['svm'] = [62.71, 54.94, 57.18]
data[categories[3]]['dt'] = [53.76, 82.40, 79.08]
data[categories[3]]['gbdt'] = [54.00, 76.13, 77.23]

# Yeast inliquid Live
data[categories[4]]['hdc'] = [69.53, 83.20, 80.92]
data[categories[4]]['svm'] = [70.35, 65.65, 69.41]
data[categories[4]]['dt'] = [52.24, 71.07, 80.46]
data[categories[4]]['gbdt'] = [53.76, 67.20, 76.77]


for i in range(0, len(categories)):
	plot_line_graph(fig, subplots[i], data['x'], data[categories[i]]['hdc'], data[categories[i]]['svm'], data[categories[i]]['dt'], data[categories[i]]['gbdt'], categories[i], xlabel, ylabel, legend_names)



#plt.subplots_adjust(left = 0.035, right = 0.985, top = 0.580, bottom = 0.055, hspace = 0.200, wspace = 0.200)
plt.show()








