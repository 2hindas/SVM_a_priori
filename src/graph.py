import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
from sklearn import svm

results = pd.read_csv('results/USPS_GS_gamma_kernel_deg_CV3.csv', skiprows=15)

plt.yticks(np.arange(1.65, 3.0, 0.1))
plt.xticks(np.arange(0, 0.031, 0.005))
plt.ylabel('Average error')
plt.xlabel('gamma')
plt.title("Grid search for gamma and the kernel degree, using 3-fold cross validation.")

colors = ['', 'orangered', 'darkorange', 'gold', 'limegreen', 'lightseagreen', 'deepskyblue']

for degree in range(1, 5, 1):
    data = results.loc[results['Degree'] == degree]
    data_x = data['Gamma'].values
    data_y = data['Avg Error'].values

    print("2 CV")
    print(f"Degree: {degree}")
    index = np.argmin(data_y)
    print(data_x[index])
    print(data_y[index])

    plt.plot(data_x, data_y, color=colors[degree], label=f'Degree = {degree}', alpha=0.8)
    plt.scatter(data_x, data_y, color=colors[degree], marker=".", s=25, alpha=0.8)
    # plt.plot(data_x[index:index+1], data_y[index:index+1],'.', color='red')

plt.legend()
plt.savefig('USPS_GS_gamma_kernel_deg_CV3_figure.eps', bbox_inches='tight', format='eps')
plt.show()
