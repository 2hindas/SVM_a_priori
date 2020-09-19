## Parameters

**Fixed**

$C = 2$  (Decoste and Schölkopf 2002)

Filter stride $= 1$

Base kernel: $K(x, y) = \exp(-\gamma_{\text{rbf}} \| x-y\|^2)$ with  $\gamma_{\text{rbf}} = \displaystyle \frac{1}{\text{number of features}}$

Pooling kernel: $K(x, y) = (\gamma_{\text{poly}} \cdot \langle m(x),  m(y)\rangle + 1)^d$, where $m(x)$ is a function that represents a max-pooling filter with a variable filter size and with stride 1.

The full kernel is defined as:
$$
\begin{split}
K(x, y) & = \text{rbf}(x, y)\ \cdot \ \text{poly}(m(x), m(y)) \\
  & = \exp(-\gamma_{\text{rbf}} \| x-y\|^2)\ \cdot \ (\gamma_{\text{poly}} \cdot \langle m(x), m(y)\rangle + 1)^d 
\end{split}
$$


**Variable**

Filter size $\in \{2, 3\}$

Pooling kernel degree $d$ $\in [1, 4]$

polynomial $\gamma_{\text{poly}} \in [0, 0.003]$ with steps of $0.0005$



Repeat set of experiment for both filter sizes and for combined filter sizes.

Use 3-fold cross validation and grid search to determine the ideal combination of the polling kernel degree and $\gamma$. 

Try this for both types of  $\gamma$ for the RBF kernel

Also show that invariances are actually incorporated by testing on a randomly variated test set 10 times, and take an average at the end. 





## Experiments



##### 1. Gamma and kernel degree Cross Validation (USPS, filter size = 2)

Grid search has been performed on the USPS dataset using 3-fold cross validation for a pooling kernel with a filter size of 2. Gamma and the kernel degree have been optimized. For each degree an optimal gamma is determined. The USPS dataset has been padded at all sides with 2 extra black pixels.

<img src="/Users/tuhindas/Documents/Tuhin/Computer Science/Honours/SVM/SVM_a_priori/results/USPS_GS_gamma_kernel_deg_CV3_size2_figure.png" alt="USPS_GS_gamma_kernel_deg_CV3_size2_figure" style="zoom:24%;float:left" />

```python
No Pooling											  Error: 5.683
Kernel degree: 1  gamma = 0.0060  Error: 4.835
Kernel degree: 2  gamma = 0.0230  Error: 4.337
Kernel degree: 3  gamma = 0.0265  Error: 4.487
Kernel degree: 4  gamma = 0.0110  Error: 4.636
```

Results can be found [here](/Users/tuhindas/Documents/Tuhin/Computer Science/Honours/SVM/SVM_a_priori/results/USPS_GS_gamma_kernel_deg_CV3.csv) 



##### 2. Gamma and kernel degree Cross Validation (USPS, filter size = 3)

The same as experiment 1, but now with a filter size of 3.

<img src="/Users/tuhindas/Documents/Tuhin/Computer Science/Honours/SVM/SVM_a_priori/results/USPS_GS_gamma_kernel_deg_CV3_size3_figure.png" alt="USPS_GS_gamma_kernel_deg_CV3_size3_figure" style="zoom:24%;" /> 

```python
No Pooling											  Error: 5.683
Kernel degree: 1  gamma = 0.0285  Error: 4.586
Kernel degree: 2  gamma = 0.0215  Error: 4.686
Kernel degree: 3  gamma = 0.0075  Error: 4.536
Kernel degree: 4  gamma = 0.0035  Error: 4.636
```

Results can be found [here](USPS_grid_search_gamma_size3.csv) 



##### 3. Gamma and kernel degree Validation optimization (USPS, filter size = 2)

25% of the training set has been turned into a validation set. Again grid search has been performed using this validation set, with a filter size of 2. Gamma and the kernel degree have been optimized. 

<img src="/Users/tuhindas/Documents/Tuhin/Computer Science/Honours/SVM/SVM_a_priori/results/USPS_gamma&amp;kerneldegree_size2_figure.png" alt="USPS_gamma&amp;kerneldegree_size2_figure" style="zoom:24%;" />

```python
No Pooling											 Error: 5.683
Kernel degree: 1  gamma = 0.016  Error: 4.690
Kernel degree: 2  gamma = 0.033  Error: 4.340
Kernel degree: 3  gamma = 0.007  Error: 4.440
Kernel degree: 4  gamma = 0.006  Error: 4.440
```

Results can be found [here](USPS_gamma&kerneldegree_size2.csv) 

