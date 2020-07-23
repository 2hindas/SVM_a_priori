## Experiment Descriptions





##### 1. Gamma and kernel degree optimizatiion for the USPS dataset

Grid search has been performed on the USPS dataset using 3-fold cross validation. Gamma and the kernel degree have been optimized. For each degree an optimal gamma is determined:

<img src="/Users/tuhindas/Documents/Tuhin/Computer Science/Honours/SVM/SVM_a_priori/results/USPS_GS_gamma_kernel_deg_CV3_figure.png" alt="USPS_GS_gamma_kernel_deg_CV3_figure" style="zoom:24%;float:left" />

```python
No Pooling											  Error: 5.683
Kernel degree: 1  gamma = 0.0060  Error: 4.835
Kernel degree: 2  gamma = 0.0230  Error: 4.337
Kernel degree: 3  gamma = 0.0265  Error: 4.487
Kernel degree: 4  gamma = 0.0110  Error: 4.636
```

Results can be found here: [USPS_GS_gamma_kernel_deg_CV3.csv](/Users/tuhindas/Documents/Tuhin/Computer Science/Honours/SVM/SVM_a_priori/results/USPS_GS_gamma_kernel_deg_CV3.csv) 



