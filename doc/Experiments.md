

## Experiments



#### How does the number of machines affect the performance of the ensemble?

```python
Number of divisions: 2

times_2 = [105.1088, 103.4469, 103.3686, 103.4656, 103.5224, 103.9007, 103.6636, 102.864, 102.5188, 102.8258]
errors_2 = [3.4895, 3.7388, 3.6391, 3.3898, 3.6391, 3.5394, 3.5892, 3.5394, 3.4895, 3.4397]
mean_error_2 = 3.5493500000000004

Number of divisions: 3

times_3 = [82.1375, 82.263, 82.1243, 82.2614, 82.2969, 82.6667, 82.7804, 83.009, 82.4172, 82.9398]
errors_3 = [3.6391, 3.6889, 3.5394, 3.5394, 3.6391, 3.4397, 3.8883, 3.7388, 3.4895, 3.5394]
mean_error_3 = 3.6141600000000005

Number of divisions: 4

times_4 = [71.4064, 70.8728, 71.0048, 71.3184, 71.0637, 70.8885, 71.2327, 71.2218, 71.3399, 71.3489]
errors_4 = [3.8385, 3.7886, 3.7886, 3.6889, 3.7886, 3.6889, 3.9382, 3.8883, 3.6889, 3.7388]
mean_error_4 = 3.7836299999999996

Number of divisions: 5

times_5 = [63.5363, 63.7566, 63.8012, 63.7853, 63.6895, 63.7201, 63.7372, 63.7533, 63.8887, 63.7648]
errors_5 = [3.988, 3.8883, 3.7388, 3.8883, 3.7388, 3.6391, 3.9382, 3.7388, 3.7388, 3.8385]
mean_error_5 = 3.81356

Number of divisions: 6

times_6 = [58.3003, 57.8809, 59.4827, 57.7945, 57.4311, 57.9001, 57.5244, 57.5924, 57.5922, 57.6198]
errors_6 = [3.8883, 3.9382, 3.8385, 3.8385, 3.7886, 3.7388, 3.8385, 3.8883, 4.0379, 3.7886]
mean_error_6 = 3.85842

Number of divisions: 7

times_7 = [53.1967, 53.0843, 53.4004, 53.303, 53.1468, 54.4173, 53.2625, 53.2514, 53.2313, 53.4495]
errors_7 = [3.7886, 3.8385, 4.0877, 3.8883, 3.9382, 4.0379, 3.9382, 3.9382, 3.988, 3.8883]
mean_error_7 = 3.9331900000000006

Number of divisions: 8

times_8 = [49.5869, 49.6032, 49.619, 49.8748, 49.6068, 49.591, 49.6124, 49.7423, 49.5545, 49.5455]
errors_8 = [3.6391, 3.988, 3.9382, 4.0379, 3.8385, 3.8385, 3.988, 3.8883, 4.0877, 3.8385]
mean_error_8 = 3.9082700000000004
```

##### 1 machine

```python
e = [3.0409, 3.0409, 3.0409, 3.0409, 3.0409, 3.0409, 3.0409, 3.0409, 3.0409, 3.0409, 3.0409, 3.0409]
e2 = [3.0409, 3.0409, 3.0409, 3.0409, 3.0409, 3.0409, 3.0409, 3.0409, 3.0409, 3.0409, 3.0409, 3.0409]
```

##### 2 machines

```python
e = [3.5892, 3.6391, 3.4895, 3.4397, 3.6391, 3.6391, 3.4895, 3.5892, 3.2901, 3.1406, 3.2901, 3.4895]
e2 = [3.34, 3.5394, 3.5394, 3.4895, 3.4895, 3.6391, 3.2403, 3.5892, 3.5394, 3.4895, 3.3898, 3.4397]
```

##### 3 machines

```python
e = [3.6391, 3.6391, 3.5394, 3.5892, 3.5394, 3.34, 3.7388, 3.4895, 3.8385, 3.7886, 3.4895, 3.8385]
e2 = [3.5394, 3.5892, 3.6391, 3.8385, 3.3898, 3.7388, 3.6391, 3.8883, 3.5394, 3.6889, 3.5394, 3.7388]
```

##### 4 machines

```python
e = [4.0379, 3.8883, 3.7886, 3.8385, 3.8385, 3.8385, 3.6889, 3.8385, 3.7388, 3.7886, 3.7388, 3.8385]
e2 = [3.9382, 3.9382, 3.7886, 3.7388, 3.7388, 3.7388, 3.6391, 3.5892, 3.7388, 3.6889, 3.8385, 3.8385]
```

##### 5 machines

```python
e = [3.9382, 4.0379, 4.0877, 3.988, 3.988, 3.9382, 3.988, 3.9382, 3.9382, 4.1376, 4.1376, 3.8385]
e2 = [3.7388, 3.7388, 3.7886, 3.8883, 3.988, 3.8883, 3.9382, 3.988, 4.0379, 3.7388, 3.8883, 4.0379]
```

##### 6 machines

```python
e = [3.6391, 4.0379, 3.7886, 3.9382, 3.8883, 3.8883, 3.988, 3.7388, 3.8883, 3.988, 3.8385, 3.8385]
e2 = [3.8883, 3.7388, 3.8385, 3.9382, 3.8883, 3.7388, 3.7388, 3.8883, 3.7886, 3.8385, 3.7388, 3.6889]
```

##### 7 machines

```python
e = [4.0379, 4.0877, 3.988, 4.0877, 3.6889, 4.0877, 3.7388, 3.8385, 4.0379, 3.8883, 3.7886, 3.7886]
e2 = [3.8883, 3.988, 3.988, 4.0379, 3.8883, 4.0379, 3.9382, 3.7886, 4.0379, 3.988, 4.0877, 3.8385]
```

##### 8 machines

```python
e = [4.0379, 4.0379, 3.9382, 3.8883, 3.7886, 3.7886, 3.988, 3.9382, 3.9382, 3.8385, 3.988, 4.1376]
e2 = [3.8385, 3.8385, 3.7886, 3.8883, 4.0379, 3.8385, 3.9382, 3.9382, 3.8385, 3.8385, 3.9382, 3.8385]
```

##### 9 machines

```python
e = [4.0877, 4.2373, 3.988, 3.8385, 4.1376, 4.0379, 4.1874, 4.2871, 4.1376, 3.988, 3.7886, 3.988]
e2 = [3.988, 3.9382, 4.0379, 4.0877, 4.0877, 3.988, 4.0877, 4.0877, 3.988, 4.1376, 4.0379, 4.0877]
```

#### How does the number of machines affect the speed of the ensemble?

##### 1 machine

```python
t = [131.7033, 132.2467, 131.0693, 130.8953, 130.99, 130.9401, 134.2763, 135.2952, 135.0455, 137.5707, 138.4344, 138.442]
```

##### 2 machines

```python
t = [104.7408, 104.3827, 104.2331, 105.5216, 105.4213, 105.5073, 105.4762, 106.0643, 105.6314, 105.9786, 105.5897, 105.1523]
```

##### 3 machines

```python
t = [85.9378, 86.1235, 86.1571, 85.9934, 85.3229, 85.8998, 85.4461, 85.6692, 86.0365, 86.3385, 87.0174, 85.9514]
```

##### 4 machines

```python
t = [73.3103, 73.5435, 73.5854, 73.0442, 72.5285, 72.2959, 72.3718, 72.3904, 72.0670, 72.3545, 71.4483, 71.7728]
```

##### 5 machines

```python
t = [64.2603, 64.0795, 64.4433, 64.4566, 64.3479, 63.897, 64.2188, 64.4841, 64.5543, 64.1244, 63.805, 64.1887]
```

##### 6 machines

```python
t = [58.5765, 58.7632, 58.467, 57.7467, 57.9765, 58.5727, 56.7891, 58.0302, 57.1317, 57.1552, 57.0896, 56.9412]
```

##### 7 machines

```python
t = [57.7096, 57.2417, 57.1741, 56.9301, 57.1365, 56.9475, 57.6265, 56.8145, 56.9351, 57.0664, 56.685, 57.0828]
```

##### 8 machines

```python
t = [53.177, 52.7785, 52.7497, 52.9508, 52.817, 53.145, 52.863, 53.5903, 54.6665, 53.5325, 53.6165, 54.0285]
```

##### 9 machines

```python
t = [50.3321, 50.3729, 50.6532, 50.5383, 50.571, 50.648, 50.5795, 50.4015, 50.3209, 50.483, 50.5673, 50.5472]
```

#### How does the selection of the groups affect the performance?





```
Seed 42
Number of divisions: 3 
Run 1
Run 2
Run 3
Run 4
Run 5
Run 6
Run 7
Run 8
Run 9
Run 10
times_3 = [84.4126, 85.2376, 84.5392, 84.3149, 85.6319, 84.6752, 84.407, 84.1703, 84.4867, 84.7868]
errors_3 = [3.7388, 3.988, 3.7886, 3.7388, 3.7886, 3.7886, 3.6889, 3.7388, 3.5892, 3.6889]
mean_error_3 = 3.75372
```





Specs:



```
processor	: 0
vendor_id	: GenuineIntel
cpu family	: 6
model		: 63
model name	: Intel(R) Xeon(R) CPU @ 2.30GHz
stepping	: 0
microcode	: 0x1
cpu MHz		: 2300.000
cache size	: 46080 KB
physical id	: 0
siblings	: 2
core id		: 0
cpu cores	: 1
apicid		: 0
initial apicid	: 0
fpu		: yes
fpu_exception	: yes
cpuid level	: 13
wp		: yes
flags		: fpu vme de pse tsc msr pae mce cx8 apic sep mtrr pge mca cmov pat pse36 clflush mmx fxsr sse sse2 ss ht syscall nx pdpe1gb rdtscp lm constant_tsc rep_good nopl xtopology nonstop_tsc cpuid tsc_known_freq pni pclmulqdq ssse3 fma cx16 pcid sse4_1 sse4_2 x2apic movbe popcnt aes xsave avx f16c rdrand hypervisor lahf_lm abm invpcid_single ssbd ibrs ibpb stibp fsgsbase tsc_adjust bmi1 avx2 smep bmi2 erms invpcid xsaveopt arat md_clear arch_capabilities
bugs		: cpu_meltdown spectre_v1 spectre_v2 spec_store_bypass l1tf mds swapgs itlb_multihit
bogomips	: 4600.00
clflush size	: 64
cache_alignment	: 64
address sizes	: 46 bits physical, 48 bits virtual
power management:

processor	: 1
vendor_id	: GenuineIntel
cpu family	: 6
model		: 63
model name	: Intel(R) Xeon(R) CPU @ 2.30GHz
stepping	: 0
microcode	: 0x1
cpu MHz		: 2300.000
cache size	: 46080 KB
physical id	: 0
siblings	: 2
core id		: 0
cpu cores	: 1
apicid		: 1
initial apicid	: 1
fpu		: yes
fpu_exception	: yes
cpuid level	: 13
wp		: yes
flags		: fpu vme de pse tsc msr pae mce cx8 apic sep mtrr pge mca cmov pat pse36 clflush mmx fxsr sse sse2 ss ht syscall nx pdpe1gb rdtscp lm constant_tsc rep_good nopl xtopology nonstop_tsc cpuid tsc_known_freq pni pclmulqdq ssse3 fma cx16 pcid sse4_1 sse4_2 x2apic movbe popcnt aes xsave avx f16c rdrand hypervisor lahf_lm abm invpcid_single ssbd ibrs ibpb stibp fsgsbase tsc_adjust bmi1 avx2 smep bmi2 erms invpcid xsaveopt arat md_clear arch_capabilities
bugs		: cpu_meltdown spectre_v1 spectre_v2 spec_store_bypass l1tf mds swapgs itlb_multihit
bogomips	: 4600.00
clflush size	: 64
cache_alignment	: 64
address sizes	: 46 bits physical, 48 bits virtual
power management:
```

Memory

cpu count = `2`

virtual memory = `svmem(total=13653561344, available=12692234240, percent=7.0, used=1442799616, free=9946988544, active=912216064, inactive=2521096192, buffers=85061632, cached=2178711552, shared=925696, slab=177098752)`

