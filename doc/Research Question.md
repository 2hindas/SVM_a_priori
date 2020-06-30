#### Research Questions



###### Keywords

Support Vector Machines, Virtual Support Vectors, Invariances, Ensemble Learning



###### Research Questions

Why would ensemble training be used to incorporate invariances in classification by training multiple support vector machines on virtual support vector subsets?  



###### Sub questions

How does the number of sub machines affect the accuracy of the complete machine?

How should the virtual support vectors be divided over the support vector machines?

Should VSV subsets be chosen uniformly according to the bootstrap aggregating model, or selected with a priori knowledge. 

To what extent can a speedup be achieved by training multiple support vector machines on subsets of virtual support vectors, instead of one machine on all virtual support vectors?

How should the predictions of multiple support vector machines be combined?







Try out:

randomly sample the ORIGINAL dataset
create SV each time and run the VSV algorithm for each base classifier

TRIED:
Prune train:
Sample SV's with replacement and create invariances then train SMOL dataset
Did not really work. 4.3 vs 5.0 or something

TRY:
create invariances then sample with replacement SMOL dataset
create invariances then sample without replacement SMOL dataset 




#### 