# BpCDMinference-
Bayesian Inference of a point Compound Dislocation model based on forward model of Nikkhoo, M., Walter, T. R., Lundgren, P. R., Prats-Iraola, P. (2016)


Main file is pCDM_BI_JC.py

change variables in the if __name__ == __main__ to use own data 
Does not provide downsampling, sill nugget, and range estimate. This needs adding as code, but for now can be done in GBIS 

Inputs are a 
dict of initial guesses for values 'custom_initial'
dict of custom priors 'custom_priors'
dict of learning rates 'custom_learning_rates'
dict of max_step_sizes 'Max_step_sizes'

The learning rates are the initial learning rate for each value, which is adaptively adjusted to produce a uniform change across the params i.e the change in each param causes a similar amount of change in the output model. 

The maximum step size is there so that this adaptive step size does not blow up to very large values. 
<img width="1995" height="1425" alt="Figure_2" src="https://github.com/user-attachments/assets/284e7c63-4aae-46d0-8d3d-afdce6eec0f3" />
<img width="1500" height="800" alt="Figure_3" src="https://github.com/user-attachments/assets/dddeec4e-e678-4118-ac53-59f7111c674a" />
<img width="2053" height="1322" alt="Figure_4" src="https://github.com/user-attachments/assets/e5ede3e3-22ed-4c20-8a7b-7e2dee67987d" />
<img width="2213" height="1359" alt="Figure_7" src="https://github.com/user-attachments/assets/1b0dedc1-0848-4154-87df-97a828d022cd" />
<img width="1200" height="800" alt="Figure_5" src="https://github.com/user-attachments/assets/f6848020-2f6d-43a8-95df-f05de696949e" />
<img width="2035" height="924" alt="Figure_6" src="https://github.com/user-attachments/assets/74230810-933d-4565-a2bd-1914ecd09152" />
<img width="1476" height="593" alt="Screenshot from 2025-10-17 16-47-13" src="https://github.com/user-attachments/assets/773af76a-9e59-4801-9da5-b1f4c1b181c9" />
<img width="1491" height="592" alt="Screenshot from 2025-10-17 16-48-21" src="https://github.com/user-attachments/assets/22a645dc-d876-4c40-8000-978c4782a309" />
