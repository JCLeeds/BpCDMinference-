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
