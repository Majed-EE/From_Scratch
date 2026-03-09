import numpy as np
total_samples=100
rent_mean_a=4
rent_mean_b=5
ret_mean_a=3
ret_mean_b=2 
max_car = 20
sample_ret_a=np.random.poisson(lam=ret_mean_a,samples=total_samples)
sample_ret_b=np.random.poisson(lam=ret_mean_b,samples=total_samples)
sample_rent_a=np.random.poisson(lam=rent_mean_a,samples=total_samples)
sample_rent_a=np.random.poisson(lam=rent_mean_a,samples=total_samples)

state_val =  np.zeros(max_car+1,max_car+1)

