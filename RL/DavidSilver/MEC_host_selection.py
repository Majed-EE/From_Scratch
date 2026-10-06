import numpy as np
import matplotlib.pyplot as plt
# u_i
UPF_dict= {
    "u_1":{"num_concurrent_apps":None,"bw_11":None,"bw_12":None},
    "u_2":{"num_concurrent_apps":None,"bw_22":None,"bw_23":None}} 

# m_j
MEC_dict = {
    "m_1":{"num_app":None,"total_compute_resource":None},
    "m_2":{"num_app":None,"total_compute_resource":None}
}


APP_dict= {
    "app_1":{"compute_time":None,"data_send":None} # n_j is the compute
}


# latency cost-> num of app on a channel/ BW
# unit of n_ij
# latency cost unit?

# total C = all T_n(i,j) + T_c(j) -> so basically across all apps?


# Poisson Process and Exponential Inter-Arrival Time

lambda_arrival = 5
T = 10 # simulation length

# inter arrival time of app -> is app in a queue eating the bw?
inter_arrivals = np.random.exponential(1/lambda_arrival, 10000)

arrival_times = np.cumsum(inter_arrivals)

# count arrivals within simulation length
num_arrivals = np.sum(arrival_times < T)

print("Number of arrivals in time", T, ":", num_arrivals)

poisson_counts = np.random.poisson(lambda_arrival*T, 10000)
print(poisson_counts)
print("done")


### active app are the one serving plus queue?