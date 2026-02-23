import numpy as np



# Number of states and observations
num_states = 9 # 3 resolution, 3 channel states
num_observations = 5 # delay
 
# -------------------------------
# 1. Initial State Distribution
# -------------------------------
pi = np.random.rand(num_states)
pi = pi / pi.sum()

print("Initial State Distribution (pi):")
print(pi)
print("Sum:", pi.sum())
print()

# -------------------------------
# 2. Transition Probability Matrix (9x9) yes
# -------------------------------


P_high_res= np.random.rand(3,3)
P_mid_res= np.random.rand(3,3)
P_low_res= np.random.rand(3,3)
P_high_res= P_high_res/P_high_res.sum(axis=1,keepdims=True)
P_mid_res= P_mid_res/P_mid_res.sum(axis=1,keepdims=True)
P_low_res= P_low_res/P_low_res.sum(axis=1,keepdims=True)

print(f"sum of transition probability matrix: {P_high_res.sum(axis=1)} , {P_mid_res.sum(axis=1)}, {P_low_res.sum(axis=1)} ")
A=np.zeros(num_states,num_states)
A[0:3,0,3]=P_high_res
A[3:6,3:6]=P_mid_res
A[6:,6:] = P_low_res

print("Transition Matrix A (9x9):")
print(A)
print("Row sums:", A.sum(axis=1))
print()

# -------------------------------
# 3. Emission Probability Matrix (9x5)
# -------------------------------
B = np.random.rand(num_states, num_observations)
B = B / B.sum(axis=1, keepdims=True)

print("Emission Matrix B (9x5):")
print(B)
print("Row sums:", B.sum(axis=1))
print()

# -------------------------------
# 4. Observation Labels
# -------------------------------
observations = ["Obs1", "Obs2", "Obs3", "Obs4", "Obs5"]

print("Observation Symbols:")
print(observations)
