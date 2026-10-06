import numpy as np



# Number of states and observations
channel_state =3 # good(D3) medium(D2) bad(D1)
res_state= 3 # high 700p (R3) , medium 500p (R2), low 300p(R1)
num_states = channel_state*res_state # 3 resolution, 3 channel states
num_observations = 5 # delay

# -------------------------------
# 1. Initial State Distribution
# -------------------------------
pi = np.random.rand(channel_state) # good, medium, bad
pi = pi / pi.sum() 


pi_state = np.zeros((num_states))
pi_state[[0,3,6]] = pi[0]/res_state # good
pi_state[[1,4,7]] =pi[1]/res_state # medium
pi_state[[2,5,8]] =pi[2]/res_state # bad

# the above is done so that the delay may land on any of the valid state based on channel delay
print("Initial State Distribution (pi):")
print(pi_state)
print("Sum:", pi_state.sum())

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
A=np.zeros((num_states,num_states))
A[0:3,0:3]=P_high_res
A[3:6,3:6]=P_mid_res
A[6:,6:] = P_low_res

print("Transition Matrix A (9x9):")
print(A)
print("Row sums:", A.sum(axis=1))
# print()

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
observations = ["L1", "L2", "L3", "L4", "L5"]
states = ["R3D3","R3D2","R3D1","R2D3","R2D2","R2D1","R1D3","R1D2","R1D1"]




print("Observation Symbols:")
print(observations)




def generate_hmm_sequence(pi, A, B,  length=10):
    """
    Generate a sequence from an HMM.
    
    Returns:
        obs_sequence  : list of observation symbols
        state_sequence: list of hidden states (indices)
    """
    
    num_states = A.shape[0]
    observations = B.shape[1]
    
    # Choose initial state
    current_state = np.random.choice(num_states, p=pi)
    
    state_sequence = [current_state]
    # print(f"current state: {state_sequence}")
    obs_sequence = []
    
    # Generate first observation
    obs = np.random.choice(observations, p=B[current_state])
    obs_sequence.append(obs)
    
    # Generate remaining sequence
    for _ in range(length - 1):
        # Transition to next state
        current_state = np.random.choice(num_states, p=A[current_state])
        state_sequence.append(current_state)
        
        # Emit observation
        obs = np.random.choice(observations, p=B[current_state])
        obs_sequence.append(obs)
    
    return obs_sequence, state_sequence








obs_seq,state_seq= generate_hmm_sequence(pi_state,A,B, length=10)
for obs in range(len(obs_seq)):
    print(f"Sample {obs}\nObervation: {observations[obs_seq[obs]]}, State: {states[state_seq[obs]]}")

