import numpy as np
import matplotlib.pyplot as plt



states=7
action=2

alpha=0.05
gamma=0.4


state_val_mc_ev=np.zeros(states) # every visit
state_val_mc_fv=np.zeros(states) # first visit
state_val_mc_im = np .zeros(states) # incremental mean
count_mc_fv=np .zeros(states) # n(s) first visit
count_mc_ev=np .zeros(states) # n(s) every visit

state_val_td=np.zeros(states) # temporal difference
reward_Ra_s_sd=np.zeros((states,action))
reward_Ra_s_sd[5][1]=1

# f_visit=np.zeros(7)


print("done")

t_episode=10000
action_hist=[]
state_hist=[]
reward_hist=[]
episode_state_val_td=[]
episode_state_val_mc_fv=[]
episode_state_val_mc_ev=[]
# rms td vs MC
rms_td= np.zeros((t_episode,states))
rms_mc_fv = np.zeros((t_episode,states))
rms_mc_ev=np.zeros((t_episode,states))
true_val=np.linspace(0,1,7)

for episode in range(t_episode):
    # t_action=0
    start_state = 3
    episode_state=[start_state]
    episode_action = []
    episode_reward = []
    e_visit=np.zeros(states) # episode visit
    
     # c in the figure 
    # R_t_plus_1=0.0
    next_state = start_state
    
    terminal_State=[0,6]
    while True:
        
        crnt_state=next_state
        if crnt_state in terminal_State:
            # print(f"breaking in terminal state {crnt_state}")
            # terminating state
            break

        e_visit[crnt_state]+=1
        
        sample = np.random.binomial(n=1,p=0.5)
        action = 1 if sample==1 else -1
        episode_action.append(action)
        next_state = crnt_state+action
        episode_state.append(next_state)
        R_t_plus_1 = reward_Ra_s_sd[crnt_state][sample] 
        episode_reward.append(R_t_plus_1)
        # TD(0) update
        state_val_td[crnt_state]=state_val_td[crnt_state]+alpha*(R_t_plus_1+gamma*state_val_td[next_state]-state_val_td[crnt_state])
        # transition happen at the end of the loop

    # MC update
    # gt for every termination        
    gt=0
    for t in range(len(episode_reward)):
        gt+=(gamma**(t))*episode_reward[t]
    for state in range(0,states):
        # what is S(s)-> incremental total return is what?
        # every visit
        S_s_ev=state_val_mc_ev[state]*count_mc_ev[state]
        S_s_fv=state_val_mc_fv[state]*count_mc_fv[state]
        count_mc_ev[state]+=e_visit[state]  # N(s)<-N(s)+1 for every visit
        if count_mc_ev[state]!=0: state_val_mc_ev[state]= ((S_s_ev+gt)/count_mc_ev[state])
        # first visit
        if e_visit[state]>0: 
            count_mc_fv[state]+=1 
        if count_mc_ev[state]!=0:state_val_mc_fv[state] = ((S_s_fv+gt)/count_mc_fv[state])

        # incremental mean MC
    episode_state_val_td.append(state_val_td)
    episode_state_val_mc_ev.append(state_val_mc_ev)
    episode_state_val_mc_fv.append(state_val_mc_fv)

        # rms td vs MC
    rms_td[episode] = np.sqrt(np.mean(true_val- state_val_td)**2)
    rms_mc_fv[episode] = np.sqrt(np.mean(true_val- state_val_mc_fv)**2)
    rms_mc_ev[episode] = np.sqrt(np.mean(true_val- state_val_mc_ev)**2)
    # print(f"Episode {episode} summary: ")
    # print(f"State transition history {episode_state}")
    # print(f"Reward history: {episode_reward}")
    # print(f"Action history: {episode_action}")
    reward_hist.append(episode_reward)
    state_hist.append(episode_state)
    action_hist.append(episode_action)



print(f"state_val_mc_ev: {state_val_mc_ev}")
print(f"state_val_mc_fv: {state_val_mc_fv}")
print(f"state_val_td: {state_val_td}")
print(f"true state val: {true_val}")

# plot
plt.figure(figsize=(10, 6))


mean_rms_td = np.mean(rms_td, axis=1)
mean_rms_mc_fv = np.mean(rms_mc_fv, axis=1)
mean_rms_mc_ev = np.mean(rms_mc_ev, axis=1)

plt.plot(mean_rms_td, label='TD(0)', color='blue')
plt.plot(mean_rms_mc_fv[1:-1], label='MC First Visit', color='green')
plt.plot(mean_rms_mc_ev[1:-1], label='MC Every Visit', color='red')


plt.xlabel('Episodes')
plt.ylabel('Root Mean Squared Error')
plt.title('RMS Error for TD(0), MC First Visit, and MC Every Visit')
plt.legend()
plt.grid()
plt.show()



