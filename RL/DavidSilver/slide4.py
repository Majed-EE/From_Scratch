import numpy as np
import matplotlib.pyplot as plt



states=7
action=2

alpha=0.05
gamma=0.1


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

t_episode=100
action_hist=[]
state_hist=[]
reward_hist=[]
episode_state_val_td=[]
episode_state_val_mc_fv=[]
episode_state_val_mc_ev=[]
# big Dhammak S_s
S_s_fv=np.zeros(states)
S_s_ev=np.zeros(states)
# rms td vs MC
rms_td= np.zeros((t_episode,states))
rms_mc_fv = np.zeros((t_episode,states))
rms_mc_ev=np.zeros((t_episode,states))
true_val=np.linspace(0,1,states)


for episode in range(t_episode):
    # t_action=0
    start_state = 3
    episode_state=[start_state]
    episode_action = []
    episode_reward = [0]
    e_visit=np.zeros(states) # episode visit
    state_visit_hist=list([] for _ in range(states))
    next_state = start_state
    termination_state=[0,6]
    step_t=0
    # print(state)
    while True:
        crnt_state=next_state
        state_visit_hist[crnt_state].append(step_t)
        # print(f"length {len(state_visit_hist)}")
        if crnt_state in termination_state: # terminating state
            # TD(0) 
            state_val_td[crnt_state]=state_val_td[crnt_state]+alpha*(R_t_plus_1+gamma*state_val_td[next_state]-state_val_td[crnt_state])
            
            break

        # e_visit[crnt_state]+=1
        
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
        step_t+=1

    # print(f"Episode {episode} summary: ")
    # print(f"State transition history {episode_state}")
    # print(f"Reward history: {episode_reward}")
    # print(f"Action history: {episode_action}")
    # print(f"length state hist {len(state_visit_hist)}")




    # MC update
    # gt for every episode termination    
     
    gt_s_ev = np.zeros(states)
    
    for state in range(1,states-1):# not including terminating states for MC
        # first visit
        gt_fv=0
        gt_ev=0
        # print(f"length state hist {len(state_visit_hist)}")
        if (len(state_visit_hist[state])>0): 
            # state visited for the first time at time t
            count_mc_fv[state]+=1 # state visited for the first time
            start_t=state_visit_hist[state][0] # first visit
            t_crnt=0
            # reward for first visit
            for itr in range(start_t+1,len(episode_reward)):
                gt_fv+=episode_reward[itr]*gamma**(t_crnt)
                t_crnt+=1
            
            ##### for every visit ####
            for crnt_visit in range(len(state_visit_hist[state])): # every time state is visited
                
                count_mc_ev[state]+=1
                start_t = state_visit_hist[state][crnt_visit]
                t_crnt=0
                for itr in range(start_t+1,len(episode_reward)):
                    gt_ev+=episode_reward[itr]*gamma**(t_crnt)
                    t_crnt+=1
        # update S_s
        S_s_ev[state]+=gt_ev
        S_s_fv[state]+=gt_fv 

        
        
        # update value function estimate
        if count_mc_fv[state]!=0:state_val_mc_fv[state] = (S_s_fv[state])/count_mc_fv[state]
        if count_mc_ev[state]!=0:state_val_mc_ev[state] = (S_s_ev[state])/count_mc_ev[state]

        
    
        # incremental mean MC
    episode_state_val_td.append(state_val_td)
    episode_state_val_mc_ev.append(state_val_mc_ev)
    episode_state_val_mc_fv.append(state_val_mc_fv)

        # rms td vs MC
    rms_td[episode] = np.sqrt(np.mean(true_val- state_val_td)**2)
    rms_mc_fv[episode] = np.sqrt(np.mean(true_val- state_val_mc_fv)**2)
    rms_mc_ev[episode] = np.sqrt(np.mean(true_val- state_val_mc_ev)**2)

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

plt.plot(mean_rms_td[1:-1], label='TD(0)', color='blue')
plt.plot(mean_rms_mc_fv[1:-1], label='MC First Visit', color='green')
plt.plot(mean_rms_mc_ev[1:-1], label='MC Every Visit', color='red')


plt.xlabel('Episodes')
plt.ylabel('Root Mean Squared Error')
plt.title('RMS Error for TD(0), MC First Visit, and MC Every Visit')
plt.legend()
plt.grid()
plt.show()



