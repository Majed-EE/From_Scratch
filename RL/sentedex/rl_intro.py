import gym
import matplotlib as plt
import numpy as np

# Create the environment



LEARNING_RATE=0.1
DISCOUNT=0.95
EPISODES=2000
SHOW_EVERY=500
epsilon=0.5
START_EPSILON_DECAYING=1 # 20:40 sentdex
END_EPSILON_DECAYING=EPISODES//2
epsilon_decay_value=epsilon/(END_EPSILON_DECAYING-START_EPSILON_DECAYING) # is low then goes to high?




env = gym.make("MountainCar-v0", render_mode="human")  # Older versions may not have render_mode
observation = env.reset()
print(observation) # place and velocity
print(env.observation_space.high)
print(env.observation_space.low)
print(env.action_space.n)

DISCRETE_OS_SIZE=[20]*len(env.observation_space.high) # [20,20]
discrete_os_win_size=(env.observation_space.high-env.observation_space.low)/DISCRETE_OS_SIZE # [ (obs_dis_high-obs_dis_low)/20, (obs_vel_high-obs_vel_low)/20 ]
print(discrete_os_win_size) # quantized os size [place vel]
print(DISCRETE_OS_SIZE) # [20,20] 

q_table=np.random.uniform(low=-2,high=0,size=(DISCRETE_OS_SIZE+[env.action_space.n])) # (20,20,3)
print(q_table.shape)

ep_rewards=[]
aggr_ep_rewards={'ep':[], 'avg':[],"min":[],"max":[]}


def get_discrete_state(state):
    discrete_state=(state-env.observation_space.low)/discrete_os_win_size
    return tuple(discrete_state.astype(int)) # if discrete state is [12.4423,9.12]--> (12, 9)



for episode in range(EPISODES):
    episode_reward=0
    if episode % SHOW_EVERY==0:
        print(f"shwoing episode: {episode} ")
        render = True
    else:
        render = False

    discrete_state=get_discrete_state(env.reset()[0]) # gives initial state and something else

    done=False


    while not done:
        
        if np.random.random() > epsilon: # exploit
            # Get action from Q table
            action = np.argmax(q_table[discrete_state])
        else: # explore
            # Get random action
            action = np.random.randint(0, env.action_space.n)

        ################################## Action state ######################################
        # action= np.argmax(q_table[discrete_state])
        new_state,reward,termination,turncation,info=env.step(action)
        episode_reward+=reward
        new_discrete_state=get_discrete_state(new_state)
        if render:
            # env.render()
            print("render episode")



        if not done:
            ############################# update state #####################################
            # Q learning control Algorithm
            # Q(S,A) <-- Q(S,A) + learning_rate * (R + discount * max( Q(S`,a`)) - Q(S,A) ) --> equation from lec 5 page 41 
            max_future_q = np.max(q_table[new_discrete_state]) #  max( Q(S`,a`)) (across all action (a` which is max))
            current_q = q_table[discrete_state+(action,)] # getting the q value for that action--> q_table[(9,4) +(1,)]-->  q_table[9][4][action]
            # Q(S,A) <-- Q(S,A) + learning_rate * (R + discount * max( Q(S`,a`)) - Q(S,A) ) --> equation from lec 5 page 41 
            new_q = (1-LEARNING_RATE) * current_q + LEARNING_RATE * ( reward + DISCOUNT * max_future_q)
            q_table[discrete_state+(action,)] = new_q # update q table
            

        elif new_state[0]>=env.goal_position:
            print(f"we made it on episode {episode}")
            q_table[discrete_state+(action,)]=0

        discrete_state=new_discrete_state # update state
            
        # Decaying is being done every episode if episode number is within decaying range
        if END_EPSILON_DECAYING >= episode >= START_EPSILON_DECAYING:
            epsilon -= epsilon_decay_value
        
        
        ep_rewards.append(episode_reward)

        if not episode%SHOW_EVERY:
            average_reward=sum(ep_rewards[-SHOW_EVERY:])/len(ep_rewards[-SHOW_EVERY:])
            aggr_ep_rewards["ep"].append(episode)
            aggr_ep_rewards["avg"].append(average_reward)
            aggr_ep_rewards["min"].append(min(ep_rewards[-SHOW_EVERY:]))
            aggr_ep_rewards["max"].append(max(ep_rewards[-SHOW_EVERY:]))
            # aggr_ep_rewards["ep"].append(episode)

            print(f"Episode:{episode } avg: {average_reward} min: {min(ep_rewards[-SHOW_EVERY:])} max {max(ep_rewards[-SHOW_EVERY:])}")


    print(f"episode number: {episode}")


env.close()


plt.plot(aggr_ep_rewards["ep"], aggr_ep_rewards["avg"],label="avg")
plt.plot(aggr_ep_rewards["ep"], aggr_ep_rewards["min"],label="min")
plt.plot(aggr_ep_rewards["ep"], aggr_ep_rewards["max"],label="max")
plt.legend(loc=4)
plt.show()
