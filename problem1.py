from PPO_continuous import *
from matplotlib import pyplot as plt
from reacher import *
from reacher_wall import *
from pusher import *
def main():
    ############## Hyperparameters ##############
    env_name = "Problem1"
    model_name='Problem1'
    plot_results=False
    render = False
    solved_reward = 200         # stop training if avg_reward > solved_reward
    log_interval = 35           # print avg reward in the interval
    max_episodes = 400        # max training episodes
    max_timesteps = 1500        # max timesteps in one episode
    rewards=[[0],[0],[0]]
    total_episodes=[[0],[0],[0]]
    update_timestep = 750      # update policy every n timesteps
    action_std = 0.5            # constant std for action distribution (Multivariate Normal)
    K_epochs = 80               # update policy for K epochs
    eps_clip = 0.2              # clip parameter for PPO
    gamma = 0.99                # discount factor
    record_video=True
    video_step=100
    lr = 0.0003                 # parameters for Adam optimizer
    betas = (0.9, 0.999)
    
    random_seeds = [0,1,2]
    #############################################
    
    for seed_num,random_seed in enumerate(random_seeds):
        # creating environment
        env = ReacherEnv()
        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]
        
        if random_seed:
            print("Random Seed: {}".format(random_seed))
            torch.manual_seed(random_seed)
            # env.seed(random_seed)
            np.random.seed(random_seed)
        
        memory = Memory()
        ppo = PPO(state_dim, action_dim, action_std, lr, betas, gamma, K_epochs, eps_clip)
        print(lr,betas)
        
        # logging variables
        running_reward = 0
        avg_length = 0
        time_step = 0
        out = cv2.VideoWriter('./gif/problem1.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 30, (640,480)) 
        robot=env.robot
        robot_base = robot.arm.robot_base_pos
        robot.cam.setup_camera(focus_pt=robot_base, dist=3, yaw=55, pitch=-30, roll=0)
        # training loop
        for i_episode in range(1, max_episodes+1):
            state = env.reset()
            for t in range(max_timesteps):
                time_step +=1
                # Running policy_old:
                action = ppo.select_action(state, memory)
                state, reward, done, _ = env.step(action)
                
                # Saving reward and is_terminals:
                memory.rewards.append(reward)
                memory.is_terminals.append(done)
                if i_episode%video_step==0:
                    if record_video:
                        img = robot.cam.get_images(get_rgb=True, get_depth=False)[0]
                        out.write(np.array(img)) 
                # update if its time
                if time_step % update_timestep == 0:
                    ppo.update(memory)
                    memory.clear_memory()
                    time_step = 0
                running_reward += reward
                if render:
                    env.render()
                if done:
                    # print('hit done?')
                    break
            
            avg_length += t
            if i_episode%video_step==0 and record_video:
                out.release()
                out = cv2.VideoWriter('./p1_video/'+str(i_episode)+'.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 30, (640,480))
            # stop training if avg_reward > solved_reward
            if running_reward > (log_interval*solved_reward):
                print("########## Solved! ##########")
                torch.save(ppo.policy.state_dict(), './models/solved_'+model_name+'.pth')
                break
            
            # save every 10 episodes
            if i_episode % 50 == 0:
                torch.save(ppo.policy.state_dict(), './models/'+model_name+'.pth')
                
            # logging
            if i_episode % log_interval == 0:
                avg_length = int(avg_length/log_interval)
                running_reward = int((running_reward/log_interval))
                rewards[seed_num].append(running_reward)
                total_episodes[seed_num].append(total_episodes[seed_num][-1]+log_interval)
                print('Episode {} \t Avg length: {} \t Avg reward: {}'.format(i_episode, avg_length, running_reward))
                running_reward = 0
                avg_length = 0

    avg_rewards=rewards[0]
    # for i,reward in enumerate(rewards):
    #     if i==1:
    #         continue
    #     else:
    #         avg_rewards=[sum(x) for x in zip(avg_rewards, reward)]
    # avg_rewards=[x / len(rewards) for x in avg_rewards]
    if plot_results==True:
        plt.plot(total_episodes[0][1:],rewards[0][1:])
        plt.plot(total_episodes[1][1:],rewards[1][1:])
        plt.plot(total_episodes[2][1:],rewards[2][1:])
        plt.xlabel('Elapsed Episodes')
        plt.ylabel('Avg Reward over Episode')
        plt.title('Reacher Performance Over 3 seeds')
        plt.show()
        
        avg_rewards=rewards[0]
        for i,reward in enumerate(rewards):
            if i==0:
                continue
            else:
                avg_rewards=[sum(x) for x in zip(avg_rewards, reward)]
        avg_rewards=[x / len(rewards) for x in avg_rewards]

        plt.plot(total_episodes[0][1:],avg_rewards[1:])
        plt.xlabel('Elapsed Episodes')
        plt.ylabel('Avg Reward over Episode')
        plt.title('Reacher Avg Performance Over 3 seeds')
        plt.show()


    
if __name__ == '__main__':
    main()

    
