import gym
from PPO_continuous import PPO, Memory
from PIL import Image
import torch
from reacher import *
from reacher_wall import *
import cv2
from pusher import *
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def test():
    ############## Hyperparameters ##############
    env_name = 'Problem2_original'
    env = ReacherWallEnv()
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    
    n_episodes = 1          # num of episodes to run
    max_timesteps = 1500    # max timesteps in one episode
    render = True           # render the environment
    save_gif = True        # png images are saved in gif folder
    
    # filename and directory to load model from
    filename = env_name+ ".pth"
    directory = "./models/"

    action_std = 0.5        # constant std for action distribution (Multivariate Normal)
    K_epochs = 80           # update policy for K epochs
    eps_clip = 0.2          # clip parameter for PPO
    gamma = 0.99            # discount factor
    
    lr = 0.0003             # parameters for Adam optimizer
    betas = (0.9, 0.999)
    #############################################
    
    memory = Memory()
    ppo = PPO(state_dim, action_dim, action_std, lr, betas, gamma, K_epochs, eps_clip)
    ppo.policy_old.load_state_dict(torch.load(directory+filename))
    out = cv2.VideoWriter('./p2_video/final_eval_og.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 30, (640,480)) 
    robot=env.robot
    robot_base = robot.arm.robot_base_pos
    robot.cam.setup_camera(focus_pt=robot_base, dist=3, yaw=55, pitch=-30, roll=0)
    for ep in range(1, n_episodes+1):
        ep_reward = 0
        state = env.reset()
        for t in range(max_timesteps):
            action = ppo.select_action(state, memory)
            state, reward, done, _ = env.step(action)
            ep_reward += reward
            if render:
                env.render()
            if save_gif:
                img = robot.cam.get_images(get_rgb=True, get_depth=False)[0]
                out.write(np.array(img))
            if done:
                break
            
        print('Episode: {}\tReward: {}'.format(ep, int(ep_reward)))
        ep_reward = 0
        env.close()
        out.release()
    
if __name__ == '__main__':
    test()
    
    