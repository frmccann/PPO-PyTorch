## Dependencies
Trained and tested on:
```
Python 3.6
PyTorch 1.0
NumPy 1.15.3
gym 0.10.8
Pillow 5.3.0
```

The code is based off of this repository https://github.com/nikhilbarhate99/PPO-PyTorch. I simply deleted the policy gradient and clipping parts and rewrote them
## WriteUp
Please look at the attached PDF for answers to the questions and more info. 

## Problem 1 instructions
In order to train this model simply run the problem1.py script and specify the location to save the trained weights. There are a bunch of hyperparamters you can mess around with but the ones currently set are the ones I trained the model with. If you want to plot results you can change the plots hyperparamater to True

In addittion if you wish to create a video of the model in action run the eval_model.py script and specifcy a location to save the video and the path from which to save the model you just trained. If the model goal hits a certain threshold it will stop training for that seed and save a copy of that model with the name you specified and the prefix solved. All of the files below work this way. Sometimes it won't hit this threshold because the threshold is very high and only solutions that have basically reached optimal speed hit it. 

## Problem 2 instructions 

The instructions to run this part of the PSET are very similar to those for problem 1. Do the same thing but with the problem2.py file. However, if you wish to change the reward function to the one used for problem 1 you will have to go into the reacher_wall file and uncomment the line in the reward function that sets the wall_penalty to 0. 

You can create a video of the model in the same way as in problem 1, by running the eval_model.py script and changing the loaded enviornment to load reacherWallEnv instead of reacherEnv.

## Problem 3 instructions
The instructions to run this part of the PSET are very similar to those for problem 1. Do the same thing but with the problem2.py file. 

You can create a video of the model in the same way as in problem 1, by running the eval_model.py script and changing the loaded enviornment to load reacherWallEnv instead of pusherEnv.



