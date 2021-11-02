import mj_envs
import click 
import os
import gym
import numpy as np
import pickle
from mjrl.utils.gym_env import GymEnv
#import allegro_gym
import mj_allegro_envs
DESC = '''
Helper script to visualize demonstrations.\n
USAGE:\n
    Visualizes demonstrations on the env\n
    $ python utils/visualize_demos --env_name relocate-v0\n
'''

# MAIN =========================================================
@click.command(help=DESC)
@click.option('--env_name', type=str, help='environment to load', required= True)
@click.option('--filename', type=str,help='file to load', required= True)
def main(env_name,filename):
    if env_name is "":
        print("Unknown env.")
        return
    if filename is "":
        print("Unknown pickle!")
        return
    demos = [pickle.load(open(filename,'rb'))]
    #demos = pickle.load(open('./demonstrations/'+env_name+'_demos.pickle', 'rb'))
    # render demonstrations
    demo_playback(env_name, demos)

def demo_playback(env_name, demo_paths):
    e = GymEnv(env_name)
    e.reset()
    for path in demo_paths:
        e.set_env_state(path['init_state_dict'])
        actions = path['actions']
        for t in range(actions.shape[0]):
            e.step(actions[t])
            for i in range(10):
                e.env.mj_render()
            # img = e.env.render(mode='rgb_array')
            # print(img.shape)
if __name__ == '__main__':
    main()
