import subprocess
import os
import gym
import cv2
import time
import numpy as np
import json

rand = False

game_processes = []

# port render_freq msg_freq server
game_processes.append(subprocess.Popen("./test.x86_64 5000 50 10 1 abc", shell=True, stdout=subprocess.PIPE, preexec_fn=os.setsid))

time.sleep(7)

game = gym.make('Lis-v2')
game.configure("5000")

def get_extra(obs):
    data = str(bytearray(obs["extra"]))
    obj = json.loads(data)

    return obj

for i in range(1000):
    # prepare
    observation, _, _, _ = game.step("-1 -1 -1")

    obj = get_extra(observation)
    loc = np.array(obj["coords"]).flatten()
    
    print("true initial coordinates: %s" % loc)

    cv2.imshow("frame", np.array(observation["image"][0])[:,:,[2,1,0]])
    cv2.waitKey(1)

    # act
    if rand:
        x = np.random.uniform(-5, 5, 2)
    else:
        x = [loc[0], loc[2]]

    new_observation, reward, end_episode, _ = game.step("%s %s 1" % (x[0], x[1]))

    # results
    new_obj = get_extra(new_observation)
    print("touch sensor: %s" % new_obj["touch"])
    print("reward: %s" % reward)

    print("-"*10)