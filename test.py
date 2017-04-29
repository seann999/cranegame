import subprocess
import os
import gym
# import cv2
import time
import numpy as np
import json
import signal

# AA
rand = True
linux = False

game_processes = []

# port render_freq msg_freq server
if linux:
    game_processes.append(
        subprocess.Popen("./game_linux.x86_64 5000 10 10 1 aaaaaaaaaa", shell=True, stdout=subprocess.PIPE,
                         preexec_fn=os.setsid))
else:
    game_processes.append(
        subprocess.Popen("open -a game_mac.app --args 5000 10 10 1 aaaaaaaaaa", shell=True, stdout=subprocess.PIPE,
                         preexec_fn=os.setsid))

time.sleep(7)

game = gym.make('Unity-v0')
game.configure("5000")

def signal_handler(signal, frame):
    print("killing game processes...")
    for pro in game_processes:
        try:
            os.killpg(os.getpgid(pro.pid), signal.SIGTERM)
            pro.kill()
        except:
            pass

# doesn't always work somehow
signal.signal(signal.SIGINT, signal_handler)

def get_extra(obs):
    data = bytearray(obs["extra"]).decode("utf-8")
    obj = json.loads(data)

    return obj

for i in range(1000):
    # act
    for _ in range(100):
        a = [0, 0]

        a_i = np.random.randint(4)

        if a_i == 0:
            a[0] = 1
        elif a_i == 1:
            a[0] = -1
        elif a_i == 2:
            a[1] = 1
        elif a_i == 3:
            a[1] = -1

        for k in range(10):
            new_observation, reward, end_episode, _ = game.step("move %s 0 %s" % (a[0], a[1]))

            if reward > 0:
                print("reward: %s" % reward)

    # new_observation, reward, end_episode, _ = game.step("autograb")
    new_observation, reward, end_episode, _ = game.step("reset")

    print("-" * 10)
