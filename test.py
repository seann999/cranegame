import subprocess
import os
import gym
# import cv2
import time
import numpy as np
import json
import signal

# AA
rand = False
linux = False

game_processes = []

# port render_freq msg_freq server
if linux:
    game_processes.append(
        subprocess.Popen("./game_linux.x86_64 5000 10 10 1 abcdef", shell=True, stdout=subprocess.PIPE,
                         preexec_fn=os.setsid))
else:
    game_processes.append(
        # subprocess.Popen("open -a game_mac.app --args 5000 10 10 1 g", shell=True, stdout=subprocess.PIPE,
        subprocess.Popen("open -a game_mac.app --args 5000 10 10 1 i", shell=True, stdout=subprocess.PIPE,

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
    # prepare
    observation, _, _, _ = game.step("move 0 0 0")

    obj = get_extra(observation)
    loc = np.array(obj["coords"]).flatten()

    print("true initial coordinates: %s" % loc)

    # cv2.imshow("frame", np.array(observation["image"][0])[:,:,[2,1,0]])
    # cv2.waitKey(1)

    # act
    if rand:
        x = np.random.uniform(-5, 5, 2)
    else:
        x = [loc[0], loc[2]]

    # new_observation, reward, end_episode, _ = game.step("auto %s 0 %s" % (x[0], x[1]))


    for _ in range(10):
        x = np.random.uniform(-5, 5, 2)
        y = 2  # np.random.uniform(0, 10)

        for k in range(10):
            new_observation, reward, end_episode, _ = game.step("moveTo %s %s %s" % (x[0], y, x[1]))

            #cv2.imshow("frame", np.array(new_observation["image"][0])[:, :, [2, 1, 0]])
            #cv2.waitKey(1)

            new_obj = get_extra(new_observation)
            print("touch sensor: %s" % new_obj["touch"])

        new_observation, reward, end_episode, _ = game.step("toggleClaw")
        # cv2.imshow("frame", np.array(new_observation["image"][0])[:,:,[2,1,0]])
        # cv2.waitKey(1)

        for k in range(5):
            new_observation, reward, end_episode, _ = game.step("move 0 0 0")
            #cv2.imshow("frame", np.array(new_observation["image"][0])[:, :, [2, 1, 0]])
            #cv2.waitKey(1)

            new_obj = get_extra(new_observation)
            print("touch sensor: %s" % new_obj["touch"])

        for k in range(5):
            new_observation, reward, end_episode, _ = game.step("move 0 3 0")
            #cv2.imshow("frame", np.array(new_observation["image"][0])[:, :, [2, 1, 0]])
            #cv2.waitKey(1)

            new_obj = get_extra(new_observation)
            print("touch sensor: %s" % new_obj["touch"])

    # new_observation, reward, end_episode, _ = game.step("autograb")
    new_observation, reward, end_episode, _ = game.step("reset")

    # cv2.imshow("frame", np.array(new_observation["image"][0])[:,:,[2,1,0]])
    # cv2.waitKey(1)

    # results
    new_obj = get_extra(new_observation)
    print("touch sensor: %s" % new_obj["touch"])
    print("reward: %s" % reward)

    print("-" * 10)
