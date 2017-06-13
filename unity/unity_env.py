# -*- coding: utf-8 -*-

import websocket
import msgpack
import gym
import io
from PIL import Image
from PIL import ImageOps
from gym import spaces
import numpy as np


class UnityEnv(gym.Env):
    def __init__(self):  # 環境が作られたとき
        # self.action_space = spaces.Discrete(3)  # 3つのアクションをセット
        self.depth_image_dim = 32 * 32
        self.depth_image_count = 0
        self.ws = None
        self.observation, _, _ = self.receive()

    def reset(self):
        return self.observation

    def configure(self, *args, **kwargs):
        print("port: %s" % args[0])
        self.ws = websocket.create_connection("ws://localhost:%s/CommunicationGym" % args[0])

    def step(self, action):  # ステップ処理 、actionを外から受け取る

        actiondata = msgpack.packb({"command": str(action)})  # アクションをpack
        self.ws.send(actiondata)  # 送信

        # Unity Process

        observation, reward, end_episode = self.receive()

        return observation, reward, end_episode, {}

    def receive(self):
        if self.ws is None:
            return None, 0, False

        statedata = self.ws.recv()  # 状態の受信
        state = msgpack.unpackb(statedata)  # 受け取ったデータをunpack

        # byte to string keys for Python 3 support
        state2 = {}
        for k, v in state.items():
            state2[k.decode('utf-8')] = v
        state = state2

        image = []
        for i in range(1):
            print(state.keys())
            image.append(Image.open(io.BytesIO(bytearray(state['image'][i]))))
        depth = []
        for i in range(self.depth_image_count):
            d = (Image.open(io.BytesIO(bytearray(state['depth'][i]))))

            # d.save('stephoge.png')

            depth.append(d)  # np.array(ImageOps.grayscale(d)).reshape(self.depth_image_dim))

        observation = {"image": image, "depth": depth, "extra": state["extra"]}
        reward = state['reward']
        end_episode = state['endEpisode']

        return observation, reward, end_episode

    def close(self):  # コネクション終了処理
        if self.ws is not None:
            self.ws.close()  # コネクション終了
