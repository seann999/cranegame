# Crane Game (Claw Machine) Simulator
For a machine learning project.

## Dependencies
* Unity Game Engine

Python packages
* OpenAI Gym (``pip install gym`` for minimum installation)
* websocket (``pip install websocket-client``)
* msgpack (``pip install msgpack-python``)

## Setup
Instructions are based on [here](https://github.com/openai/gym/wiki/Environments)
1. Copy the ``unity`` directory in your ``gym/envs`` directory, so there will be a ``gym/envs/unity/unity_env.py``
2. In ``gym/envs/__init__.py``, append
```register(
    id='Unity-v0',
    entry_point='gym.envs.unity:UnityEnv',
)```

## For manual play
From terminal
#### Linux
```
# ./game_linux.x86_64 <port> <render_every> <msg_server_every> <use_server> <object_spawn>
./test.x86_64 5000 10 0 0 abbbc
```
#### Mac
```
#open -a game_mac.app --args <port> <render_every> <msg_server_every> <use_server> <object_spawn>
open -a game_mac.app --args 5000 10 0 0 abbbc
```

## Controls for manual play
WASD to move, space to grab
Q to toggle claw
Z and X to lower and raise claw

## Spawn codes
* a = die; サイコロ
* b = dumbbell; ダンベル
* c = ball; ボール（テニス）
* d = mug; マグカップ
* e = maraca; マラッカ
* f = ball 2; ボール（バスケ）
* g = ball 3; ボール（サッカー）
* h = Android mascot; アンドロイドマスコット

### Spawn code example
```
# サイコロ２個、サッカーボール２個、マグカップ１個、アンドロイド１個
aaggdh
```
