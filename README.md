# Crane Game (Claw Machine) Simulator
持ち上げられないサッカーボールを追加。

<!-- <img src="images/image1.png " width="400"> -->

## Dependencies
* Unity Game Engine

Python packages
* OpenAI Gym (``pip install gym`` for minimum installation)
* websocket (``pip install websocket-client``)
* msgpack (``pip install msgpack-python``)

## Setup
Instructions are based on [here](https://github.com/openai/gym/wiki/Environments)
1. Copy the ``unity`` directory in your ``gym/envs`` directory, so there will be a ``gym/envs/unity/unity_env.py``
2. In ``gym/envs/__init__.py``, append:

```
register(
    id='Unity-v0',
    entry_point='gym.envs.unity:UnityEnv',
)
```

## For manual play
From terminal

#### Linux
```
# ./game_linux.x86_64 <port> <render_every> <msg_server_every> <use_server> <object_spawn>
./test.x86_64 5000 10 0 0 abbbc
```
#### Mac
```
# open -a game_mac.app --args <port> <render_every> <msg_server_every> <use_server> <object_spawn>
open -a game_mac.app --args 5000 10 0 0 abbbc
```

## Controls for manual play
WASD to move, space to grab

Q to toggle claw (TODO: glitchy)

X and Z to lower and raise claw

## Actions for script/AI play
In ``env.step()``, pass a command string:
* ``move <x> <y> <z>`` move claw by (x, y, z)
* ``moveTo <x> <y> <z>`` move claw a fixed distance towards (x, y, z)
* ``toggleClaw`` toggle grab/release of claw
* ``auto <x> <y> <z>`` automatically move claw to (x, y, z), move down, grab, raise, move over the opening, and release. This makes learning easier because it reduces the length of the required action sequence. (TODO: bug)
* ``reset`` resets the environment: moves the claw back to the center and respawns all objects in new random locations

## Observations
* ``observation["image"]`` provides the RGB camera image of the environment
* ``observation["extra"]`` provides miscellaneous information
  ```
  data = bytearray(observation["extra"]).decode("utf-8")
  obj = json.loads(data)
  # obj is dict
  ```
  * ``obj["touch"]`` is a list of length 2, each number indicating if the corresponding sensor (the yellow tips of the claw) is touching something. For example, ``[1, 0]`` indicates that the left sensor is touching.
  * ``obj["coordinates"]`` is a list of length N*3, where N is the number of objects. It is a concatenation of (x, y, z) coordinates of each object. Currently, there is no way to tell which 3 numbers correspond to which object (TODO).

## Spawn codes
* a = die; サイコロ
* b = dumbbell; ダンベル
* c = ball; ボール（テニス）
* d = mug; マグカップ
* e = maraca; マラッカ
* f = ball 2; ボール（バスケ）
* g = ball 3; ボール（サッカー）
* h = Android mascot; アンドロイドマスコット
* i = ball 3; 持ち上げられないボール（サッカー）

### Spawn code example
```
# サイコロ２個、サッカーボール２個、マグカップ１個、アンドロイド１個
aaggdh
```

## Run quick demo
In ``test.py`` you may want to change ``linux=False``.
```
python test.py
```
