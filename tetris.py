from pufferlib.ocean.tetris import tetris
import time

env = tetris.Tetris(seed=int(time.time() * 1e6))

obs, info = env.reset(seed=int(time.time() * 1e6))
done = False
total_reward = 0

while not done:
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    # frame = env.render() # comment out if you want headless training

    total_reward += reward
    done = terminated or truncated


print(f"Episode finished! Total reward: {total_reward}")
env.close()
