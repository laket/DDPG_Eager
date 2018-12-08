#!/usr/bin/python3

# Copyright 2018 Oiki Tomoaki. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import gym
import time
import argparse
import numpy as np
import tensorflow as tf

import tf_DDPG

tf.enable_eager_execution()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", default="HalfCheetah-v2")
    # TODO load saved model
    parser.add_argument("--policy", default="tf_DDPG")
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--render", default=False, action="store_true")
    parser.add_argument("--parameter", default="parameter", required=True)
    args = parser.parse_args()

    env = gym.make(args.env)

    render = args.render
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])

    # Set seeds
    env.seed(args.seed)
    tf.set_random_seed(args.seed)
    np.random.seed(args.seed)

    # Initialize policy
    if args.policy == "tf_DDPG":
        policy = tf_DDPG.DDPG(state_dim, action_dim, max_action)
    else:
        raise NotImplementedError()

    checkpoint = tf.train.Checkpoint(policy=policy)
    checkpoint.restore(args.parameter)

    eval_episodes = 10

    avg_reward = 0.
    for _ in range(eval_episodes):
        obs = env.reset()

        done = False
        while not done:
            action = policy.select_action(np.array(obs))
            obs, reward, done, _ = env.step(action)

            if render:
                env.render()
                time.sleep(0.02)

            avg_reward += reward

    avg_reward /= eval_episodes

    print ("---------------------------------------")
    print ("Evaluation over %d episodes: %f" % (eval_episodes, avg_reward))
    print ("---------------------------------------")


if __name__ == "__main__":
    main()
