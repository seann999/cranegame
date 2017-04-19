import tensorflow as tf
from tensorflow.contrib import layers
import numpy as np
import scipy.signal
import tensorflow.contrib.slim as slim
import gym
import cv2
import math
import json
import os
import scipy.misc
import tensorflow.contrib.layers as layers
import time

master_name = 'global'
FLAGS = tf.app.flags.FLAGS

class A3C_Network():
    def __init__(self, name, action_size, trainer):
        with tf.variable_scope(name):
            action_size = 5

            self.inputs = tf.placeholder(shape=[None, 227, 227, 3], dtype=tf.float32, name="images")
            self.resized_inputs = tf.image.resize_bilinear(self.inputs, (84, 84))
            self.resized_inputs_ph = tf.placeholder_with_default(self.resized_inputs, [None, 84, 84, 3], name="resized_images")
            self.train_phase = tf.placeholder_with_default(True, [], name="train")

            conv = slim.conv2d(activation_fn=tf.nn.elu,
                                     inputs=self.resized_inputs_ph, num_outputs=16,
                                     kernel_size=[8, 8], stride=[4, 4], padding='VALID')

            conv = slim.conv2d(activation_fn=tf.nn.elu,
                                inputs=conv, num_outputs=32,
                                kernel_size=[4, 4], stride=[2, 2], padding='VALID')

            self.hidden = slim.fully_connected(slim.flatten(conv), 256, activation_fn=tf.nn.elu)

            self.policy_logits = slim.fully_connected(self.hidden, action_size,
                                                    activation_fn=None,
                                                    biases_initializer=tf.constant_initializer(0))
            self.policy = tf.nn.softmax(self.policy_logits)

            self.value = slim.fully_connected(self.hidden, 1,
                                              activation_fn=None,
                                              biases_initializer=tf.constant_initializer(0))

            self.loc_pred = slim.fully_connected(tf.stop_gradient(self.hidden), 2,
                                           activation_fn=None,
                                              biases_initializer=tf.constant_initializer(0))
            self.loc_truth = tf.placeholder(tf.float32, [None, 2], name="true_locations")
            self.loc_loss = tf.reduce_mean(tf.reduce_sum((self.loc_pred - self.loc_truth) ** 2.0, axis=1))

            self.vae_decoder()

            if name == master_name:
                self.pretrain_op = tf.train.AdamOptimizer(1e-4).minimize(self.rec_loss + self.loc_loss)
            else:
                self.actions = tf.placeholder(tf.int32, [None], name="actions")
                self.value_truth = tf.placeholder(tf.float32, [None], name="values")
                self.advantages = tf.placeholder(tf.float32, [None], name="advantages")

                self.value_loss = 0.5 * tf.reduce_mean(tf.square(self.value_truth - tf.reshape(self.value, [-1])))

                log_prob_tf = tf.nn.log_softmax(self.policy_logits)
                self.entropy = -tf.reduce_sum(self.policy * log_prob_tf)

                # Policy loss
                actions_one_hot = tf.one_hot(self.actions, action_size)
                multinomial_nll = -tf.reduce_sum(log_prob_tf * actions_one_hot, 1)
                self.policy_loss = tf.reduce_sum(tf.multiply(multinomial_nll, self.advantages))

                self.loss = 1.0 * (5.0 * self.value_loss + self.policy_loss - 1e-2 * self.entropy) + 0.0 * self.rec_loss\
                            + 1.0 * self.loc_loss

                local_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, name)
                gradients = tf.gradients(self.loss, local_vars)  # to apply to weights
                self.var_norms = tf.global_norm(local_vars)
                keep = [grad is not None for grad in gradients]
                grads = [grad for k, grad in zip(keep, gradients) if k]
                checks = [tf.check_numerics(grad, grad.name) for grad in grads]
                with tf.control_dependencies(checks):
                    grads, self.grad_norms = tf.clip_by_global_norm(grads, 40.0)

                global_vars = [x for k, x in zip(keep, tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, master_name)) if k]


                self.apply_gradients = trainer.apply_gradients(zip(grads, global_vars))

    def vae_decoder(self):
        mean_hidden = slim.fully_connected(self.hidden, 8, activation_fn=None)
        var_hidden = slim.fully_connected(self.hidden, 8, activation_fn=None)
        epsilon = tf.random_normal(tf.shape(var_hidden), name='epsilon')
        hidden_z = mean_hidden + var_hidden * epsilon

        rec_hidden = slim.fully_connected(hidden_z, 512, activation_fn=tf.nn.elu)
        rec_hidden = slim.fully_connected(rec_hidden, 7*7*64, activation_fn=tf.nn.elu)
        rec_hidden = tf.reshape(rec_hidden, [-1, 7, 7, 64])
        rec_hidden = slim.conv2d_transpose(rec_hidden, 64, [6, 6], stride=2, padding='VALID', activation_fn=tf.nn.elu)
        rec_hidden = slim.conv2d_transpose(rec_hidden, 32, [6, 6], stride=2, padding='VALID', activation_fn=tf.nn.elu)
        rec_hidden = slim.conv2d_transpose(rec_hidden, 3, [6, 6], stride=2, padding='VALID', activation_fn=tf.nn.sigmoid)
        self.rec = tf.reshape(rec_hidden, [-1, 84, 84, 3])# + tf.get_variable("b_f", [84, 84, 3], tf.float32, initializer=tf.constant_initializer(0))

        KLD = -0.5 * tf.reduce_sum(1 + var_hidden - tf.pow(mean_hidden, 2) - tf.exp(var_hidden),
                                       reduction_indices=1)
        beta = 1.0
        L = tf.reduce_sum(tf.pow(self.rec - self.resized_inputs, 2.0), axis=(1,2,3))
        self.rec_loss = tf.reduce_mean(L + beta * KLD)

def sync(from_scope, to_scope):
    from_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, from_scope)
    to_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, to_scope)

    assign_ops = []

    for from_var, to_var in zip(from_vars, to_vars):
        assign_ops.append(to_var.assign(from_var))

    return assign_ops


def discount(x, gamma):
    return scipy.signal.lfilter([1], [1, -gamma], x[::-1], axis=0)[::-1]  # ??


def summary_float(step, name, value, summary_writer):
    summary = tf.Summary(
        value=[tf.Summary.Value(tag=name, simple_value=float(value))])
    summary_writer.add_summary(summary, global_step=step)


class Worker():
    def __init__(self, worker_i, action_size, trainer, global_episodes, train):
        self.name = "worker_" + str(worker_i)
        self.trainer = trainer
        self.worker_i = worker_i
        self.global_episodes = global_episodes
        self.increment = self.global_episodes.assign_add(1)
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_mean_values = []
        self.action_size = action_size

        self.local_network = A3C_Network(self.name, action_size, trainer)
        self.sync_op = sync(master_name, self.name)
        self.timestep = 0

        self.train_weights = train

        if train:
            self.game = gym.make('Lis-v2')
            self.game.configure(str(5000 + self.worker_i))

    def sample(self, prob):
        x = np.random.uniform(0, 1)
        s = 0
        for i, p in enumerate(prob):
            s += p
            if s > x:
                return i

    def work(self, gamma, sess, coord, saver, writer):
        total_local_steps = 0

        def process_obs(obs):
            data = str(bytearray(obs["extra"]))
            obj = json.loads(data)
            misc = obj

            return np.asarray(obs["image"][0])[:,:,[2,1,0]] / 255.0, misc
            #data = str(bytearray(obs["extra"]))
            #obj = json.loads(data)

            #return np.array(obj).flatten()

        episode_buffer = []
        episode_count = sess.run(self.global_episodes)

        self.game.step("autograb")

        with sess.as_default(), sess.graph.as_default():
            while not coord.should_stop():
                # beginning of an episode

                observation = None

                while observation == None:
                    for _ in range(1):
                        try:
                            observation, _, _, _ = self.game.step("move 0 0 0")
                        except Exception as e:
                            print(e)
                            print("some init error in %i" % self.worker_i)

                observation, misc = process_obs(observation)
                loc = np.array(misc["coords"])
                touch = np.array(misc["touch"])

                episode_reward = 0
                end_episode = False
                catch = 0

                sess.run(self.sync_op)
                steps = 0

                while not end_episode:
                    for local_t in range(5):
                        value, policy, entropy = sess.run(
                            [self.local_network.value, self.local_network.policy, self.local_network.entropy],
                            feed_dict={
                                self.local_network.inputs: [observation]
                            }
                        )

                        action = np.random.randint(5)#self.sample(policy[0])
                        move_x, move_z = 0, 0

                        if action == 0:
                            move_x = 1
                        elif action == 1:
                            move_x = -1
                        elif action == 2:
                            move_z = 1
                        elif action == 3:
                            move_z = -1

                        move_x *= 5.0
                        move_z *= 5.0

                        if steps == 50:
                            target_y = 2.0#np.random.randint(2) * 7 + 2.0

                            for _ in range(20):
                                self.game.step("moveTo 0 %s 0" % target_y)

                        if steps == 100:
                            cmd_msg = "autograb"
                        else:
                            cmd_msg = "move %s %s %s" % (move_x, 0, move_z)

                        steps += 1

                        new_observation, reward, end_episode, _ = self.game.step(cmd_msg)
                        new_observation, misc = process_obs(new_observation)

                        catch += np.floor(reward)
                        reward = np.clip(reward, -1, 1)

                        episode_buffer.append((observation, action, reward, value[0, 0], [loc[0], loc[2]]))

                        print("touch sensor: %s" % misc["touch"])
                        cv2.imshow("frame", observation)
                        cv2.waitKey(1)

                        observation = new_observation

                        t = int(time.time() * 1000)
                        #scipy.misc.imsave("dataset2/rgb/%s.png" % t, observation)
                        #with open("dataset2/meta/%s.txt" % t, "w+") as f:
                        #    f.write(",".join([str(i) for i in loc]))


                        episode_reward += reward
                        total_local_steps += 1

                        if end_episode:
                            break

                    if len(episode_buffer) >= 1 and self.train_weights:
                        if end_episode:
                            bootstrap_value = 0
                        else:
                            bootstrap_value = value[0, 0]

                        #value_loss, policy_loss, pred_loss, sp_loss, rec_loss, rec, error_img, entropy_f, grad_norms, var_norms, adv = self.train(episode_buffer, sess,
                        #                                                                       gamma, bootstrap_value)

                        value_loss, policy_loss, loc_loss, rec_loss, rec, entropy_f, grad_norms, var_norms, adv = self.train(
                            episode_buffer, sess, 0, 0)

                        episode_count = sess.run(self.increment)

                        if self.worker_i == 0:
                            print(cmd_msg)
                            cv2.imshow("frame", episode_buffer[-1][0])
                            cv2.imshow("reconstruction", rec[0] / np.amax(rec[0]))
                            #cv2.imshow("error image", error_img[0])
                            cv2.waitKey(1)

                        #if episode_count % 1000 == 0:
                            #saver.save(sess, "%s/model" % FLAGS.model_dir, episode_count)

                        if episode_count % 1 == 0:
                            if self.worker_i == 0:
                                print("policy loss", policy_loss)
                                print("value loss", value_loss)
                                print("advantage", adv)

                            print("%i: %f" % (episode_count, episode_reward))
                            summary_float(episode_count, "episode total reward", episode_reward, writer)
                            summary_float(episode_count, "value loss", float(value_loss), writer)
                            summary_float(episode_count, "policy loss", float(policy_loss), writer)
                            summary_float(episode_count, "gradient norm", float(grad_norms), writer)
                            summary_float(episode_count, "variable norm", float(var_norms), writer)
                            summary_float(episode_count, "entropy", float(entropy), writer)
                            summary_float(episode_count, "catch rate", float(catch), writer)
                            summary_float(episode_count, "advantage", float(adv[0]), writer)
                            summary_float(episode_count, "rec loss", float(rec_loss), writer)
                            summary_float(episode_count, "loc loss", float(loc_loss), writer)

                        episode_buffer = []

        print("work stopped")

    def train(self, buffer, sess, gamma, bootstrap_value):
        observations, actions, rewards, values, loc = zip(*buffer)

        # The advantage function uses "Generalized Advantage Estimation"
        self.rewards_plus = np.asarray(list(rewards) + [bootstrap_value])
        discounted_rewards = discount(self.rewards_plus, gamma)[:-1]
        self.value_plus = np.asarray(list(values) + [bootstrap_value])
        advantages = rewards + gamma * self.value_plus[1:] - self.value_plus[:-1]
        advantages = discount(advantages, gamma)

        feed_dict = {
            self.local_network.inputs: observations,
            self.local_network.value_truth: discounted_rewards,
            self.local_network.actions: actions,
            self.local_network.advantages: advantages,
            self.local_network.loc_truth: loc
        }

        value_loss, policy_loss, loc_loss, rec_loss, rec, entropy, grad_norms, var_norms, _ = sess.run([
            self.local_network.value_loss,
            self.local_network.policy_loss,
            self.local_network.loc_loss,
            self.local_network.rec_loss,
            self.local_network.rec,
            self.local_network.entropy,
            self.local_network.grad_norms,
            self.local_network.var_norms,
            self.local_network.apply_gradients
        ], feed_dict=feed_dict)

        return value_loss, policy_loss, loc_loss, rec_loss, rec, entropy, grad_norms, var_norms, advantages
