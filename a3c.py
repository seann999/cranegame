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

            self.inputs = tf.placeholder(shape=[None, 227, 227, 3], dtype=tf.float32, name="images")
            resized_inputs = tf.image.resize_bilinear(self.inputs, (84, 84))
            self.resized_inputs_ph = tf.placeholder_with_default(resized_inputs, [None, 84, 84, 3], name="resized_images")
            self.train_phase = tf.placeholder_with_default(True, [], name="train")

            """
            hidden = slim.fully_connected(slim.flatten(self.resized_inputs_ph), 128, activation_fn=tf.nn.elu)
            hidden = slim.fully_connected(hidden, 4, activation_fn=tf.nn.elu)
            self.loc_pred = slim.fully_connected(tf.stop_gradient(hidden), 2,
                                                 activation_fn=None,
                                                 biases_initializer=tf.constant_initializer(0))
            rec_hidden = layers.fully_connected(hidden, 128, activation_fn=tf.nn.elu)
            rec_hidden = layers.fully_connected(rec_hidden, 84*84*3, activation_fn=tf.nn.sigmoid)
            self.rec = tf.reshape(rec_hidden, [-1, 84, 84, 3])
            """
            d = 4

            conv = slim.conv2d(activation_fn=tf.nn.elu,
                                     inputs=self.resized_inputs_ph, num_outputs=16,
                                     kernel_size=[8, 8], stride=[4, 4], padding='VALID')

            for _ in range(2):
                conv = slim.conv2d(activation_fn=tf.nn.elu,
                                    inputs=conv, num_outputs=32,
                                    kernel_size=[4, 4], stride=[2, 2], padding='VALID')

            hidden = slim.fully_connected(slim.flatten(conv), 256, activation_fn=tf.nn.elu)
            #hidden = slim.fully_connected(hidden, 8, activation_fn=tf.nn.elu)
            """hidden = tf.contrib.layers.batch_norm(hidden,
                                              updates_collections=None,
                                              scale=False, center=False,
                                              is_training=self.train_phase,
                                              scope="bn")"""

            mean_hidden = slim.fully_connected(hidden, 8, activation_fn=None)
            var_hidden = slim.fully_connected(hidden, 8, activation_fn=None)
            epsilon = tf.random_normal(tf.shape(var_hidden), name='epsilon')
            std_encoder = tf.exp(0.5 * var_hidden)
            #hidden = mean_hidden# + tf.multiply(std_encoder, epsilon)


            #hidden = layers.layer_norm(hidden)
            #hidden = hidden + tf.random_normal(tf.shape(hidden))

            self.loc_pred = slim.fully_connected(tf.stop_gradient(hidden), 2,
                                               activation_fn=None,
                                                  biases_initializer=tf.constant_initializer(0))
            #self.loc_pred = slim.fully_connected(self.loc_pred, 2,
            #                                     activation_fn=None,
            #                                     biases_initializer=tf.constant_initializer(0))

            rec_hidden = slim.fully_connected(hidden, 512 / d, activation_fn=tf.nn.elu)
            rec_hidden = slim.fully_connected(rec_hidden, 7*7*64 / d, activation_fn=tf.nn.elu)
            rec_hidden = tf.reshape(rec_hidden, [-1, 7, 7, 64 / d])
            rec_hidden = slim.conv2d_transpose(rec_hidden, 64 / d, [6, 6], stride=2, padding='VALID', activation_fn=tf.nn.elu)
            rec_hidden = slim.conv2d_transpose(rec_hidden, 32 / d, [6, 6], stride=2, padding='VALID', activation_fn=tf.nn.elu)
            rec_hidden = slim.conv2d_transpose(rec_hidden, 3, [6, 6], stride=2, padding='VALID', activation_fn=tf.nn.sigmoid)
            self.rec = tf.reshape(rec_hidden, [-1, 84, 84, 3])# + tf.get_variable("b_f", [84, 84, 3], tf.float32, initializer=tf.constant_initializer(0))

            KLD = -0.5 * tf.reduce_sum(1 + var_hidden - tf.pow(mean_hidden, 2) - tf.exp(var_hidden),
                                       reduction_indices=1)
            beta = 1.0
            L = tf.reduce_sum(tf.pow(self.rec - resized_inputs, 2.0), axis=(1,2,3))
            self.rec_loss = tf.reduce_mean(L + beta * KLD)

            #hidden = tf.stop_gradient(hidden)
            #hidden = slim.fully_connected(hidden, 256, activation_fn=tf.nn.elu)

            self.policy_mean = slim.fully_connected(hidden, action_size,
                                                    activation_fn=None,
                                                    biases_initializer=tf.constant_initializer(0))
            self.var_z = slim.fully_connected(hidden, 1,
                                               activation_fn=None,
                                                biases_initializer=tf.constant_initializer(0))
            self.policy_var = tf.nn.softplus(self.var_z) + 1e-5
            self.value = slim.fully_connected(hidden, 1,
                                              activation_fn=None,
                                              biases_initializer=tf.constant_initializer(0))

            self.loc_truth = tf.placeholder(tf.float32, [None, 2], name="true_locations")
            self.error_img = tf.pow(self.rec - resized_inputs, 2.0)
            #self.rec_loss = tf.reduce_mean(tf.reduce_sum(self.error_img, axis=(1, 2, 3)))
            self.loc_loss = tf.reduce_mean(tf.reduce_sum((self.loc_pred - self.loc_truth) ** 2.0, axis=1))
            self.sparsity_loss = tf.reduce_mean(tf.reduce_sum(tf.abs(hidden), axis=1))

            if name == master_name:
                self.pretrain_op = tf.train.AdamOptimizer(1e-4).minimize(self.rec_loss + 0.0 * self.sparsity_loss + self.loc_loss)
            else:
                self.actions = tf.placeholder(tf.float32, [None, action_size])
                self.value_truth = tf.placeholder(tf.float32, [None])

                self.advantages = tf.placeholder(tf.float32, [None])

                self.value_loss = 0.5 * tf.reduce_mean(tf.square(self.value_truth - tf.reshape(self.value, [-1])))

                """"#var_tiled = tf.tile(self.policy_var, [1, action_size])
                self.entropy = 0.5 * (tf.log(2.0 * np.pi * self.policy_var) + 1.0)

                a = 0.0

                #self.policy_mean = tf.clip_by_value(self.policy_mean, -5, 5)
                x_power = -0.5 * tf.square(tf.subtract(self.actions, self.policy_mean)) * tf.exp(-tf.log(var_tiled))
                gaussian_nll = tf.reduce_sum(x_power, axis=1) \
                              - 0.5 * (tf.reduce_sum(tf.log(var_tiled), axis=1) + action_size * tf.log(2.0 * np.pi))

                #self.policy_loss = -tf.nn.tanh(tf.reduce_mean(tf.multiply(tf.reduce_sum(gaussian_ll, axis=1), self.advantages)))
                self.policy_loss = -tf.reduce_mean(tf.multiply(gaussian_ll, self.advantages))
                """

                self.entropy = 0.5 * (tf.log(2. * np.pi * self.policy_var) + 1.)

                # Policy loss
                D = action_size
                x_prec = tf.exp(-self.policy_var)
                x_diff = tf.subtract(self.actions, self.policy_mean)
                x_power = tf.square(x_diff) * x_prec * -0.5
                gaussian_nll = (tf.reduce_sum(self.policy_var) + D * tf.log(2. * np.pi)) / 2. - tf.reduce_sum(x_power)
                self.policy_loss = tf.multiply(gaussian_nll, self.advantages)

                self.loss = 1.0 * (5.0 * self.value_loss + self.policy_loss - 1e-2 * self.entropy) + 0.0 * self.rec_loss\
                            + 1.0 * self.loc_loss + 0.0 * self.sparsity_loss# + 1.0 * tf.reduce_mean(1.0 / (tf.minimum(1.0, self.policy_var) ** 4.0))

                local_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, name)
                gradients = tf.gradients(self.loss, local_vars)  # to apply to weights
                self.var_norms = tf.global_norm(local_vars)
                keep = [grad is not None for grad in gradients]
                grads = [grad for k, grad in zip(keep, gradients) if k]
                checks = [tf.check_numerics(grad, grad.name) for grad in grads]
                with tf.control_dependencies(checks):
                    grads, self.grad_norms = tf.clip_by_global_norm(grads, 40.0)
                    #grads = [tf.clip_by_value(grad, -1.0, 1.0) for grad in grads]

                global_vars = [x for k, x in zip(keep, tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, master_name)) if k]


                self.apply_gradients = trainer.apply_gradients(zip(grads, global_vars))

    def pretrain(self, global_episodes, sess, writer):
        files = os.listdir("dataset2/rgb")
        iter = 0
        cv2.namedWindow("rec", cv2.WINDOW_NORMAL)
        self.increment = global_episodes.assign_add(1)

        while True:
            order = np.random.permutation(np.arange(len(files)))
            batch = []

            for i in order:
                img = cv2.imread(os.path.join("dataset2/rgb", files[i]))
                if img is None:
                    continue

                img = img / 255.0
                txt_path = os.path.join("dataset2/meta", files[i][:-4] + ".txt")

                if not os.path.exists(txt_path):
                    continue

                with open(txt_path) as f:
                    loc = f.readline().strip().split(",")
                    loc = [float(loc[0]), float(loc[2])]

                    if abs(loc[0]) > 10 or abs(loc[1]) > 10:
                        continue

                batch.append((img, loc))

                if len(batch) >= 32:
                    imgs, locs = zip(*batch)

                    _, loc_loss, rec_loss, sparsity_loss, rec, loc_pred = sess.run([self.pretrain_op, self.loc_loss, self.rec_loss, self.sparsity_loss, self.rec, self.loc_pred], feed_dict={
                        self.inputs: imgs,
                        self.loc_truth: locs
                    })

                    episode_count = sess.run(self.increment)
                    summary_float(episode_count, "loc loss", float(loc_loss), writer)
                    summary_float(episode_count, "rec loss", float(rec_loss), writer)
                    summary_float(episode_count, "sparsity loss", float(sparsity_loss), writer)
                    print("%i: %s" % (episode_count, rec_loss))

                    if episode_count % 1 == 0:
                        map = np.zeros([16, 16, 1])

                        def draw_dot(loc):
                            loc = [int((i / 5.0 + 1.0) * 16.0 / 2.0) for i in loc]
                            loc[1] = 16 - loc[1]
                            loc = np.clip(loc, 0, 15)
                            map[loc[1], loc[0]] = 1.0

                        draw_dot(loc_pred[0])
                        draw_dot(locs[0])
                        print(loc_pred[0])

                        cv2.imshow("pred map", map)
                        cv2.imshow("orig", imgs[0])
                        cv2.imshow("rec", rec[0])
                        cv2.waitKey(1)



                    batch = []

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

                while not end_episode:


                    for local_t in range(5):

                        value, mu, sigma, entropy, var_z = sess.run(
                            [self.local_network.value, self.local_network.policy_mean, self.local_network.policy_var, self.local_network.entropy,
                             self.local_network.var_z],
                            feed_dict={
                                self.local_network.inputs: [observation]
                            }
                        )

                        eps = 0#max(0, 3.0 - ((episode_count - 40000.0) / 10000.0) * 3.0)

                        if self.worker_i == 0:
                            print(sigma[0], var_z, eps)
                        action = np.random.uniform(-5, 5, 2)#np.random.normal(mu[0], sigma[0] + eps)
                        action = np.clip(action, -5, 5)

                        #print(action)

                        #action = [loc[0], loc[2]]

                        if math.isnan(action[0]) or math.isnan(action[1]):
                            print("nan error in worker #%s" % self.worker_i)
                            print(mu, sigma)
                            print("entropy %s" % entropy)
                            print(action)
                            exit()

                        cmd_msg = "auto %s 9 %s" % (action[0], action[1])

                        #print(loc)
                        distance = np.sqrt(np.sum(np.square([action[0]-loc[0], action[1]-loc[2]])))
                        mu_error = np.sqrt(np.sum(np.square([mu[0, 0]-loc[0], mu[0, 1]-loc[2]])))

                        new_observation, reward, end_episode, _ = self.game.step(cmd_msg)
                        new_observation, misc = process_obs(new_observation)

                        catch += np.floor(reward)
                        #print(np.array(misc["touch"]))
                        #if distance < 1: # prevent box touch exploit
                        #    reward += np.sum(np.array(misc["touch"]) * 0.1)
                        #print(reward)
                        #reward -= distance / 10.0
                        reward = np.clip(reward, -1, 1)

                        episode_buffer.append((observation, action, reward, value[0, 0], [loc[0], loc[2]]))

                        observation = new_observation
                        loc = np.array(misc["coords"])

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

                        value_loss, policy_loss, loc_loss, sp_loss, rec_loss, rec, error_img, entropy_f, grad_norms, var_norms, adv = self.train(
                            episode_buffer, sess, 0, 0)

                        episode_count = sess.run(self.increment)

                        if self.worker_i == 0:
                            print(cmd_msg)
                            cv2.imshow("frame", episode_buffer[0][0])
                            cv2.imshow("reconstruction", rec[0] / np.amax(rec[0]))
                            #cv2.imshow("error image", error_img[0])
                            cv2.waitKey(1)

                        #if episode_count % 1000 == 0:
                            #saver.save(sess, "%s/model" % FLAGS.model_dir, episode_count)

                        if episode_count % 1 == 0:
                            if self.worker_i == 0:
                                print("policy loss", policy_loss)
                                print("value loss", value_loss)
                                print("mu", mu)
                                print("mu error", mu_error)
                                print("var", sigma)
                                print("advantage", adv)

                            print("%i: %f" % (episode_count, episode_reward))
                            summary_float(episode_count, "episode total reward", episode_reward, writer)
                            summary_float(episode_count, "value loss", float(value_loss), writer)
                            summary_float(episode_count, "policy loss", float(policy_loss), writer)
                            summary_float(episode_count, "gradient norm", float(grad_norms), writer)
                            summary_float(episode_count, "variable norm", float(var_norms), writer)
                            summary_float(episode_count, "entropy", float(entropy), writer)
                            summary_float(episode_count, "sigma", float(sigma), writer)
                            #summary_float(episode_count, "distance", float(distance), writer)
                            #summary_float(episode_count, "mu error", float(mu_error), writer)
                            summary_float(episode_count, "catch rate", float(catch), writer)
                            summary_float(episode_count, "advantage", float(adv), writer)
                            summary_float(episode_count, "rec loss", float(rec_loss), writer)
                            summary_float(episode_count, "loc loss", float(loc_loss), writer)
                            summary_float(episode_count, "sparsity loss", float(sp_loss), writer)

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
        #print(rewards, self.value_plus, advantages, discounted_rewards)

        feed_dict = {
            self.local_network.inputs: observations,
            self.local_network.value_truth: discounted_rewards,
            self.local_network.actions: actions,
            self.local_network.advantages: advantages,
            self.local_network.loc_truth: loc
        }

        value_loss, policy_loss, loc_loss, rec_loss, sp_loss, rec, error_img, entropy, grad_norms, var_norms, _ = sess.run([
            self.local_network.value_loss,
            self.local_network.policy_loss,
            self.local_network.loc_loss,
            self.local_network.rec_loss,
            self.local_network.sparsity_loss,
            self.local_network.rec,
            self.local_network.error_img,
            self.local_network.entropy,
            self.local_network.grad_norms,
            self.local_network.var_norms,
            self.local_network.apply_gradients
        ], feed_dict=feed_dict)

        return value_loss, policy_loss, loc_loss, sp_loss, rec_loss, rec, error_img, entropy, grad_norms, var_norms, advantages
