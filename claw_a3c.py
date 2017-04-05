
import random
import string
import numpy as np
import inv_a3c as a3c
import tensorflow as tf
import threading
import time
import subprocess
import os
import cv2

flags = tf.app.flags
FLAGS = flags.FLAGS

class RNNLearner():
    def __init__(self):
        self.letters = []

        flags.DEFINE_string("model_dir", "summaries/test00", "model dir")
        flags.DEFINE_integer("threads", 1, "threads")

        gamma = .9

        workers = []
        game_processes = []

        for i in range(FLAGS.threads):
            game_processes.append(subprocess.Popen("./game.x86_64 %s 50 10 1 a" % str(5000 + i),
                                                        shell=True, stdout=subprocess.PIPE, preexec_fn=os.setsid))

        time.sleep(7)

        with tf.device('/cpu:0'):
            self.sess = tf.Session()
            global_episodes = tf.Variable(0, dtype=tf.int32, name='global_t', trainable=False)
            trainer = tf.train.AdamOptimizer(1e-4)#RMSPropOptimizer(1e-4, decay=0.99, epsilon=0.01)
            master_network = a3c.A3C_Network(a3c.master_name, 2, None)

            writer = tf.summary.FileWriter(FLAGS.model_dir, graph_def=self.sess.graph_def)

            for i in range(FLAGS.threads):
                worker = a3c.Worker(i, 2, trainer, global_episodes, True)
                workers.append(worker)

            saver = tf.train.Saver(max_to_keep=5)

        self.sess.run(tf.global_variables_initializer())


        ckpt = tf.train.get_checkpoint_state(FLAGS.model_dir)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(self.sess, ckpt.model_checkpoint_path)
            print("restored %s" % ckpt.model_checkpoint_path)
        else:
            """
            ckpt = tf.train.get_checkpoint_state("vae,adam")
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(self.sess, ckpt.model_checkpoint_path)
                print("restored %s" % ckpt.model_checkpoint_path)
            """
            pass

        with self.sess as sess:
            try:
                coord = tf.train.Coordinator()

                worker_threads = []
                for worker in workers:
                    worker_threads.append(threading.Thread(target=worker.work, args=(gamma, sess, coord, saver, writer)))

                for t in worker_threads:
                    t.start()

                coord.join(worker_threads)
            except KeyboardInterrupt:
                print("saving")
                saver.save(sess, "%s/model" % FLAGS.model_dir, sess.run(global_episodes))
                print("saved")

                for p in game_processes:
                    p.kill()

if __name__ == "__main__":
    cv2.namedWindow("pred map", cv2.WINDOW_NORMAL)
    RNNLearner()
