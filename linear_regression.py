import tensorflow as tf
import numpy as np
import tempfile
from multiprocessing import Process
import multiprocessing

def start_local(pool, ps_hosts, worker_hosts, job_name, task_index, train_x, train_y):
    pool.apply_async(distributed_train, (ps_hosts, worker_hosts, job_name, task_index, train_x, train_y))

def distributed_train(ps_hosts, worker_hosts, job_name, task_index, train_x, train_y):
    # create cluster
    cluster = tf.train.ClusterSpec({'ps': ps_hosts, 'worker': worker_hosts})
    server = tf.train.Server(cluster, job_name=job_name, task_index=task_index)

    if job_name == 'ps':
        server.join()
    elif job_name == "worker":
        with tf.device(tf.train.replica_device_setter(
                worker_device="/job:worker/task:%d" % task_index,
                cluster=cluster)):

            x_data = tf.placeholder(tf.float32, [100])
            y_data = tf.placeholder(tf.float32, [100])

            W = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
            b = tf.Variable(tf.zeros([1]))
            y = tf.add(tf.multiply(x_data, W), b)           #W * x_data + b
            loss = tf.reduce_mean(tf.square(y - y_data))

            global_step = tf.Variable(0, name="global_step", trainable=False)
            optimizer = tf.train.GradientDescentOptimizer(0.1)
            train_op = optimizer.minimize(loss, global_step=global_step)

            tf.summary.scalar('cost', loss)
            summary_op = tf.summary.merge_all()
            init_op = tf.global_variables_initializer()

        # The StopAtStepHook handles stopping after running given steps.
        hooks = [tf.train.StopAtStepHook(last_step=300)]
        # The MonitoredTrainingSession takes care of session initialization,
        # restoring from a checkpoint, saving to a checkpoint, and closing when done
        # or an error occurs.
        with tf.train.MonitoredTrainingSession(master="grpc://" + worker_hosts[task_index],
                                               is_chief=(task_index == 0),
                                               # set the task with index 0 to be the chief task for parameter initializing,
                                               # save summary and recover
                                               checkpoint_dir="/tmp/tf_train_logs",
                                               save_checkpoint_secs=None,
                                               hooks=hooks) as mon_sess:
            while not mon_sess.should_stop():
                # Run a training step asynchronously.
                # See `tf.train.SyncReplicasOptimizer` for additional details on how to
                # perform *synchronous* training.
                # mon_sess.run handles AbortedError in case of preempted PS.
                # train_x = np.random.rand(100).astype(np.float32)
                # train_y = train_x * 0.1 + 0.3
                _, step, loss_v, weight, biase = mon_sess.run([train_op, global_step, loss, W, b],
                                                              feed_dict={x_data: train_x, y_data: train_y})
                if step > 0:
                    print("step: %d, weight: %f, biase: %f, loss: %f" % (step, weight, biase, loss_v))
                print("Optimization finished.")