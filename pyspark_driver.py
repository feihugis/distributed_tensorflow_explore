import numpy as np
from pyspark.sql import SparkSession
from util import get_train_X_Y
import linear_regression
import logging
import multiprocessing
import threading


def train(ps_hosts, worker_hosts, idx, partition, job_name):
    logging.debug("****** Start to train the model")

    pool = multiprocessing.Pool(processes=1)
    if job_name == "ps":
        #pool.apply_async(linear_regression.distributed_train, (ps_hosts, worker_hosts, "ps", idx, None, None))
        linear_regression.distributed_train(ps_hosts, worker_hosts, "ps", idx, None, None)
    else:
        X, Y = get_train_X_Y(partition)
        #pool.apply_async(linear_regression.distributed_train, (ps_hosts, worker_hosts, "worker", idx, X, Y))
        linear_regression.distributed_train(ps_hosts, worker_hosts, "worker", idx, X, Y)
    pool.close()
    pool.join()
    return ["Task # %d done".format(idx)]


def train_test(pool, ps_hosts, worker_hosts, task_index):
    logging.debug("**** Start to train in TensorFlow")
    if task_index == 0 :
        linear_regression.start_local(pool, ps_hosts, worker_hosts, "ps", task_index, None, None)
    else:
        train_x = np.random.rand(100).astype(np.float32)
        train_y = train_x * 0.1 + 0.3
        linear_regression.start_local(pool, ps_hosts, worker_hosts, "worker", task_index - 1, train_x, train_y)


def train_pyspark(ps_hosts, worker_hosts, task_index):
    logging.debug("**** Start to train in TensorFlow")
    pool = multiprocessing.Pool(processes=1)
    if task_index == 0 :
        linear_regression.start_local(pool, ps_hosts, worker_hosts, "ps", task_index, None, None)
    else:
        train_x = np.random.rand(100).astype(np.float32)
        train_y = train_x * 0.1 + 0.3
        linear_regression.start_local(pool, ps_hosts, worker_hosts, "worker", task_index - 1, train_x, train_y)
    pool.close()
    pool.join()
    return ["TF %d task finished".format(task_index)]


def pyspark_test():
    ps_hosts = ["localhost:30000"]
    worker_hosts = ["localhost:30001", "localhost:30002"]

    spark = (SparkSession.builder.appName("Distributed Tensorflow Test").master("local[4]").getOrCreate())
    sc = spark.sparkContext
    sc.setLogLevel("INFO")

    worker_rdd = sc.parallelize(np.random.random(1000)*100, len(worker_hosts)).map(lambda x: (x, x * 0.68 + 1.2))
    ps_rdd = sc.parallelize(ps_hosts, len(ps_hosts))

    ps_rdd.mapPartitionsWithIndex(lambda idx, it: train(ps_hosts, worker_hosts, idx, it, "ps")).count()
    worker_rdd.mapPartitionsWithIndex(lambda idx, it: train(ps_hosts, worker_hosts, idx, it, "worker")).count()


    #rdd = sc.parallelize(range(0,3), 3).mapPartitionsWithIndex(lambda idx, it: train_pyspark(ps_hosts, worker_hosts, idx))
    #rdd.count()


def local_test():
    ps_hosts = ["localhost:30000"]
    worker_hosts = ["localhost:30001", "localhost:30002"]

    pool = multiprocessing.Pool(processes=3)

    train_test(pool, ps_hosts, worker_hosts, 0)
    train_test(pool, ps_hosts, worker_hosts, 1)
    train_test(pool, ps_hosts, worker_hosts, 2)

    pool.close()
    pool.join()

    print("Sub-process done")

if __name__ == "__main__":
    #local_test()
    pyspark_test()

