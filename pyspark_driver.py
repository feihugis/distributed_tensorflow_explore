import numpy as np
from pyspark.sql import SparkSession
from util import get_train_X_Y
import linear_regression
import logging
import multiprocessing
import threading

BATCH_SIZE = 100

def train(ps_hosts, worker_hosts, idx, partition, job_name):
    pool = multiprocessing.Pool(processes=1)
    if job_name == "ps":
        pool.apply_async(linear_regression.distributed_train, (ps_hosts, worker_hosts, "ps", idx, None, None))
        #linear_regression.distributed_train(ps_hosts, worker_hosts, "ps", idx, None, None)
        pool.close()
        #pool.join()
    else:
        X, Y = get_train_X_Y(partition)
        pool.apply_async(linear_regression.distributed_train, (ps_hosts, worker_hosts, "worker", idx, X, Y))
        #linear_regression.distributed_train(ps_hosts, worker_hosts, "worker", idx, X, Y)
        pool.close()
        #pool.join()
    return ["job #{} - task #{} is submitted".format(job_name, idx)]


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


def pyspark_local_test():
    ps_hosts = ["localhost:30000"]
    worker_hosts = ["localhost:30001", "localhost:30002"]

    spark = (SparkSession.builder.appName("Distributed Tensorflow Test").master("local[4]").getOrCreate())
    sc = spark.sparkContext
    sc.setLogLevel("INFO")

    worker_rdd = sc.parallelize(np.random.rand(100*len(worker_hosts)).astype(np.float32), len(worker_hosts)).map(lambda x: (x, x * 0.68 + 1.2))
    ps_rdd = sc.parallelize(ps_hosts, len(ps_hosts))

    ps_rdd.mapPartitionsWithIndex(lambda idx, it: train(ps_hosts, worker_hosts, idx, it, "ps")).count()
    worker_rdd.mapPartitionsWithIndex(lambda idx, it: train(ps_hosts, worker_hosts, idx, it, "worker")).count()

def pyspark_cluster_test(sc, ps_hosts, worker_hosts):
    worker_rdd = sc.parallelize(np.random.rand(100 * 30000).astype(np.float32),
                                len(worker_hosts))\
        .map(lambda x: (x, x * 0.68 + 1.2))
    ps_rdd = sc.parallelize(ps_hosts, len(ps_hosts))

    ps_rdd.mapPartitionsWithIndex(lambda idx, it: train(ps_hosts, worker_hosts, idx, it, "ps")).count()
    worker_rdd.mapPartitionsWithIndex(lambda idx, it: train(ps_hosts, worker_hosts, idx, it, "worker")).count()

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
    pyspark_local_test()

