# -*- coding: utf-8 -*-
# Hanbin Seo (github.com/5eo1ab)
# 2017.10.14.
# run_conv_net.py
# Train Input-Output data, Test Input-Output data => Result
#####################################

import os
import numpy as np
import tensorflow as tf

print(os.getcwd())  # /home/seo1ab/PycharmProjects/CourseWork
if os.getcwd().split('/')[-1] != "assignment_CNN":
    os.chdir("./assignment_CNN")
    print("Changing Working Dir.")


TrainIn, TestIn, TrainOut, TestOut = np.load("./splited_data/TrainIn.npy"), \
                                    np.load("./splited_data/TestIn.npy"),   \
                                    np.load("./splited_data/TrainOut.npy"), \
                                    np.load("./splited_data/TestOut.npy")
print("Shape of TrainInput: {}, "
      "Shape of TrainOutput: {}".format(TrainIn.shape, TrainOut.shape))
  # Shape of TrainInput: (17500, 49152), Shape of TrainOutput: (17500,)
print("Shape of TestInput: {}, "
      "Shape of TestOutput: {}".format(TestIn.shape, TestOut.shape))
  # Shape of TestInput: (7500, 49152), Shape of TestOutput: (7500,)


"""### Convert data shape for Input of convolution 
    Resizing : [No. of record, width, height, depth]
"""
UNIT_SIZE, UNIT_DEPTH = 128, 3
TrainIn = np.reshape(TrainIn, [TrainIn.shape[0], UNIT_SIZE, UNIT_SIZE, UNIT_DEPTH])
TestIn = np.reshape(TestIn, [TestIn.shape[0], UNIT_SIZE, UNIT_SIZE, UNIT_DEPTH])
TrainOut = np.reshape(TrainOut, [TrainOut.shape[0]])
TestOut = np.reshape(TestOut, [TestOut.shape[0]])
print("Reshaping...\nShape of TrainInput: {}, "
      "Shape of TrainOutput: {}".format(TrainIn.shape, TrainOut.shape))
  # Shape of TrainInput: (17500, 128, 128, 3), Shape of TrainOutput: (17500,)
print("Shape of TestInput: {}, "
      "Shape of TestOutput: {}".format(TestIn.shape, TestOut.shape))
  # Shape of TestInput: (7500, 128, 128, 3), Shape of TestOutput: (7500,)

"""### Placeholder, One-hot-coding setting
"""
sizeInputConv = TrainIn.shape[1:]
print(sizeInputConv, type(sizeInputConv))  # (128, 128, 3) <class 'tuple'>

X = tf.placeholder(tf.float32,
                   [None, sizeInputConv[0], sizeInputConv[1], sizeInputConv[2]])
Y = tf.placeholder(tf.uint8, [None])
Learning_Rate = tf.placeholder(tf.float32)
train_bool = tf.placeholder(tf.bool)
keep_prob = tf.placeholder(tf.float32)
print("PlaceHolder complete!")
# One_Hot encoding based on target variable
Y_real = tf.one_hot(Y, 2)


"""### Functions
"""
def He_Init(K_W, K_H, I_F, O_F):
    randomInit = tf.truncated_normal([K_W, K_H, I_F, O_F], stddev=1, mean=0)
    HeInit = randomInit / (tf.sqrt(K_W * K_H * I_F / 2))
    return tf.Variable(HeInit, dtype=tf.float32)
def bias_variable(shape):
    initial = tf.constant(0.0, shape=shape)
    return tf.Variable(initial)
def apply_conv_relu(Input, Weight, layer_nm):
    iniConv = tf.nn.conv2d(Input, Weight, strides=[1, 1, 1, 1], padding='SAME')
    unitConv = tf.add(iniConv, bias_variable([np.shape(Weight)[-1]]))
    batchNorm = tf.layers.batch_normalization(unitConv, momentum=0.9, training=train_bool)
    resTensor = tf.nn.relu(batchNorm, name=layer_nm)
    print("Shape of InputTensor: {}\nShape of Filter: {}\nShape of After Conv: {}".format(
        Input.shape.as_list(), Weight.shape.as_list(), resTensor.shape.as_list()))
    return resTensor
def apply_max_pool(x, kernel_size, stride_size):
    resTensor = tf.nn.max_pool(x, ksize=[1, kernel_size, kernel_size, 1],
                          strides=[1, stride_size, stride_size, 1], padding='VALID')
    print("Shape of Pooled Tensor: {}".format(resTensor.shape.as_list()))
    return resTensor

"""### Graph node setting
"""
Weight1 = He_Init(4, 4, sizeInputConv[-1], 32)
Conv1 = apply_conv_relu(X, Weight1, 'Conv1')
  # Shape of InputTensor: [None, 128, 128, 3]
  # Shape of Filter: [4, 4, 3, 32]
  # Shape of After Conv: [None, 128, 128, 32]
PooledConv1 = apply_max_pool(Conv1, 2, 2)
  # Shape of Pooled Tensor: [None, 64, 64, 32]

Weight2 = He_Init(4, 4, 32, 32)
Conv2 = apply_conv_relu(PooledConv1, Weight2, 'Conv2')
  # Shape of InputTensor: [None, 64, 64, 32]
  # Shape of Filter: [4, 4, 32, 32]
  # Shape of After Conv: [None, 64, 64, 32]
PooledConv2 = apply_max_pool(Conv2, 2, 2)
  # Shape of Pooled Tensor: [None, 32, 32, 32]

Weight3 = He_Init(4, 4, 32, 16)
Conv3 = apply_conv_relu(PooledConv2, Weight3, 'Conv2')
  # Shape of InputTensor: [None, 32, 32, 32]
  # Shape of Filter: [4, 4, 32, 16]
  # Shape of After Conv: [None, 32, 32, 16]
#PooledConv3 = apply_max_pool(Conv3, 2, 2)
  # Shape of Pooled Tensor: [None, 16, 16, 16]


"""### Fully Connected Layer
"""
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)
def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

flatten = tf.contrib.layers.flatten(Conv3)
sizeIn, sizeOut = flatten.shape.as_list()[-1], 1024
W_FC1, B_FC1 = weight_variable([sizeIn, sizeOut]), bias_variable([sizeOut])
print(flatten.shape, W_FC1.shape, B_FC1.shape)
  # (?, 16384) (16384, 1024) (1024,)
H_FC1_In = tf.add(tf.matmul(flatten, W_FC1), B_FC1)
H_FC1_Out = tf.nn.relu(
    tf.layers.batch_normalization(H_FC1_In, momentum=0.9, training=train_bool), name='FC1')
H_FC1_drop = tf.nn.dropout(H_FC1_Out, keep_prob)
print(H_FC1_Out.shape, H_FC1_drop.shape)  # (?, 1024) (?, 1024)

sizeIn, sizeOut = H_FC1_Out.shape.as_list()[-1], 2
W_FC2, B_FC2 = weight_variable([sizeIn, sizeOut]), bias_variable([sizeOut])
print(H_FC1_Out.shape, W_FC2.shape, B_FC2.shape)
H_FC2_In = tf.add(tf.matmul(H_FC1_drop, W_FC2), B_FC2)
Y_pred = tf.nn.relu(
    tf.layers.batch_normalization(H_FC2_In, momentum=0.9, training=train_bool), name='FC2')
print(Y_pred.shape)  # (?, 2)

"""### Optimization
"""
Y_softmax = tf.nn.softmax(Y_pred)
Y_idx = tf.argmax(Y_softmax, axis=1)  # Index_Max ??
SoftMax = tf.nn.softmax_cross_entropy_with_logits(labels=Y_real, logits=Y_pred)
loss = tf.reduce_mean(SoftMax)
tf.summary.scalar('Loss', loss)
# optimizer
optimizer = tf.train.RMSPropOptimizer(Learning_Rate)
train = optimizer.minimize(loss)
# session run
sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)
extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)  # ??
print("Optimization method complete!")

"""### Learning
"""
Tensorboard_Root_Path = os.getcwd()+"/Vis_tensorboard/"
merged = tf.summary.merge_all()
train_writer = tf.summary.FileWriter(Tensorboard_Root_Path + '/train', sess.graph)
test_writer = tf.summary.FileWriter(Tensorboard_Root_Path + '/test')

BATCH_SIZE = 32
for i in range(1000):
    trainIdx = np.random.choice(TrainIn.shape[0], BATCH_SIZE, replace=False)
    _, __ = sess.run([train, extra_update_ops],
                    feed_dict= {
                        X : TrainIn[trainIdx, :, :, :],
                        Y : TrainOut[trainIdx],
                        Learning_Rate: 0.00001, train_bool: True, keep_prob: 1
                    })
    if (i % 10 == 0):
        trainIdx = np.random.choice(TrainIn.shape[0], BATCH_SIZE, replace=False)
        summaryTrain = sess.run(merged,
                               feed_dict={
                                   X: TrainIn[trainIdx, :, :, :],
                                   Y: TrainOut[trainIdx],
                                   Learning_Rate: 0.00001, train_bool: True, keep_prob: 1
                               })
        testIdx = np.random.choice(TestIn.shape[0], BATCH_SIZE, replace=False)
        summaryTrain = sess.run(merged,
                                feed_dict={
                                    X: TrainIn[testIdx, :, :, :],
                                    Y: TrainOut[testIdx],
                                    Learning_Rate: 0.00001, train_bool: True, keep_prob: 1
                                })
        train_writer.add_summary(summaryTrain, i)
        test_writer.add_summary(summaryTrain, i)
        print(i)

"""### Final Inference
"""
Output_List, Iter = list(), TestIn.shape[0]//100
for j in range(Iter):
    testIdx = range(0+(100*j), 100*(j+1))
    Output = sess.run(Y_idx,
                      feed_dict={
                          X: TestIn[testIdx, :, :, :],
                          Y: TestOut[testIdx],
                          Learning_Rate: 0.00001, train_bool: False, keep_prob: 1
                      })
    Output_List.append(Output)
    print("{}/{}".format(j+1, Iter))

Y_Hat=np.concatenate(Output_List,axis=0)
Test_Accuracy=np.sum((TestOut==Y_Hat))/np.shape(Y_Hat)[0]
print("Test Accuracy : "+ str(Test_Accuracy))

