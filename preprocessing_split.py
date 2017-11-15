# -*- coding: utf-8 -*-
# Hanbin Seo (github.com/5eo1ab)
# 2017.10.14.
# preprocessing_split.py
# RGB numeric data => Train Input-Output data, Test Input-Output data
#####################################

import os
import numpy as np

print(os.getcwd())  # /home/seo1ab/PycharmProjects/CourseWork
if os.getcwd().split('/')[-1] != "assignment_CNN":
    os.chdir("./assignment_CNN")
    print("Changing Working Dir.")

data = np.load('./DataRGB.npy')
print(data.shape)  # (25000, 49153)

data_input = data[:, :-1]
data_output = data[:, -1]
print("Shape of Input: {}, "
      "Shape of Output: {}".format(data_input.shape, data_output.shape))
  # Shape of Input: (25000, 49152), Shape of Output: (25000,)

from collections import Counter
print(Counter(data_output.tolist()))  # Counter({0: 12500, 1: 12500})

from sklearn.model_selection import train_test_split
TrainIn, TestIn, TrainOut, TestOut = train_test_split(data_input, data_output,
                                                    test_size = 0.3,
                                                    random_state = 100,
                                                    stratify = data_output
                                                    )
print("Shape of TrainInput: {}, "
      "Shape of TrainOutput: {}".format(TrainIn.shape, TrainOut.shape))
  # Shape of TrainInput: (17500, 49152), Shape of TrainOutput: (17500,)
print("Shape of TestInput: {}, "
      "Shape of TestOutput: {}".format(TestIn.shape, TestOut.shape))
  # Shape of TestInput: (7500, 49152), Shape of TestOutput: (7500,)
print("Dist. of Train: {}".format(Counter(TrainOut.tolist())))
  # Dist. of Train: Counter({0: 8750, 1: 8750})
print("Dist. of Test: {}".format(Counter(TestOut.tolist())))
  # Dist. of Test: Counter({0: 3750, 1: 3750})

if not os._exists('./splited_data'):
    os.mkdir('./splited_data')
np.save("./splited_data/TrainIn.npy", TrainIn)
np.save("./splited_data/TrainOut.npy", TrainOut)
np.save("./splited_data/TestIn.npy", TestIn)
np.save("./splited_data/TestOut.npy", TestOut)
print("Go Next Step.")
