# Copyright 2019 The Google Research Authors.
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


"""Runnable example, as shown in the README.md."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import warnings

warnings.filterwarnings("ignore")

from nasbench import api


INPUT = "input"
OUTPUT = "output"
CONV1X1 = "conv1x1-bn-relu"
CONV3X3 = "conv3x3-bn-relu"
MAXPOOL3X3 = "maxpool3x3"

# NASBENCH_TFRECORD = "nasbench/nasbench_full.tfrecord"    # Too large to load in RAM


def get_nb101_api(NASBENCH_TFRECORD="nasbench/nasbench_only108.tfrecord"):
    warnings.filterwarnings("ignore")
    # Load the data from file (this will take some time)
    nasbench = api.NASBench(NASBENCH_TFRECORD)

    # acc = []
    # for unique_hash in nasbench.hash_iterator():
    #     fixed_metrics, computed_metrics = nasbench.get_metrics_from_hash(unique_hash)
    #     acc.append(computed_metrics[108][0]["final_test_accuracy"])
    # # 打印acc列表的最小值，1/4分位数，1/2分位数，3/4分位数，最大值
    # print("acc min:", min(acc))
    # print("acc 1/4:", sorted(acc)[int(len(acc) * 0.25)])
    # print("acc 1/2:", sorted(acc)[int(len(acc) * 0.5)])
    # print("acc 3/4:", sorted(acc)[int(len(acc) * 0.75)])
    # print("acc max:", max(acc))

    return nasbench

