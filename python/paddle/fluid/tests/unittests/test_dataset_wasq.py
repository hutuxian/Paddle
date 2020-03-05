#   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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
"""
TestCases for Dataset,
including create, config, run, etc.
"""

from __future__ import print_function
import paddle.fluid as fluid
import paddle.compat as cpt
import paddle.fluid.core as core
import numpy as np
import os
import shutil
import unittest


class TestDataset(unittest.TestCase):
    """  TestCases for Dataset. """

    def setUp(self):
        self.use_data_loader = False
        self.epoch_num = 10
        self.drop_last = False

    def test_dataset_create(self):
        """ Testcase for dataset create. """
        try:
            dataset = fluid.DatasetFactory().create_dataset("InMemoryDataset")
        except:
            self.assertTrue(False)

    def test_config(self):
        """
        Testcase for python config.
        """
        dataset = fluid.InMemoryDataset()
        dataset.set_parse_ins_id(False)
        dataset.set_parse_content(False)
        dataset.set_parse_logkey(True)
        #self.assertTrue(dataset.parse_ins_id)
        self.assertTrue(dataset.parse_logkey)

    def test_run_with_dump(self):
        """
        Testcase for InMemoryDataset from create to run.
        """
        with open("test_run_with_dump_a.txt", "w") as f:
            data = "1 1702f830e2e19501ad7429505f714c1d 1 1 2 3 3 4 5 5 5 5 1 1\n"
            data += "1 1702f81d1de0de019118711254c8fba2 1 2 2 3 4 4 6 6 6 6 1 2\n"
            data += "1 1702f81d1de195019118711254c8fba2 1 3 2 3 5 4 7 7 7 7 1 3\n"
            f.write(data)
        with open("test_run_with_dump_b.txt", "w") as f:
            data = "1 1702f82d32734e018e854f464db5dbc3 1 4 2 3 3 4 5 5 5 5 1 4\n"
            data += "1 1702f830be90e50194971f4b4b638c2c 1 5 2 3 4 4 6 6 6 6 1 5\n"
            data += "1 1702f81ed061950199dd44f272ebf4a0 1 6 2 3 5 4 7 7 7 7 1 6\n"
            data += "1 1702f83033c19501651ab7f1d4fd009d 1 7 2 3 6 4 8 8 8 8 1 7\n"
            f.write(data)

        slots = ["slot1", "slot2", "slot3", "slot4"]
        slots_vars = []
        for slot in slots:
            var = fluid.layers.data(
                name=slot, shape=[1], dtype="int64", lod_level=1)
            slots_vars.append(var)

        dataset = fluid.DatasetFactory().create_dataset("InMemoryDataset")
        dataset.set_feed_type("TwoPhaseDataFeed")
        dataset.set_batch_size(32)
        dataset.set_thread(200)
        dataset.set_filelist(
            ["test_run_with_dump_a.txt", "test_run_with_dump_b.txt"])
        #dataset.set_parse_ins_id(True)
        #dataset.set_parse_content(True)
        dataset.set_parse_logkey(True)
        dataset.set_pipe_command("cat")
        dataset.set_use_var(slots_vars)
        dataset.load_into_memory()
        dataset.set_fea_eval(10000, True)
        # dataset.local_shuffle()
        dataset.merge_pv_instance()

        # exe = fluid.Executor(fluid.CPUPlace())
        exe = fluid.Executor(fluid.CUDAPlace(0))
        exe.run(fluid.default_startup_program())
        exe.train_from_dataset(fluid.default_main_program(), dataset, thread=3)
        dataset.divide_pv_instance()
        box = core.BoxWrapper()
        box.flip_pass_flag()
        exe.train_from_dataset(fluid.default_main_program(), dataset, thread=3)


if __name__ == '__main__':
    unittest.main()
