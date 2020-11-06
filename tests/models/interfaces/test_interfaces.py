# Copyright (c) Facebook, Inc. and its affiliates.

import os
import unittest
from pathlib import Path

import numpy as np
import tests.test_utils as test_utils
from mmf.models.mmbt import MMBT
from mmf.utils.configuration import get_mmf_env


class TestModelInterfaces(unittest.TestCase):
    @test_utils.skip_if_no_network
    @test_utils.skip_if_windows
    @test_utils.skip_if_macos
    def test_mmbt_hm_interface(self):
        model = MMBT.from_pretrained("mmbt.hateful_memes.images")
        result = model.classify(
            "https://i.imgur.com/tEcsk5q.jpg", "look how many people love you"
        )

        self.assertEqual(result["label"], 0)
        np.testing.assert_almost_equal(result["confidence"], 0.9993, decimal=3)
        result = model.classify(
            "https://i.imgur.com/tEcsk5q.jpg", "they have the privilege"
        )
        self.assertEqual(result["label"], 0)
        np.testing.assert_almost_equal(result["confidence"], 0.9777, decimal=1)
        result = model.classify("https://i.imgur.com/tEcsk5q.jpg", "hitler and jews")
        self.assertEqual(result["label"], 1)
        np.testing.assert_almost_equal(result["confidence"], 0.8342, decimal=3)

    @test_utils.skip_if_no_network
    @test_utils.skip_if_windows
    @test_utils.skip_if_macos
    def test_mmbt_hm_interface_from_file(self):
        home = str(Path.home())
        data_dir = get_mmf_env(key="data_dir")
        model_file = os.path.join(home, data_dir, "models", "mmbt.hateful_memes.images")
        model = MMBT.from_pretrained("mmbt.hateful_memes.images", from_file=model_file)
        result = model.classify(
            "https://i.imgur.com/tEcsk5q.jpg", "look how many people love you"
        )

        self.assertEqual(result["label"], 0)
        np.testing.assert_almost_equal(result["confidence"], 0.9993, decimal=3)
        result = model.classify(
            "https://i.imgur.com/tEcsk5q.jpg", "they have the privilege"
        )
        self.assertEqual(result["label"], 0)
        np.testing.assert_almost_equal(result["confidence"], 0.9777, decimal=1)
        result = model.classify("https://i.imgur.com/tEcsk5q.jpg", "hitler and jews")
        self.assertEqual(result["label"], 1)
        np.testing.assert_almost_equal(result["confidence"], 0.8342, decimal=3)
