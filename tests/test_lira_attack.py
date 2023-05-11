"""test_lira_attack.py
Copyright (C) Jim Smith2022  <james.smith@uwe.ac.uk>
"""
# pylint: disable = duplicate-code

import os
import sys

import logging

# import json
from unittest.mock import patch
from unittest import TestCase

import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from aisdc.attacks import likelihood_attack
from aisdc.attacks.dataset import Data  # pylint: disable = import-error
from aisdc.attacks.likelihood_attack import (  # pylint: disable = import-error
    LIRAAttack,
    LIRAAttackArgs,
)

N_SHADOW_MODELS = 20

logger = logging.getLogger(__file__)

def clean_up(name):
    """removes unwanted files or directory"""
    if os.path.exists(name) and os.path.isfile(name):
        os.remove(name)
        logger.info("Removed %s", name)


class TestLiraAttack(TestCase):

    @classmethod
    def setUpClass(cls):
        '''Setup the common things for the class'''
        logger.info("Setting up test class")
        X, y = load_breast_cancer(return_X_y=True, as_frame=False)
        cls.train_X, cls.test_X, cls.train_y, cls.test_y = train_test_split(X, y, test_size=0.3)
        cls.dataset = Data()
        cls.dataset.add_processed_data(cls.train_X, cls.train_y, cls.test_X, cls.test_y)
        cls.target_model = RandomForestClassifier(
            n_estimators=100, min_samples_split=2, min_samples_leaf=1
        )
        cls.target_model.fit(cls.train_X, cls.train_y)


    def test_lira_attack(self):
        """tests the lira code two ways"""
        args = LIRAAttackArgs(n_shadow_models=N_SHADOW_MODELS, report_name="lira_example_report")
        attack_obj = LIRAAttack(args)
        attack_obj.setup_example_data()
        attack_obj.attack_from_config()
        attack_obj.example()
        
        args2 = LIRAAttackArgs(n_shadow_models=N_SHADOW_MODELS, report_name="lira_example2_report")
        attack_obj2 = LIRAAttack(args2)
        attack_obj2.attack(self.dataset, self.target_model)
        output2 = attack_obj2.make_report()  # also makes .pdf and .json files
        _ = output2["attack_metrics"][0]


    def test_check_and_update_dataset(self):
        """test the code that removes items from test set with classes
        not present in training set"""
        args = LIRAAttackArgs(n_shadow_models=N_SHADOW_MODELS, report_name="lira_example_report")
        attack_obj = LIRAAttack(args)
        attack_obj.setup_example_data()
        attack_obj.attack_from_config()
        attack_obj.example()

        # now make test[0] have a  class not present in training set#
        local_test_y = np.copy(self.test_y)
        local_test_y[0] = 5
        local_dataset = Data()
        local_dataset.add_processed_data(
            self.train_X,
            self.train_y,
            self.test_X,
            local_test_y
        )
        attack_obj._check_and_update_dataset(  # pylint:disable=protected-access
            local_dataset, self.target_model
        )


    def test_main(self):
        """test invocation via command line"""

        # option 1
        testargs = ["prog", "run-example"]
        with patch.object(sys, "argv", testargs):
            likelihood_attack.main()

        # option 2
        testargs = ["prog", "run-attack", "--j", "tests/lrconfig.json"]
        with patch.object(sys, "argv", testargs):
            likelihood_attack.main()

        # option 3 "setup-example-data"
        testargs = ["prog", "setup-example-data"]  # , "--j", "tests/lrconfig.json"]
        with patch.object(sys, "argv", testargs):
            likelihood_attack.main()
    
    @classmethod
    def tearDownClass(cls):
        """cleans up various files made during the tests"""
        names = [
            "lr_report.pdf",
            "log_roc.png",
            "lr_report.json",
            "lira_example2_report.json",
            "lira_example2_report.pdf",
            "test_preds.csv",
            "config.json",
            "train_preds.csv",
            "test_data.csv",
            "train_data.csv",
        ]
        for name in names:
            clean_up(name)
