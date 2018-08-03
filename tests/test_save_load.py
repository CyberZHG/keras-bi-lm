import os
import unittest
from keras_bi_lm import BiLM


class TestSaveLoad(unittest.TestCase):

    def setUp(self):
        self.tmp_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'tmp')
        if not os.path.exists(self.tmp_path):
            os.makedirs(self.tmp_path)

    def test_save_load(self):
        model_path = os.path.join(self.tmp_path, 'save_load.h5')
        model = BiLM(token_num=101)
        model.save_model(model_path)
        model.load_model(model_path)

    def test_init_load(self):
        model_path = os.path.join(self.tmp_path, 'save_load.h5')
        model = BiLM(token_num=101)
        model.save_model(model_path)
        BiLM(model_path=model_path)
