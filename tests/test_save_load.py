import os
import unittest
import tempfile
from keras_bi_lm import BiLM


class TestSaveLoad(unittest.TestCase):

    def test_save_load(self):
        with tempfile.TemporaryDirectory() as temp_path:
            model_path = os.path.join(temp_path, 'save_load.h5')
            model = BiLM(token_num=101)
            model.save_model(model_path)
            model.load_model(model_path)

    def test_init_load(self):
        with tempfile.TemporaryDirectory() as temp_path:
            model_path = os.path.join(temp_path, 'save_load.h5')
            model = BiLM(token_num=101)
            model.save_model(model_path)
            BiLM(model_path=model_path)
