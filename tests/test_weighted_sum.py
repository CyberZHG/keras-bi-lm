import os
import unittest
import keras
import numpy as np
from keras_bi_lm.weighted_sum import WeightedSum


class TestWeightedSum(unittest.TestCase):

    def setUp(self):
        self.tmp_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'tmp')
        if not os.path.exists(self.tmp_path):
            os.makedirs(self.tmp_path)

    def test_sum_layers(self):
        input_layer_1 = keras.layers.Input(shape=(5,), name='Input-1')
        input_layer_2 = keras.layers.Input(shape=(5,), name='Input-2')
        input_layer_3 = keras.layers.Input(shape=(5,), name='Input-3')
        weighted_layer = WeightedSum(name='WeightedSum')([input_layer_1, input_layer_2, input_layer_3])
        model = keras.models.Model(
            inputs=[input_layer_1, input_layer_2, input_layer_3],
            outputs=weighted_layer,
        )
        model.compile(
            optimizer='adam',
            loss='mse',
            metrics={},
        )
        model_path = os.path.join(self.tmp_path, 'save_load.h5')
        model.save(model_path)
        model = keras.models.load_model(
            model_path,
            custom_objects={'WeightedSum': WeightedSum},
        )
        inputs = [
            np.asarray([[1., 2., 3., 4., 5.]]),
            np.asarray([[1., 3., 5., 7., 9.]]),
            np.asarray([[1., 5., 9., 2., 8.]]),
        ]
        predict = model.predict(inputs)
        expect = np.asarray([[1., 10. / 3., 17 / 3., 13 / 3., 22. / 3.]])
        self.assertTrue(np.allclose(expect, predict), (expect, predict))
