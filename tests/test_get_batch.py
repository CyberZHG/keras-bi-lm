import unittest
from keras_bi_lm import BiLM


class TestGetBatch(unittest.TestCase):

    def test_get_batch(self):
        sentences = [
            ['All', 'work', 'and', 'no', 'play'],
            ['makes', 'Jack', 'a', 'dull', 'boy', '.'],
        ]
        token_dict = {
            'all': 3,
            'work': 4,
            'and': 5,
            'no': 6,
            'play': 7,
            'makes': 8,
            'a': 9,
            'dull': 10,
            'boy': 11,
            '.': 12,
        }
        inputs, outputs = BiLM.get_batch(sentences, token_dict, ignore_case=False)
        expect = [
            [1, 4, 5, 6, 7, 0],
            [8, 1, 9, 10, 11, 12],
        ]
        self.assertEqual(expect, inputs.tolist())
        expect = [
            [4, 5, 6, 7, 2, 0],
            [1, 9, 10, 11, 12, 2],
        ]
        self.assertEqual(expect, outputs[0].tolist())
        expect = [
            [2, 1, 4, 5, 6, 0],
            [2, 8, 1, 9, 10, 11],
        ]
        self.assertEqual(expect, outputs[1].tolist())
        inputs, outputs = BiLM.get_batch(sentences, token_dict, ignore_case=True)
        expect = [
            [3, 4, 5, 6, 7, 0],
            [8, 1, 9, 10, 11, 12],
        ]
        self.assertEqual(expect, inputs.tolist())
        expect = [
            [4, 5, 6, 7, 2, 0],
            [1, 9, 10, 11, 12, 2],
        ]
        self.assertEqual(expect, outputs[0].tolist())
        expect = [
            [2, 3, 4, 5, 6, 0],
            [2, 8, 1, 9, 10, 11],
        ]
        self.assertEqual(expect, outputs[1].tolist())
