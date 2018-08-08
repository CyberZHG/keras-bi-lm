import numpy as np
import unittest
from keras_bi_lm import BiLM


class TestFitPredict(unittest.TestCase):

    def test_fit_predict_overfitting(self):
        sentences = [
            ['All', 'work', 'and', 'no', 'play'],
            ['makes', 'Jack', 'a', 'dull', 'boy', '.'],
        ]
        token_dict = {
            '': 0,
            '<UNK>': 1,
            '<EOS>': 2,
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
        token_dict_rev = {v: k for k, v in token_dict.items()}
        inputs, outputs = BiLM.get_batch(sentences,
                                         token_dict,
                                         ignore_case=True,
                                         unk_index=token_dict['<UNK>'],
                                         eos_index=token_dict['<EOS>'])
        bi_lm = BiLM(token_num=len(token_dict), embedding_dim=10, rnn_units=10)
        bi_lm.model.summary()
        bi_lm.fit(np.repeat(inputs, 2 ** 12, axis=0),
                  [
                      np.repeat(outputs[0], 2 ** 12, axis=0),
                      np.repeat(outputs[1], 2 ** 12, axis=0),
                  ],
                  epochs=5)
        predict = bi_lm.predict(inputs)
        forward = predict[0].argmax(axis=-1)
        backward = predict[1].argmax(axis=-1)
        self.assertEqual('work and no play <EOS>',
                         ' '.join(map(lambda x: token_dict_rev[x], forward[0].tolist()[:-1])).strip())
        self.assertEqual('<UNK> a dull boy . <EOS>',
                         ' '.join(map(lambda x: token_dict_rev[x], forward[1].tolist())).strip())
        self.assertEqual('<EOS> all work and no',
                         ' '.join(map(lambda x: token_dict_rev[x], backward[0].tolist()[:-1])).strip())
        self.assertEqual('<EOS> makes <UNK> a dull boy',
                         ' '.join(map(lambda x: token_dict_rev[x], backward[1].tolist())).strip())

    def test_bidirectional_overfitting(self):
        sentences = [
            ['All', 'work', 'and', 'no', 'play'],
            ['makes', 'Jack', 'a', 'dull', 'boy', '.'],
        ]
        token_dict = {
            '': 0,
            '<UNK>': 1,
            '<EOS>': 2,
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
        token_dict_rev = {v: k for k, v in token_dict.items()}
        inputs, outputs = BiLM.get_batch(sentences,
                                         token_dict,
                                         ignore_case=True,
                                         unk_index=token_dict['<UNK>'],
                                         eos_index=token_dict['<EOS>'])
        bi_lm = BiLM(token_num=len(token_dict), embedding_dim=10, rnn_units=10, use_bidirectional=True)
        bi_lm.model.summary()
        bi_lm.fit(np.repeat(inputs, 2 ** 12, axis=0),
                  [
                      np.repeat(outputs[0], 2 ** 12, axis=0),
                      np.repeat(outputs[1], 2 ** 12, axis=0),
                  ],
                  epochs=5)
        predict = bi_lm.predict(inputs)
        forward = predict[0].argmax(axis=-1)
        backward = predict[1].argmax(axis=-1)
        self.assertEqual('work and no play <EOS>',
                         ' '.join(map(lambda x: token_dict_rev[x], forward[0].tolist()[:-1])).strip())
        self.assertEqual('<UNK> a dull boy . <EOS>',
                         ' '.join(map(lambda x: token_dict_rev[x], forward[1].tolist())).strip())
        self.assertEqual('<EOS> all work and no',
                         ' '.join(map(lambda x: token_dict_rev[x], backward[0].tolist()[:-1])).strip())
        self.assertEqual('<EOS> makes <UNK> a dull boy',
                         ' '.join(map(lambda x: token_dict_rev[x], backward[1].tolist())).strip())
