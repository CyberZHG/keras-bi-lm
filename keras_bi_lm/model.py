import keras
import numpy as np


class BiLM(object):

    def __init__(self,
                 token_num=128,
                 model_path='',
                 embedding_dim=100,
                 rnn_layer_num=1,
                 rnn_units=50,
                 rnn_keep_num=1,
                 rnn_type='lstm',
                 rnn_dropouts=0.1,
                 rnn_recurrent_dropouts=0.1,
                 learning_rate=1e-3):
        """Initialize Bi-LM model.

        :param token_num: Number of words or characters.
        :param model_path: Path of saved model. All the other parameters will be ignored if path is not empty.
        :param embedding_dim: The dimension of embedding layer.
        :param rnn_layer_num: The number of stacked bidirectional RNNs.
        :param rnn_units: An integer or a list representing the number of units of RNNs in one direction.
        :param rnn_keep_num: How many layers are used for predicting the probabilities of the next word.
        :param rnn_type: Type of RNN, 'gru' or 'lstm'.
        :param rnn_dropouts: A float or a list representing the dropout of the RNN unit.
        :param rnn_recurrent_dropouts: A float or a list representing the recurrent dropout of the RNN unit.
        :param learning_rate: Learning rate.

        :return model: The built model.
        """
        if model_path:
            self.load_model(model_path)
            return
        if not isinstance(rnn_units, list):
            rnn_units = [rnn_units] * rnn_layer_num
        if not isinstance(rnn_dropouts, list):
            rnn_dropouts = [rnn_dropouts] * rnn_layer_num
        if not isinstance(rnn_recurrent_dropouts, list):
            rnn_recurrent_dropouts = [rnn_recurrent_dropouts] * rnn_layer_num

        input_layer = keras.layers.Input(shape=(None,),
                                         name='Bi-LM-Input')
        embedding_layer = keras.layers.Embedding(input_dim=token_num,
                                                 output_dim=embedding_dim,
                                                 name='Bi-LM-Embedding')(input_layer)

        last_layer_forward, last_layer_backward = embedding_layer, embedding_layer
        rnn_layers_forward, rnn_layers_backward = [], []
        if rnn_type.lower() == 'gru':
            rnn = keras.layers.GRU
        else:
            rnn = keras.layers.LSTM
        for i in range(rnn_layer_num):
            rnn_layer_forward = rnn(units=rnn_units[i],
                                    dropout=rnn_dropouts[i],
                                    recurrent_dropout=rnn_recurrent_dropouts[i],
                                    go_backwards=False,
                                    return_sequences=True,
                                    name='Bi-LM-%s-Forward-%d' % (rnn_type.upper(), i + 1))(last_layer_forward)
            last_layer_forward = rnn_layer_forward
            rnn_layers_forward.append(rnn_layer_forward)
            rnn_layer_backward = rnn(units=rnn_units[i],
                                     dropout=rnn_dropouts[i],
                                     recurrent_dropout=rnn_recurrent_dropouts[i],
                                     go_backwards=True,
                                     return_sequences=True,
                                     name='Bi-LM-%s-Backward-%d' % (rnn_type.upper(), i + 1))(last_layer_backward)
            last_layer_backward = rnn_layer_backward
            rnn_layers_backward.append(rnn_layer_backward)

        if len(rnn_layers_forward) > rnn_keep_num:
            rnn_layers_forward = rnn_layers_forward[-rnn_keep_num:]
            rnn_layers_backward = rnn_layers_backward[-rnn_keep_num:]
        if len(rnn_layers_forward) == 1:
            last_layer_forward = rnn_layers_forward[0]
            last_layer_backward = rnn_layers_backward[0]
        else:
            last_layer_forward = keras.layers.Concatenate(name='Bi-LM-Forward')(rnn_layers_forward)
            last_layer_backward = keras.layers.Concatenate(name='Bi-LM-Backward')(rnn_layers_backward)

        dense_layer_forward = keras.layers.Dense(units=token_num,
                                                 name='Bi-LM-Dense-Forward')(last_layer_forward)
        dense_layer_backward = keras.layers.Dense(units=token_num,
                                                  name='Bi-LM-Dense-Backward')(last_layer_backward)

        model = keras.models.Model(inputs=input_layer, outputs=[dense_layer_forward, dense_layer_backward])
        model.compile(optimizer=keras.optimizers.Adam(lr=learning_rate),
                      loss=keras.losses.sparse_categorical_crossentropy,
                      metrics=[keras.metrics.sparse_categorical_accuracy])
        self.model = model

    def save_model(self, model_path):
        self.model.save(model_path)

    def load_model(self, model_path):
        self.model = keras.models.load_model(model_path)

    @staticmethod
    def get_batch(sentences,
                  token_dict,
                  ignore_case=False,
                  unk_index=1,
                  eos_index=2):
        """Get a batch of inputs and outputs from given sentences.

        :param sentences: A list of list of tokens.
        :param token_dict: The dict that maps a token to an integer. `<UNK>` and `<EOS>` should be preserved.
        :param ignore_case: Whether ignoring the case of the token.
        :param unk_index: The index for unknown token.
        :param eos_index: The index for ending of sentence.

        :return inputs, outputs: The inputs and outputs of the batch.
        """
        batch_size = len(sentences)
        max_sentence_len = max(map(len, sentences))
        inputs = [[0] * max_sentence_len for _ in range(batch_size)]
        outputs_forward = [[0] * max_sentence_len for _ in range(batch_size)]
        outputs_backward = [[0] * max_sentence_len for _ in range(batch_size)]
        for i, sentence in enumerate(sentences):
            outputs_forward[i][len(sentence) - 1] = eos_index
            outputs_backward[i][0] = eos_index
            for j, token in enumerate(sentence):
                if ignore_case:
                    index = token_dict.get(token.lower(), unk_index)
                else:
                    index = token_dict.get(token, unk_index)
                inputs[i][j] = index
                if j - 1 >= 0:
                    outputs_forward[i][j - 1] = index
                if j + 1 < len(sentence):
                    outputs_backward[i][j + 1] = index
        return np.asarray(inputs), [np.asarray(outputs_forward), np.asarray(outputs_backward)]
