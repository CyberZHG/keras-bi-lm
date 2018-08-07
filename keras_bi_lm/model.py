import keras
import numpy as np


class BiLM(object):

    def __init__(self,
                 token_num=128,
                 model_path='',
                 embedding_dim=100,
                 embedding_weights=None,
                 embedding_trainable=None,
                 rnn_layer_num=1,
                 rnn_units=50,
                 rnn_keep_num=1,
                 rnn_type='lstm',
                 rnn_dropouts=0.1,
                 rnn_recurrent_dropouts=0.1,
                 use_bidirectional=False,
                 learning_rate=1e-3):
        """Initialize Bi-LM model.

        :param token_num: Number of words or characters.
        :param model_path: Path of saved model. All the other parameters will be ignored if path is not empty.
        :param embedding_dim: The dimension of embedding layer.
        :param embedding_weights: The initial weights of embedding layer.
        :param embedding_trainable: Whether the embedding layer is trainable.
        :param rnn_layer_num: The number of stacked bidirectional RNNs.
        :param rnn_units: An integer or a list representing the number of units of RNNs in one direction.
        :param rnn_keep_num: How many layers are used for predicting the probabilities of the next word.
        :param rnn_type: Type of RNN, 'gru' or 'lstm'.
        :param rnn_dropouts: A float or a list representing the dropout of the RNN unit.
        :param rnn_recurrent_dropouts: A float or a list representing the recurrent dropout of the RNN unit.
        :param use_bidirectional: Use bidirectional RNN in both forward and backward prediction.
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
        if embedding_trainable is None:
            embedding_trainable = embedding_weights is None
        embedding_layer = keras.layers.Embedding(input_dim=token_num,
                                                 output_dim=embedding_dim,
                                                 weights=embedding_weights,
                                                 trainable=embedding_trainable,
                                                 name='Bi-LM-Embedding')(input_layer)

        last_layer_forward, last_layer_backward = embedding_layer, embedding_layer
        rnn_layers_forward, rnn_layers_backward = [], []
        if rnn_type.lower() == 'gru':
            rnn = keras.layers.GRU
        else:
            rnn = keras.layers.LSTM
        for i in range(rnn_layer_num):
            if rnn_layer_num == 1:
                name = 'Bi-LM-Forward'
            else:
                name = 'Bi-LM-%s-Forward-%d' % (rnn_type.upper(), i + 1)
            if use_bidirectional:
                rnn_layer_forward = keras.layers.Bidirectional(
                    rnn(
                        units=rnn_units[i],
                        dropout=rnn_dropouts[i],
                        recurrent_dropout=rnn_recurrent_dropouts[i],
                        return_sequences=True,
                    ),
                    name=name,
                )(last_layer_forward)
            else:
                rnn_layer_forward = rnn(units=rnn_units[i],
                                        dropout=rnn_dropouts[i],
                                        recurrent_dropout=rnn_recurrent_dropouts[i],
                                        go_backwards=False,
                                        return_sequences=True,
                                        name=name)(last_layer_forward)
            last_layer_forward = rnn_layer_forward
            rnn_layers_forward.append(rnn_layer_forward)
            if rnn_layer_num == 1:
                name = 'Bi-LM-Backward'
            else:
                name = 'Bi-LM-%s-Backward-%d' % (rnn_type.upper(), i + 1)
            if use_bidirectional:
                rnn_layer_backward = keras.layers.Bidirectional(
                    rnn(
                        units=rnn_units[i],
                        dropout=rnn_dropouts[i],
                        recurrent_dropout=rnn_recurrent_dropouts[i],
                        go_backwards=True,
                        return_sequences=True,
                    ),
                    name=name,
                )(last_layer_backward)
            else:
                rnn_layer_backward = rnn(units=rnn_units[i],
                                         dropout=rnn_dropouts[i],
                                         recurrent_dropout=rnn_recurrent_dropouts[i],
                                         go_backwards=True,
                                         return_sequences=True,
                                         name=name)(last_layer_backward)
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
                                                 activation='softmax',
                                                 name='Bi-LM-Dense-Forward')(last_layer_forward)
        dense_layer_backward = keras.layers.Dense(units=token_num,
                                                  activation='softmax',
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
        outputs_forward = np.expand_dims(np.asarray(outputs_forward), axis=-1)
        outputs_backward = np.expand_dims(np.asarray(outputs_backward), axis=-1)
        return np.asarray(inputs), [outputs_forward, outputs_backward]

    def fit(self, inputs, outputs, epochs=1):
        """Simple wrapper of model.fit.

        :param inputs: Inputs.
        :param outputs: List of forward and backward outputs.
        :param epochs: Number of epoch.

        :return: None
        """
        self.model.fit(inputs, outputs, epochs=epochs)

    def predict(self, inputs):
        """Simple wrapper of model.predict.

        :param inputs: Inputs.

        :return: Predicted outputs.
        """
        return self.model.predict(inputs)

    def get_feature_layers(self, input_layer=None, trainable=False):
        """Get layers that output the Bi-LM feature.

        :param input_layer: Use existing input layer.
        :param trainable: Whether the layers are still trainable.

        :return [input_layer,] output_layer: Input and output layer.
        """
        model = keras.models.clone_model(self.model, input_layer)
        if not trainable:
            for layer in model.layers:
                layer.trainable = False
        forward_layer = model.get_layer(name='Bi-LM-Forward').output
        backward_layer = model.get_layer(name='Bi-LM-Backward').output
        output_layer = keras.layers.Concatenate(name='Bi-LM-Feature')([forward_layer, backward_layer])
        if input_layer is None:
            input_layer = model.layers[0].input
            return input_layer, output_layer
        return output_layer
