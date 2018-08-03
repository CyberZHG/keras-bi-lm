import keras


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
                                         name='Input')
        embedding_layer = keras.layers.Embedding(input_dim=token_num,
                                                 output_dim=embedding_dim,
                                                 name='Embedding')(input_layer)

        last_layer, rnn_layers = embedding_layer, []
        if rnn_type.lower() == 'gru':
            rnn = keras.layers.GRU
        else:
            rnn = keras.layers.LSTM
        for i in range(rnn_layer_num):
            rnn_layer = keras.layers.Bidirectional(rnn(units=rnn_units[i],
                                                       dropout=rnn_dropouts[i],
                                                       recurrent_dropout=rnn_recurrent_dropouts[i],
                                                       return_sequences=True),
                                                   name='%s-%d' % (rnn_type.upper(), i + 1))(last_layer)
            last_layer = rnn_layer
            rnn_layers.append(rnn_layer)

        if len(rnn_layers) > rnn_keep_num:
            rnn_layers = rnn_layers[-rnn_keep_num:]
        if len(rnn_layers) == 1:
            last_layer = rnn_layers[0]
        else:
            last_layer = keras.layers.Concatenate(name='Concatenation')(rnn_layers)

        dense_layer = keras.layers.Dense(units=token_num,
                                         name='Dense')(last_layer)

        model = keras.models.Model(inputs=input_layer, outputs=dense_layer)
        model.compile(optimizer=keras.optimizers.Adam(lr=learning_rate),
                      loss=keras.losses.sparse_categorical_crossentropy,
                      metrics=[keras.metrics.sparse_categorical_accuracy])
        self.model = model

    def save_model(self, model_path):
        self.model.save(model_path)

    def load_model(self, model_path):
        self.model = keras.models.load_model(model_path)
