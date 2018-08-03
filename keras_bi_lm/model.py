import keras


def get_model(token_num,
              embedding_dim=100,
              rnn_layer_num=1,
              rnn_units=50,
              rnn_keep_num=1,
              rnn_type='lstm',
              rnn_dropouts=0.1,
              rnn_recurrent_dropouts=0.1):
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
    model.compile(optimizer=keras.optimizers.Adam(),
                  loss=keras.losses.sparse_categorical_crossentropy,
                  metrics=[keras.metrics.sparse_categorical_accuracy])
    return model
