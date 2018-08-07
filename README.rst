
Keras Bi-LM
===========


.. image:: https://travis-ci.org/PoWWoP/keras-bi-lm.svg
   :target: https://travis-ci.org/PoWWoP/keras-bi-lm
   :alt: Travis


.. image:: https://coveralls.io/repos/github/PoWWoP/keras-bi-lm/badge.svg?branch=master
   :target: https://coveralls.io/github/PoWWoP/keras-bi-lm
   :alt: Coverage


Introduction
------------

The repository contains a class for training a bidirectional language model which could be used as a feature extraction method for each position in a sentence.

Install
-------

.. code-block:: bash

   pip install keras-bi-lm

Usage
-----

Train and save the Bi-LM model
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Before using it as a feature extraction method, the language model must be trained on large corpora.

.. code-block:: python

   from keras-bi-lm import BiLM

   sentences = [
       ['All', 'work', 'and', 'no', 'play'],
       ['makes', 'Jack', 'a', 'dull', 'boy', '.'],
   ]
   token_dict = {
       '': 0, '<UNK>': 1, '<EOS>': 2,
       'all': 3, 'work': 4, 'and': 5, 'no': 6, 'play': 7,
       'makes': 8, 'a': 9, 'dull': 10, 'boy': 11, '.': 12,
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
   bi_lm.save_model('bi_lm.h5')

``BiLM()``
~~~~~~~~~~~~~~

The core class that contains the model to be trained and used. Key parameters:


* ``token_num``\ : Number of words or characters.
* ``embedding_dim``\ : The dimension of embedding layer.
* ``rnn_layer_num``\ : The number of stacked bidirectional RNNs.
* ``rnn_units``\ : An integer or a list representing the number of units of RNNs in one direction.
* ``rnn_keep_num``\ : How many layers are used for predicting the probabilities of the next word.
* ``rnn_type``\ : Type of RNN, 'gru' or 'lstm'.

``BiLM.get_batch()``
~~~~~~~~~~~~~~~~~~~~~~~~

A helper function that converts sentences to batch inputs and outputs for training the model.


* ``sentences``\ : A list of list of tokens.
* ``token_dict``\ : The dict that maps a token to an integer. ``<UNK>`` and ``<EOS>`` should be preserved.
* ``ignore_case``\ : Whether ignoring the case of the token.
* ``unk_index``\ : The index for unknown token.
* ``eos_index``\ : The index for ending of sentence.

Load and use the Bi-LM model
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from keras-bi-lm import BiLM

   bi_lm = BiLM(model_path='bi_lm.h5')  # or `bi_lm.load_model('bi_lm.h5')`
   input_layer, output_layer = bi_lm.get_feature_layers()
   model = keras.models.Model(inputs=input_layer, outputs=output_layer)
   model.summary()

The ``output_layer`` is the time-distributed feature and all the parameters in the layers of the model are not trainable.

Demo
----

See ``demo`` directory:

.. code-block:: bash

   cd demo
   ./get_data.sh
   pip install -r requirements.txt
   python setiment_analysis.py

Citation
--------

Just cite the paper you've seen.
