import os
import codecs
import random
import numpy
import keras
from keras_wc_embd import get_dicts_generator, get_word_list_eng
from keras_bi_lm import BiLM

DEBUG = False

DATA_ROOT = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'dataset', 'aclImdb')
TRAIN_ROOT = os.path.join(DATA_ROOT, 'train')
TEST_ROOT = os.path.join(DATA_ROOT, 'test')

LM_MODEL_PATH = 'lm.h5'

# Get data for training
print('Get train data entries...')
train_pos_files = os.listdir(os.path.join(TRAIN_ROOT, 'pos'))
train_neg_files = os.listdir(os.path.join(TRAIN_ROOT, 'neg'))
train_num = int(len(train_pos_files) * 0.8)
val_num = len(train_neg_files) - train_num

random.shuffle(train_pos_files)
random.shuffle(train_neg_files)
val_pos_files = train_pos_files[train_num:]
val_neg_files = train_neg_files[train_num:]
train_pos_files = train_pos_files[:train_num]
train_neg_files = train_neg_files[:train_num]

epoch_num = 10
batch_size = 16
train_steps = train_num * 2 // batch_size
val_steps = val_num * 2 // batch_size

if DEBUG:
    epoch_num = 1
    train_num //= 10
    val_num //= 10
    train_steps //= 10
    val_steps //= 10
    train_pos_files = train_pos_files[:len(train_pos_files) // 10]
    train_neg_files = train_neg_files[:len(train_neg_files) // 10]
    val_pos_files = val_pos_files[:len(val_pos_files) // 10]
    val_neg_files = val_neg_files[:len(val_neg_files) // 10]

print('Train: %d  Validate: %d' % (train_num, val_num))

# Generate dictionaries for words and characters
print('Get dictionary`....')
dicts_generator = get_dicts_generator(
    word_min_freq=5,
    char_min_freq=2,
    word_ignore_case=True,
    char_ignore_case=False,
)
sentences = []
for file_name in train_pos_files:
    with codecs.open(os.path.join(TRAIN_ROOT, 'pos', file_name), 'r', 'utf8') as reader:
        text = reader.read().strip()
        sentence = get_word_list_eng(text)
        sentences.append(sentence)
        dicts_generator(sentence=sentence)
for file_name in train_neg_files:
    with codecs.open(os.path.join(TRAIN_ROOT, 'neg', file_name), 'r', 'utf8') as reader:
        text = reader.read().strip()
        sentence = get_word_list_eng(text)
        sentences.append(sentence)
        dicts_generator(sentence=sentence)
word_dict, _, _ = dicts_generator(return_dict=True)
print('Word dict size: %d' % len(word_dict))


# Training LM
def train_lm_generator(batch_size=32):
    while True:
        index = 0
        while index * batch_size < len(sentences):
            batch_sentences = sentences[index * batch_size:(index + 1) * batch_size]
            inputs, outputs = BiLM.get_batch(batch_sentences,
                                             token_dict=word_dict,
                                             ignore_case=True)
            yield inputs, outputs


print('Fit LM...')
if os.path.exists(LM_MODEL_PATH):
    bi_lm = BiLM(model_path=LM_MODEL_PATH)
else:
    bi_lm = BiLM(token_num=len(word_dict))
    bi_lm.model.fit_generator(
        generator=train_lm_generator(batch_size=batch_size),
        steps_per_epoch=len(sentences) // batch_size,
        epochs=epoch_num,
        verbose=True,
    )
    bi_lm.save_model(LM_MODEL_PATH)

# Build model for classification
input_layer, feature_layer = bi_lm.get_feature_layers()
lstm_layer = keras.layers.Bidirectional(
    keras.layers.LSTM(units=50),
    name='Bi-LSTM',
)(feature_layer)
dense_layer = keras.layers.Dense(
    units=2,
    activation='softmax',
    name='Dense',
)(lstm_layer)
model = keras.models.Model(inputs=input_layer, outputs=dense_layer)
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['categorical_accuracy'],
)
model.summary()


# Train model
def train_batch_generator(batch_size=32, training=True):
    batch_size //= 2
    while True:
        sentences = []
        if training:
            batch_pos = random.sample(train_pos_files, batch_size)
            batch_neg = random.sample(train_neg_files, batch_size)
        else:
            batch_pos = random.sample(val_pos_files, batch_size)
            batch_neg = random.sample(val_neg_files, batch_size)
        for file_name in batch_pos:
            with codecs.open(os.path.join(TRAIN_ROOT, 'pos', file_name), 'r', 'utf8') as reader:
                text = reader.read().strip()
                sentences.append(get_word_list_eng(text))
        for file_name in batch_neg:
            with codecs.open(os.path.join(TRAIN_ROOT, 'neg', file_name), 'r', 'utf8') as reader:
                text = reader.read().strip()
            sentences.append(get_word_list_eng(text))
        word_input, _ = BiLM.get_batch(
            sentences=sentences,
            token_dict=word_dict,
            ignore_case=True,
        )
        yield word_input, keras.utils.to_categorical([1] * batch_size + [0] * batch_size)


model.fit_generator(
    generator=train_batch_generator(batch_size=batch_size, training=True),
    steps_per_epoch=train_steps,
    epochs=epoch_num,
    validation_data=train_batch_generator(batch_size=batch_size, training=False),
    validation_steps=val_steps,
    verbose=True,
)

# Check performance on test set
test_pos_files = os.listdir(os.path.join(TEST_ROOT, 'pos'))
test_neg_files = os.listdir(os.path.join(TEST_ROOT, 'neg'))
test_num = len(test_pos_files)
if DEBUG:
    test_num //= 10
test_steps = test_num * 2 // batch_size


def test_batch_generator(batch_size=32):
    batch_size //= 2
    index = 0
    while index < test_num:
        sentences = []
        batch_pos = test_pos_files[index: min(index + batch_size, test_num)]
        batch_neg = test_neg_files[index: min(index + batch_size, test_num)]
        index += batch_size
        for file_name in batch_pos:
            with codecs.open(os.path.join(TEST_ROOT, 'pos', file_name), 'r', 'utf8') as reader:
                text = reader.read().strip()
                sentences.append(get_word_list_eng(text))
        for file_name in batch_neg:
            with codecs.open(os.path.join(TEST_ROOT, 'neg', file_name), 'r', 'utf8') as reader:
                text = reader.read().strip()
            sentences.append(get_word_list_eng(text))
        word_input, _ = BiLM.get_batch(
            sentences=sentences,
            token_dict=word_dict,
            ignore_case=True,
        )
        yield word_input


predicts = model.predict_generator(
    generator=test_batch_generator(batch_size),
    steps=test_steps,
    verbose=True,
)
predicts = numpy.argmax(predicts, axis=-1).tolist()
correct = 0
for i in range(len(predicts)):
    if i % batch_size < batch_size:
        expect = 1
    else:
        expect = 0
    if predicts[i] == expect:
        correct += 1
print('Accuracy: %.3f' % (1.0 * correct / len(predicts)))
