
import tensorflow as tf
import numpy as np
import pickle

import models_trainable

from TokenizerWrap import TokenizerWrap

from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import Input, Dense, GRU, Embedding
from tensorflow.python.keras.optimizers import RMSprop
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard

from process_text import TextFilter

k_path_checkpoint = './check_p/k_21_checkpoint.keras'
p_path_checkpoint = './check_p/p_21_checkpoint.keras'
a_path_checkpoint = './check_p/a_21_checkpoint.keras'

queries = ["테러", "사고", "건강", "일본", "북미",
           "한미", "정상회담", "선거", "김기식", "외교", "국방", "국회",
           "청화대", "비핵화", "자유한국당", "더불어민주당", "개헌", "문재인", "대통령",
           "이명박", "암호화폐", "핵무기",
           "날씨", "중국", "미국", "북한",
           "FTA", "경제", "부동산",
           "미투", "박근혜"]

queries_unknown = ["쓰레기", "보이스피싱", "야구", "농구",
                   "스포츠", "개임", "자율주행", "사고",
                   "UAE", "졸음운전", "몰래카메라", "골프", "스마트폰", "전자발찌", "커피"
                                                                 "술", "마약", "폭력"]
political = ["일본", "북미", "한미", "정상회담", "선거", "김기식", "국방", "국회",
             "청화대", "비핵화", "자유한국당", "더불어민주당", "개헌", "문재인", "대통령",
             "이명박", "핵무기", "중국", "미국", "북한", "FTA", "경제", "부동산", "박근혜"]

non_political = []
mark_start = 'ssss '
mark_end = ' eeee'

num_words = 150000


def sparse_cross_entropy(y_true, y_pred):
    """
    Calculate the cross-entropy loss between y_true and y_pred.

    y_true is a 2-rank tensor with the desired output.
    The shape is [batch_size, sequence_length] and it
    contains sequences of integer-tokens.

    y_pred is the decoder's output which is a 3-rank tensor
    with shape [batch_size, sequence_length, num_words]
    so that for each sequence in the batch there is a one-hot
    encoded array of length num_words.
    """

    # Calculate the loss. This outputs a
    # 2-rank tensor of shape [batch_size, sequence_length]
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y_true,
                                                          logits=y_pred)

    # Keras may reduce this across the first axis (the batch)
    # but the semantics are unclear, so to be sure we use
    # the loss across the entire 2-rank tensor, we reduce it
    # to a single scalar with the mean function.
    loss_mean = tf.reduce_mean(loss)

    return loss_mean

def translate(model_encoder,
              model_decoder,
              tokenizer_src,
              tokenizer_dest,
              input_text,
              true_output_text=None):
    """Translate a single text-string."""


    token_start = tokenizer_dest.word_index[mark_start.strip()]
    token_end = tokenizer_dest.word_index[mark_end.strip()]

    # Convert the input-text to integer-tokens.
    # Note the sequence of tokens has to be reversed.
    # Padding is probably not necessary.
    input_tokens = tokenizer_src.text_to_tokens(text=input_text,
                                                reverse=True,
                                                padding=True)

    # Get the output of the encoder's GRU which will be
    # used as the initial state in the decoder's GRU.
    # This could also have been the encoder's final state
    # but that is really only necessary if the encoder
    # and decoder use the LSTM instead of GRU because
    # the LSTM has two internal states.
    initial_state = model_encoder.predict(input_tokens)

    vector = initial_state

    # Max number of tokens / words in the output sequence.
    max_tokens = tokenizer_dest.max_tokens

    # Pre-allocate the 2-dim array used as input to the decoder.
    # This holds just a single sequence of integer-tokens,
    # but the decoder-model expects a batch of sequences.
    shape = (1, max_tokens)
    decoder_input_data = np.zeros(shape=shape, dtype=np.int)

    # The first input-token is the special start-token for 'ssss '.
    token_int = token_start

    # Initialize an empty output-text.
    output_text = ''

    # Initialize the number of tokens we have processed.
    count_tokens = 0


    # While we haven't sampled the special end-token for ' eeee'
    # and we haven't processed the max number of tokens.
    while token_int != token_end and count_tokens < max_tokens:
        # Update the input-sequence to the decoder
        # with the last token that was sampled.
        # In the first iteration this will set the
        # first element to the start-token.
        decoder_input_data[0, count_tokens] = token_int

        # Wrap the input-data in a dict for clarity and safety,
        # so we are sure we input the data in the right order.
        x_data = \
            {
                'decoder_initial_state': initial_state,
                'decoder_input': decoder_input_data
            }

        # Note that we input the entire sequence of tokens
        # to the decoder. This wastes a lot of computation
        # because we are only interested in the last input
        # and output. We could modify the code to return
        # the GRU-states when calling predict() and then
        # feeding these GRU-states as well the next time
        # we call predict(), but it would make the code
        # much more complicated.

        # Input this data to the decoder and get the predicted output.
        decoder_output = model_decoder.predict(x_data)

        # Get the last predicted token as a one-hot encoded array.
        token_onehot = decoder_output[0, count_tokens, :]

        # Convert to an integer-token.
        token_int = np.argmax(token_onehot)

        # Lookup the word corresponding to this integer-token.
        sampled_word = tokenizer_dest.token_to_word(token_int)

        # Append the word to the output-text.
        output_text += " " + sampled_word

        # Increment the token-counter.
        count_tokens += 1

    # Sequence of tokens output by the decoder.
    output_tokens = decoder_input_data[0]

    # Print the input-text.
    print("Input text:", input_text)

    # Print the translated output-text.
    print("Translated text:",  output_text)

    # Optionally print the true translated text.
    if true_output_text is not None:
        print("True output text:", true_output_text)

    print()
    return output_text.replace(mark_end, ""), vector

def get_model():

    #
    encoder_input = Input(shape=(None, ), name='encoder_input')
    embedding_size = 128

    encoder_embedding = Embedding(input_dim=num_words,
                                  output_dim=embedding_size,
                                  name='encoder_embedding')

    state_size = 512

    encoder_gru1 = GRU(state_size, name='encoder_gru1',
                       return_sequences=True)
    encoder_gru2 = GRU(state_size, name='encoder_gru2',
                       return_sequences=True)
    encoder_gru3 = GRU(state_size, name='encoder_gru3',
                       return_sequences=False)


    def connect_encoder():
        # Start the neural network with its input-layer.
        net = encoder_input

        # Connect the embedding-layer.
        net = encoder_embedding(net)

        # Connect all the GRU-layers.
        net = encoder_gru1(net)
        net = encoder_gru2(net)
        net = encoder_gru3(net)

        # This is the output of the encoder.
        encoder_output = net

        return encoder_output

    encoder_output = connect_encoder()

    k_decoder_initial_state, k_decoder_input, k_decoder_output, k_connect_decoder \
        = get_decoder(encoder_output, len(queries)+3)

    k_model_train = Model(inputs=[encoder_input, k_decoder_input],
                        outputs=[k_decoder_output])

    k_decoder_output = k_connect_decoder(initial_state=k_decoder_initial_state)

    k_model_decoder = Model(inputs=[k_decoder_input, k_decoder_initial_state],
                          outputs=[k_decoder_output])


    p_decoder_initial_state, p_decoder_input, p_decoder_output, p_connect_decoder \
        = get_decoder(encoder_output, 2 + 3)

    p_model_train = Model(inputs=[encoder_input, p_decoder_input],
                          outputs=[p_decoder_output])

    p_decoder_output = p_connect_decoder(initial_state=p_decoder_initial_state)

    p_model_decoder = Model(inputs=[p_decoder_input, p_decoder_initial_state],
                          outputs=[p_decoder_output])

    a_decoder_initial_state, a_decoder_input, a_decoder_output, a_connect_decoder \
        = get_decoder(encoder_output, num_words)

    a_model_train = Model(inputs=[encoder_input, a_decoder_input],
                          outputs=[a_decoder_output])

    a_decoder_output = a_connect_decoder(initial_state=a_decoder_initial_state)

    a_model_decoder = Model(inputs=[a_decoder_input, a_decoder_initial_state],
                          outputs=[a_decoder_output])


    model_encoder = Model(inputs=[encoder_input],
                          outputs=[encoder_output])


    optimizer = RMSprop(lr=1e-3)

    decoder_target = tf.placeholder(dtype='int32', shape=(None, None))

    k_model_train.compile(optimizer=optimizer,
                        loss=sparse_cross_entropy,
                        target_tensors=[decoder_target])

    p_model_train.compile(optimizer=optimizer,
                        loss=sparse_cross_entropy,
                        target_tensors=[decoder_target])

    a_model_train.compile(optimizer=optimizer,
                        loss=sparse_cross_entropy,
                        target_tensors=[decoder_target])

    return k_model_train, p_model_train, a_model_train, model_encoder, k_model_decoder, p_model_decoder, a_model_decoder


def get_decoder(encoder_output, dim, state_size = 512, embedding_size = 128):

    decoder_initial_state = Input(shape=(state_size,),
                                  name='decoder_initial_state')

    decoder_input = Input(shape=(None, ), name='decoder_input')

    decoder_embedding = Embedding(input_dim=dim,
                                  output_dim=embedding_size,
                                  name='decoder_embedding')

    decoder_gru1 = GRU(state_size, name='decoder_gru1',
                       return_sequences=True)
    decoder_gru2 = GRU(state_size, name='decoder_gru2',
                       return_sequences=True)
    decoder_gru3 = GRU(state_size, name='decoder_gru3',
                       return_sequences=True)

    decoder_dense = Dense(dim,
                          activation='linear',
                          name='decoder_output')

    def connect_decoder(initial_state):
        # Start the decoder-network with its input-layer.
        net = decoder_input

        # Connect the embedding-layer.
        net = decoder_embedding(net)

        # Connect all the GRU-layers.
        net = decoder_gru1(net, initial_state=initial_state)
        net = decoder_gru2(net, initial_state=initial_state)
        net = decoder_gru3(net, initial_state=initial_state)

        # Connect the final dense layer that converts to
        # one-hot encoded arrays.
        decoder_output = decoder_dense(net)

        return decoder_output

    decoder_output = connect_decoder(initial_state=encoder_output)

    return decoder_initial_state, decoder_input, decoder_output, connect_decoder


def train_model(reload=False):

    models_trainable.initialized()

    data_src = []

    k_data_array = []
    k_data_dest = []

    p_data_array = []
    p_data_dest = []

    a_data_array = []
    a_data_dest = []

    if reload:

        text_filter = TextFilter()

        keyword_models = models_trainable.Keyword.select().where(
            models_trainable.Keyword.t_type >= 1,
            models_trainable.Keyword.t_type <= 2
        )

        for keyword in keyword_models:
            try:

                videos = models_trainable.Video.select() \
                    .join(models_trainable.Relationship, on=models_trainable.Relationship.to_video) \
                    .where(models_trainable.Relationship.from_keyword == keyword)
                print(keyword.name, len(videos))
                for video in videos:
                    title = video.title
                    text_filter.set_text(title)

                    text_filter.regex_from_text(r'\[[^)]*\]')
                    text_filter.remove_texts_from_text()
                    text_filter.remove_pumsas_from_list()
                    text_filter.remove_texts_from_text()

                    k_data_array.append([mark_start + keyword.name + mark_end,
                                       str(text_filter)])
                    p_data_array.append([mark_start + str(keyword.t_type) + mark_end,
                                       str(text_filter)])

            except models_trainable.DoesNotExist:
                print("does not exist")


        videos = models_trainable.Video.select()
        for video in videos:
            title = video.title

            text_filter.set_text(title)

            text_filter.regex_from_text(r'\[[^)]*\]')
            text_filter.remove_texts_from_text()
            text_filter.remove_pumsas_from_list()
            text_filter.remove_texts_from_text()
            a_data_array.append([mark_start + str(text_filter) + mark_end,
                               str(text_filter)])

        k_counter = 0
        for value in k_data_array:
            data_src.append(value[1])
            k_data_dest.append(value[0])
            k_counter += 1

        p_counter = 0
        for value in p_data_array:
            data_src.append(value[1])
            p_data_dest.append(value[0])
            p_counter += 1

        a_counter = 0
        for value in a_data_array:
            data_src.append(value[1])
            a_data_dest.append(value[0])
            a_counter += 1

        # saving
        with open('counters.pickle', 'wb') as handle:
            pickle.dump([k_counter, p_counter, a_counter], handle, protocol=pickle.HIGHEST_PROTOCOL)

        # saving
        with open('e_data_src.pickle', 'wb') as handle:
            pickle.dump(data_src, handle, protocol=pickle.HIGHEST_PROTOCOL)

        # saving
        with open('k_data_dest.pickle', 'wb') as handle:
            pickle.dump(k_data_dest, handle, protocol=pickle.HIGHEST_PROTOCOL)

        # saving
        with open('p_data_dest.pickle', 'wb') as handle:
            pickle.dump(p_data_dest, handle, protocol=pickle.HIGHEST_PROTOCOL)

        # saving
        with open('a_data_dest.pickle', 'wb') as handle:
            pickle.dump(a_data_dest, handle, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        # saving
        with open('counters.pickle', 'rb') as handle:
            counters = pickle.load(handle)

        # saving
        with open('e_data_src.pickle', 'rb') as handle:
            data_src = pickle.load(handle)

        # saving
        with open('k_data_dest.pickle', 'rb') as handle:
            k_data_dest = pickle.load(handle)

        # saving
        with open('p_data_dest.pickle', 'rb') as handle:
            p_data_dest = pickle.load(handle)

        # saving
        with open('a_data_dest.pickle', 'rb') as handle:
            a_data_dest = pickle.load(handle)

        k_counter, p_counter, a_counter = counters

    #since all includes political and keyword data one source tokenizer is needed
    tokenizer_src = TokenizerWrap(texts=data_src,
                                  padding='pre',
                                  reverse=True,
                                  num_words=num_words)

    a_tokenizer_dest = TokenizerWrap(texts=a_data_dest,
                                   padding='post',
                                   reverse=False,
                                   num_words=num_words)

    k_tokenizer_dest = TokenizerWrap(texts=k_data_dest,
                                   padding='post',
                                   reverse=False,
                                   num_words=len(queries) + 3)

    p_tokenizer_dest = TokenizerWrap(texts=p_data_dest,
                                     padding='post',
                                     reverse=False,
                                     num_words=2 + 3)

    tokens_src = tokenizer_src.tokens_padded
    k_tokens_dest = k_tokenizer_dest.tokens_padded
    p_tokens_dest = p_tokenizer_dest.tokens_padded
    a_tokens_dest = a_tokenizer_dest.tokens_padded

    #
    encoder_input_data = tokens_src
    k_decoder_input_data = k_tokens_dest[:, :-1]
    k_decoder_output_data = k_tokens_dest[:, 1:]
    p_decoder_input_data = p_tokens_dest[:, :-1]
    p_decoder_output_data = p_tokens_dest[:, 1:]
    a_decoder_input_data = a_tokens_dest[:, :-1]
    a_decoder_output_data = a_tokens_dest[:, 1:]

    k_model_train, p_model_train, a_model_train, \
    model_encoder, \
    k_model_decoder, p_model_decoder, a_model_decoder = get_model()

    k_callback_checkpoint = ModelCheckpoint(filepath=k_path_checkpoint,
                                          monitor='val_loss',
                                          verbose=1,
                                          save_weights_only=True,
                                          save_best_only=True)
    p_callback_checkpoint = ModelCheckpoint(filepath=k_path_checkpoint,
                                          monitor='val_loss',
                                          verbose=1,
                                          save_weights_only=True,
                                          save_best_only=True)
    a_callback_checkpoint = ModelCheckpoint(filepath=k_path_checkpoint,
                                          monitor='val_loss',
                                          verbose=1,
                                          save_weights_only=True,
                                          save_best_only=True)
    callback_early_stopping = EarlyStopping(monitor='val_loss',
                                            patience=3, verbose=1)

    callback_tensorboard = TensorBoard(log_dir='./21_logs/',
                                       histogram_freq=0,
                                       write_graph=False)

    k_callbacks = [callback_early_stopping,
                   k_callback_checkpoint,
                 callback_tensorboard]
    p_callbacks = [callback_early_stopping,
                   p_callback_checkpoint,
                 callback_tensorboard]
    a_callbacks = [callback_early_stopping,
                   a_callback_checkpoint,
                 callback_tensorboard]

    try:
        k_model_train.load_weights(k_callbacks)
    except Exception as error:
        print("Error trying to load checkpoint.")
        print(error)

    try:
        p_model_train.load_weights(p_callbacks)
    except Exception as error:
        print("Error trying to load checkpoint.")
        print(error)

    try:
        a_model_train.load_weights(a_callbacks)
    except Exception as error:
        print("Error trying to load checkpoint.")
        print(error)

    k_x_data = \
        {
            'encoder_input': encoder_input_data[:k_counter],
            'decoder_input': k_decoder_input_data
        }

    k_y_data = \
        {
            'decoder_output': k_decoder_output_data
        }

    p_x_data = \
        {
            'encoder_input': encoder_input_data[k_counter:k_counter+p_counter],
            'decoder_input': p_decoder_input_data
        }

    p_y_data = \
        {
            'decoder_output': p_decoder_output_data
        }

    a_x_data = \
        {
            'encoder_input': encoder_input_data[k_counter+p_counter:],
            'decoder_input': a_decoder_input_data
        }

    a_y_data = \
        {
            'decoder_output': a_decoder_output_data
        }

    validation_split = 500 / len(encoder_input_data)
    for _ in range(10):
        k_model_train.fit(x=k_x_data,
                        y=k_y_data,
                        batch_size=60,
                        epochs=1,
                        validation_split=validation_split,
                        callbacks=k_callbacks)

        p_model_train.fit(x=p_x_data,
                        y=p_y_data,
                        batch_size=60,
                        epochs=1,
                        validation_split=validation_split,
                        callbacks=p_callbacks)

        a_model_train.fit(x=a_x_data,
                        y=a_y_data,
                        batch_size=60,
                        epochs=1,
                        validation_split=validation_split,
                        callbacks=a_callbacks)

def get_vectors(input_videos):

    # loading
    with open('counters.pickle', 'rb') as handle:
        counters = pickle.load(handle)

    # loading
    with open('data_src.pickle', 'rb') as handle:
        data_src = pickle.load(handle)

    # loading
    with open('k_data_dest.pickle', 'rb') as handle:
        k_data_dest = pickle.load(handle)

    # loading
    with open('p_data_dest.pickle', 'rb') as handle:
        p_data_dest = pickle.load(handle)

    # loading
    with open('a_data_dest.pickle', 'rb') as handle:
        a_data_dest = pickle.load(handle)

    k_counter, p_counter, a_counter = counters

    tokenizer_src = TokenizerWrap(texts=data_src,
                                  padding='pre',
                                  reverse=True,
                                  num_words=num_words)

    a_tokenizer_dest = TokenizerWrap(texts=a_data_dest,
                                   padding='post',
                                   reverse=False,
                                   num_words=num_words)

    k_tokenizer_dest = TokenizerWrap(texts=k_data_dest,
                                   padding='post',
                                   reverse=False,
                                   num_words=len(queries) + 3)

    p_tokenizer_dest = TokenizerWrap(texts=p_data_dest,
                                     padding='post',
                                     reverse=False,
                                     num_words=2 + 3)

    k_model_train, p_model_train, a_model_train, \
    model_encoder, \
    k_model_decoder, p_model_decoder, a_model_decoder = get_model()

    try:
        k_model_train.load_weights(k_path_checkpoint)
    except Exception as error:
        print("Error trying to load k checkpoint.")
        print(error)

    try:
        p_model_train.load_weights(p_path_checkpoint)
    except Exception as error:
        print("Error trying to load p checkpoint.")
        print(error)


    try:
        a_model_train.load_weights(a_path_checkpoint)
    except Exception as error:
        print("Error trying to load a checkpoint.")
        print(error)

    keywords = []
    vectors = []
    for video in input_videos:
        keyword, vector = translate(model_encoder,
          k_model_decoder,
          tokenizer_src,
          k_tokenizer_dest,
          video.ptitle)
        keywords.append(keyword)
        vectors.append(vector)

    for video in input_videos:
        translate(model_encoder,
                  p_model_decoder,
                  tokenizer_src,
                  p_tokenizer_dest,
                  video.ptitle)

    for video in input_videos:
        translate(model_encoder,
                  a_model_decoder,
                  tokenizer_src,
                  a_tokenizer_dest,
                  video.ptitle)

    return keywords, vectors


def get_k_mean_clustered(input_videos, num_clusters = 40):

    vectors = []
    for video in input_videos:
        vectors.append(np.array(video.vector_processed[0]))

    np_vectors = np.array(vectors)

    def input_fn():
        return tf.train.limit_epochs(
            tf.convert_to_tensor(np_vectors, dtype=tf.float32), num_epochs=1)

    kmeans = tf.contrib.factorization.KMeansClustering(
        num_clusters=num_clusters, use_mini_batch=False)

    num_iterations = 10
    previous_centers = None

    for _ in range(num_iterations):
        kmeans.train(input_fn)
        cluster_centers = kmeans.cluster_centers()
        if previous_centers is not None:
            print ('delta:', cluster_centers - previous_centers)
        previous_centers = cluster_centers
    print ('cluster centers:', cluster_centers)

    # map the input points to their clusters
    cluster_indices = list(kmeans.predict_cluster_index(input_fn))

    for video, cluster_index in zip(input_videos, cluster_indices):
        setattr(video, "cluster", cluster_index)

    for cc in range(num_clusters):
        for i, point in enumerate(np_vectors):
            if cc == cluster_indices[i]:
                video_title = input_videos[i].title
                cluster_index = cluster_indices[i]
                print('video:', video_title, 'is in cluster', cluster_index)

    return cluster_centers

if __name__ == '__main__':
    train_model(reload=False)