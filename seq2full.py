import os
import tensorflow as tf
import numpy as np
import pickle

import models_trainable

from TokenizerWrap import TokenizerWrap
from tensorflow.contrib.tensorboard.plugins import projector

from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import Input, Dense, GRU, Embedding
from tensorflow.python.keras.optimizers import RMSprop
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard

from process_text import TextFilter
import numpy
import sys

f_path_checkpoint = './check_p/f_21_checkpoint.keras'

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

class Seq2MSeq():

    def __init__(self, initial=False):

        # loading
        if not initial:
            with open('data_src.pickle', 'rb') as handle:
                data_src = pickle.load(handle)

            # loading
            with open('f_data_dest.pickle', 'rb') as handle:
                f_data_dest = pickle.load(handle)

            with open('f_count.pickle', 'rb') as handle:
                f_count = pickle.load(handle)

            self.tokenizer_src = TokenizerWrap(texts=data_src,
                                          padding='pre',
                                          reverse=True,
                                          num_words=num_words)

            self.f_tokenizer_dest = TokenizerWrap(texts=f_data_dest,
                                             padding='post',
                                             reverse=False,
                                             num_words=int(f_count))

            self.f_model_train, self.f_model_encoder = self.get_model(f_count)

            try:
                self.f_model_train.load_weights(f_path_checkpoint)
            except Exception as error:
                print("Error trying to load k checkpoint.")
                print(error)


    def sparse_cross_entropy(self, y_true, y_pred):

        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y_true,
                                                              logits=y_pred)

        loss_mean = tf.reduce_mean(loss)

        return loss_mean

    def get_vector(self, model_encoder,
                  tokenizer_src,
                  input_text,
                  true_output_text=None):

        input_tokens = tokenizer_src.text_to_tokens(text=input_text,
                                                    reverse=True,
                                                    padding=True)

        initial_state = model_encoder.predict(input_tokens)

        return initial_state

    def get_model(self, num_keys):

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

        encoder_dense_1 = Dense(state_size,
                              activation='relu',
                              name='encoded_output_1')

        encoder_dense_2 = Dense(state_size,
                              activation='relu',
                              name='encoded_output_2')

        encoder_embedding = Embedding(input_dim=state_size,
                                      output_dim=embedding_size,
                                      name='encoded_embedding')

        encoder_dense_out = Dense(num_keys,
                              activation='linear',
                              name='encoded_output_3')


        def connect_encoder():
            # Start the neural network with its input-layer.
            net = encoder_input

            # Connect the embedding-layer.
            net = encoder_embedding(net)

            # Connect all the GRU-layers.
            net = encoder_gru1(net)
            net = encoder_gru2(net)
            net = encoder_gru3(net)

            net = encoder_dense_1(net)
            net = encoder_dense_2(net)

            # This is the output of the encoder.
            emdedding = encoder_embedding(net)

            encoder_output = encoder_dense_out(net)

            return encoder_output, emdedding

        encoder_output, emdedding = connect_encoder()


        f_model_train = Model(inputs=[encoder_input],
                                outputs=[encoder_output])

        f_model_encoder = Model(inputs=[encoder_input],
                                    outputs=[emdedding])

        optimizer = RMSprop(lr=1e-3)

        decoder_target = tf.placeholder(dtype='int32', shape=(None, None))

        f_model_train.compile(optimizer=optimizer,
                            loss=self.sparse_cross_entropy,
                            target_tensors=[decoder_target])


        return f_model_train, f_model_encoder


    def train_model(self, reload=False):

        models_trainable.initialized()

        data_src = []

        f_data_array = []
        f_data_dest = []

        if reload:

            text_filter = TextFilter()

            keyword_models = models_trainable.Keyword.select().where(
                models_trainable.Keyword.t_type >= 1,
                models_trainable.Keyword.t_type <= 4
            )
            keywords = []

            for keyword_model in keyword_models:
                keywords.append(keyword_model.name)

            videos = models_trainable.Video.select()

            for i, video in enumerate(videos):
                title = video.title
                text_filter.set_text(title)

                text_filter.regex_from_text(r'\[[^)]*\]')
                text_filter.remove_texts_from_text()
                text_filter.remove_pumsas_from_list()
                text_filter.remove_texts_from_text()

                matches = text_filter.get_matches(keywords)
                for keyword in matches:
                    f_data_array.append([keyword,
                                         str(text_filter)])

            f_count = len(keywords)
            print(len(f_data_array))
            for value in f_data_array:
                data_src.append(value[1])
                f_data_dest.append(value[0])


            # saving
            with open('f_count.pickle', 'wb') as handle:
                pickle.dump(len(keyword_models), handle, protocol=pickle.HIGHEST_PROTOCOL)

            # saving
            with open('data_src.pickle', 'wb') as handle:
                pickle.dump(data_src, handle, protocol=pickle.HIGHEST_PROTOCOL)

            # saving
            with open('f_data_dest.pickle', 'wb') as handle:
                pickle.dump(f_data_dest, handle, protocol=pickle.HIGHEST_PROTOCOL)

        else:
            # saving
            with open('f_count.pickle', 'rb') as handle:
                f_count = pickle.load(handle)

            # saving
            with open('data_src.pickle', 'rb') as handle:
                data_src = pickle.load(handle)

            # saving
            with open('f_data_dest.pickle', 'rb') as handle:
                f_data_dest = pickle.load(handle)


        #since all includes political and keyword data one source tokenizer is needed
        tokenizer_src = TokenizerWrap(texts=data_src,
                                      padding='pre',
                                      reverse=True,
                                      num_words=num_words)

        f_tokenizer_dest = TokenizerWrap(texts=f_data_dest,
                                       padding='post',
                                       reverse=False,
                                       num_words=f_count)

        tokens_src = tokenizer_src.tokens_padded
        f_tokens_dest = f_tokenizer_dest.tokens_padded

        #
        encoder_input_data = tokens_src
        #f_decoder_output_data = np.asarray(f_tokens_dest).reshape(-1)
        f_decoder_output_data = f_tokens_dest

        f_model_train, f_model_embedding = self.get_model(f_count)

        f_callback_checkpoint = ModelCheckpoint(filepath=f_path_checkpoint,
                                              monitor='val_loss',
                                              verbose=1,
                                              save_weights_only=True,
                                              save_best_only=True)

        callback_early_stopping = EarlyStopping(monitor='val_loss',
                                                patience=3, verbose=1)

        callback_tensorboard = TensorBoard(log_dir='./f_logs/',
                                           histogram_freq=0,
                                           write_graph=False)

        f_callbacks = [callback_early_stopping,
                       f_callback_checkpoint,
                     callback_tensorboard]

        try:
            f_model_train.load_weights(f_callbacks)
        except Exception as error:
            print("Error trying to load checkpoint.")
            print(error)

        f_x_data = \
            {
                'encoder_input': encoder_input_data
            }

        f_y_data = \
            {
                'encoded_output_3': f_decoder_output_data
            }


        validation_split = 500 / len(encoder_input_data)
        for _ in range(10):
            f_model_train.fit(x=f_x_data,
                            y=f_y_data,
                            batch_size=240,
                            epochs=1,
                            validation_split=validation_split,
                            callbacks=f_callbacks)


    def get_vectors(self, input_videos):

        vectors = []

        for i, video in enumerate(input_videos):

            vector = self.get_vector(self.model_encoder,
                                     self.tokenizer_src,
                                        video.ptitle)
            vectors.append(vector)

        return vectors


    def get_k_mean_clustered(self, input_videos, num_clusters = 40):

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

    def projection(self, input_videos):

        LOG_DIR = 'logs'
        metadata = os.path.join('metadata.tsv')

        input_data = []
        labels = []
        for video in input_videos:
            input_data.append(numpy.frombuffer(video.vector).tolist())
            labels.append(video.channel.name +" "+video.title)

        images = tf.Variable(input_data, name='data')

        with open(metadata, 'w') as metadata_file:
            for row in labels:
                metadata_file.write('%s\n' % row)

        with tf.Session() as sess:
            saver = tf.train.Saver([images])

            sess.run(images.initializer)
            saver.save(sess, os.path.join(LOG_DIR, 'title.ckpt'))

            config = projector.ProjectorConfig()
            # One can add multiple embeddings.
            embedding = config.embeddings.add()
            embedding.tensor_name = images.name
            # Link this tensor to its metadata file (e.g. labels).
            embedding.metadata_path = metadata
            # Saves a config file that TensorBoard will read during startup.
            projector.visualize_embeddings(tf.summary.FileWriter(LOG_DIR), config)

if __name__ == '__main__':
    seq2mseq = Seq2MSeq()
    seq2mseq.train_model(reload=False)