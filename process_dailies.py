from process_text import TextFilter
import models_trainable
#import seq2conv
#from seq2mseq import Seq2MSeq
import seq2keyword

import operator
import numpy
from peewee import IntegrityError
import array

queries = ["테러", "사고", "건강", "일본", "북미",
           "한미", "정상회담", "선거", "김기식", "외교", "국방", "국회",
           "청화대", "비핵화", "자유한국당", "더불어민주당", "개헌", "문재인", "대통령",
           "이명박", "암호화폐", "핵무기",
           "중국", "미국", "북한",
           "FTA", "경제", "부동산",
           "미투", "박근혜"]

removables = ["날씨", "클로징", "다시보기", "헤드라인"]

priors = ["단독"]

class Dailies:

    def __init__(self, channels = None, retrain = False):

        self.channels = channels
        self.final_videos = []
        self.videos = []
        self.filtered = []
        self.processed_channels = []
        self.nouns = []
        #self.seq2mseq = Seq2MSeq()
        if channels:
            self.preprocess(channels, retrain)


    def preprocess(self, channels, retrain = False):

        self.channels = channels
        self.final_videos = []
        self.videos = []
        self.filtered = []
        self.processed_channels = []
        self.nouns = []

        self.__process_videos()

        if retrain:
            self.__process_vector()
            self.__process_noun()
        else:
            self.__process_vector_from_prev()
        self.__filter_videos()

    def __process_videos(self):

        text_filter = TextFilter()
        for videos in self.channels:
            for video in videos:
                text_filter.set_text(video.title)

                text_filter.regex_from_text(r'\[[^)]*\]')
                text_filter.remove_texts_from_text()
                text_filter.remove_pumsas_from_list()
                text_filter.remove_texts_from_list()

                setattr(video, "ptitle", str(text_filter))
                setattr(video, "ntitle", text_filter.get_texts_from_list("Noun"))

                self.videos.append(video)

    def __process_vector(self):
        #vectors  = self.seq2mseq.get_vectors(self.videos)
        strings, vectors = seq2keyword.get_vectors(self.videos)

        for video, vector in zip(self.videos, vectors):
            setattr(video, "vector_processed", vector)
            video.vector = numpy.array(vector, numpy.float32).tobytes()
            video.save()

    def __process_vector_from_prev(self):
        for video in self.videos:
            if len(video.vector)>0:
                value = numpy.frombuffer(video.vector)
                setattr(video, "vector_processed", numpy.array([value,]))
            video.save()

    def __process_noun(self):
        noun_dict = {}
        for video in self.videos:
            for noun in video.ntitle:
                if noun in noun_dict.keys():
                    noun_dict[noun] += 1
                else:
                    noun_dict[noun] = 1
        self.nouns = sorted(noun_dict.items(), key=operator.itemgetter(1), reverse=True)
        combination_dict = {}
        for video in self.videos:
            for i in range(30):
                for j in range(i+1, 30):
                    first = self.nouns[i][0]
                    second = self.nouns[j][0]
                    if first in video.ntitle and second in video.ntitle and len(first) > 1:
                        combination_noun = first+" "+second
                        if combination_noun in combination_dict.keys():
                            combination_dict[combination_noun] += 1
                        else:
                            combination_dict[combination_noun] = 1

        removable = []
        for key in combination_dict.keys():
            values = key.split()
            removable.append(values[1])

        for i in range(30):
            noun = self.nouns[i][0]
            if not noun in removable:
                try:
                    print("Trying to add Keyword : {}".format(noun))
                    models_trainable.Keyword.create(
                        name=noun
                    )
                except IntegrityError:
                    print("Already Exist Keyword : ", self.nouns[i][0])
                    pass

    def process_relevance(self):

        video_dict = {}

        for video in self.videos:
            master = ""

            if video.duration > 120:
                master = "Removables"

            for prior in priors:
                if prior in video.ptitle.split():
                    master = "Prior"
                    break

            for removable in removables:
                if removable in video.ptitle.split():
                    master = "Removables"
                    break

            for removable in removables:
                if removable in video.title.split():
                    master = "Removables"
                    break

            for query in queries:
                if query in video.ptitle.split():
                    master = query
                    break

            if master == "":
                master = video.keyword_processed

            if master in video_dict.keys():
                video_dict[master].append(video)
            else:
                video_dict[master] = [video]
            video.keyword_processed = master

        prior_processed = []
        unknown_processed = []
        removable_processed = []
        multi_processed = []

        for video_vect in video_dict.keys():
            if "unknown" in video_vect.split():
                for video in video_dict[video_vect]:
                    videos_col = [video]
                    unknown_processed.append(videos_col)
            elif "Removables" in video_vect.split():
                videos_col = []
                for i in range(max(len(video_dict[video_vect]), 5)):
                    if i < len(video_dict[video_vect]):
                        videos_col.append(video_dict[video_vect][i])
                removable_processed.append(videos_col)
            elif "Prior" in video_vect.split():
                for video in video_dict[video_vect]:
                    videos_col = [video]
                    prior_processed.append(videos_col)
            else:
                videos_col = []
                for i in range(max(len(video_dict[video_vect]), 5)):
                    if i < len(video_dict[video_vect]):
                        videos_col.append(video_dict[video_vect][i])

                multi_processed.append(videos_col)

        #Sorting
        multi_processed = sorted(multi_processed, key=lambda x: x[0].relevance, reverse=True)
        unknown_processed = sorted(unknown_processed, key=lambda x: x[0].relevance, reverse=True)

        #Empty processed channels
        self.processed_channels = []

        #Extend found channels
        self.processed_channels.extend(prior_processed)
        self.processed_channels.extend(multi_processed)
        self.processed_channels.extend(unknown_processed)
        self.processed_channels.extend(removable_processed)

        return self.processed_channels

    def __filter_videos(self):

        #filter removables and/or video that are more than 2 minute
        self.filtered = []
        for video in self.videos:
            include = True

            if len(video.vector)==0:
                include = False

            for removable in removables:
                if removable in video.title:
                    include = False

            if video.duration > 120:
                include = False

            if include:
                self.filtered.append(video)

    def process_vector_relevance(self):

        #Relevance matrix calculated using vectors from ML
        relevance_matrix = numpy.zeros((len(self.filtered), len(self.filtered)))
        for i, video1 in enumerate(self.filtered):
            for j, video2 in enumerate(self.filtered):
                vector1 = video1.vector_processed
                vector2 = video2.vector_processed

                dist = numpy.linalg.norm(vector1 - vector2)
                relevance_matrix[i, j] = dist

        average = relevance_matrix.mean(0)

        #check for minimum relevent
        index = numpy.argmin(average)

        selected = []
        while len(selected) < len(self.filtered):
            sorted_values = numpy.argsort(relevance_matrix[index])
            for next_index in sorted_values:
                if not next_index in selected:
                    index = next_index
                    selected.append(index)
                    break

        self.processed_channels = []
        for i, index in enumerate(selected):
            if i % 3 == 0:
                column = []
                self.processed_channels.append(column)
            column.append(self.filtered[index])

        return self.processed_channels


    def process_kmeans_clusters(self, numb_clusters=30):

        #mean_cluster = self.seq2mseq.get_k_mean_clustered(self.filtered, numb_clusters)
        mean_cluster = seq2keyword.get_k_mean_clustered(self.filtered, numb_clusters)

        self.processed_channels = []

        bucket_video = []
        bucket_vector = []
        for i in range(numb_clusters):
            bucket_video.append([])
            bucket_vector.append([])

        for video in self.filtered:
            bucket_video[video.cluster].append(video)
            dist = numpy.linalg.norm(mean_cluster[video.cluster] - video.vector_processed)
            bucket_vector[video.cluster].append(dist)

        bucket_vector = numpy.array(bucket_vector)

        bucket_lens = []
        for i, vectors in enumerate(bucket_vector):
            bucket_lens.append(len(bucket_vector))
            sorted_values = numpy.argsort(vectors)
            column = []
            for j, index in enumerate(sorted_values):
                if len(column)> 2:
                    break

                if j==0:
                    column.append(bucket_video[i][index])
                else:
                    if column[0].channel.name != bucket_video[i][index].channel.name:
                        column.append(bucket_video[i][index])
            self.processed_channels.append(column)


        self.processed_channels = sorted(self.processed_channels, key=lambda x: len(x), reverse=True)

        return self.processed_channels
