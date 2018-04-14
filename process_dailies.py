from konlpy.tag import Twitter

import seq2conv
import operator
import numpy
import models_trainable
from peewee import IntegrityError

queries = ["테러", "사고", "건강", "일본", "북미",
           "한미", "정상회담", "선거", "김기식", "외교", "국방", "국회",
           "청화대", "비핵화", "자유한국당", "더불어민주당", "개헌", "문재인", "대통령",
           "이명박", "암호화폐", "핵무기",
           "중국", "미국", "북한",
           "FTA", "경제", "부동산",
           "미투", "박근혜"]

removables = ["날씨", "클로징", "다시", "헤드라인"]

priors = ["단독"]


class Dailies:

    def __init__(self, channels):
        self.channels = channels
        self.final_videos = []
        self.videos = []
        self.processed_channels = []
        self.nouns = []

        self.__process_videos()
        self.__process_vector()
        self.__process_noun()

    def __process_videos(self):

        twitter = Twitter()

        for videos in self.channels:
            for video in videos:
                title = video.title
                title_processed = ""
                title_list = twitter.pos(title, norm=True, stem=True)
                title_nouns = []
                for word, pumsa in title_list:
                    if not pumsa in ["Josa", "Eomi", "Punctuation", "URL", "Unknown"] \
                            and not word in ['SBS', 'JTBC', 'MBC', 'TVCHOSUN', 'KBS', '뉴스', 'News']:
                        title_processed += word
                        title_processed += " "
                        if pumsa == "Noun" and len(word) > 1:
                            title_nouns.append(word)
                print(title_processed)
                setattr(video, "ptitle", title_processed)
                setattr(video, "ntitle", title_nouns)

                content = video.content
                content_processed = ""
                content_list = twitter.pos(content, norm=True, stem=True)
                for word, pumsa in content_list:
                    if not pumsa in ["Josa", "Eomi", "Punctuation", "URL", "Unknown"]:
                        content_processed += word
                        content_processed += " "

                setattr(video, "pcontent", content_processed)

                self.videos.append(video)

    def __process_vector(self):
        keywords, vectors  = seq2conv.get_vectors(self.videos)
        for video, keyword, vector in zip(self.videos, keywords, vectors):
            setattr(video, "keyword_processed", keyword)
            setattr(video, "vector_processed", vector)

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
                    if first in video.ntitle and second in video.ntitle:
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
            if not self.nouns[i][0] in removable:
                try:
                    models_trainable.Keyword.create(
                        name=self.nouns[i][0],
                        t_type=3
                    )
                except IntegrityError:
                    print("Error : ", self.nouns[i][0])
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

    def process_vector_relevance(self):
        print(len(self.nouns))

        for word in self.nouns:
            print(word)

        filtered = []
        for video in self.videos:
            if video.duration < 120:
                filtered.append(video)

        relevance_matrix = numpy.zeros((len(filtered), len(filtered)))
        for i, video1 in enumerate(filtered):
            for j, video2 in enumerate(filtered):
                vector1 = video1.vector_processed
                vector2 = video2.vector_processed

                dist = numpy.linalg.norm(vector1 - vector2)
                relevance_matrix[i, j] = dist

        average = relevance_matrix.mean(0)

        #check for minimum relevent
        index = numpy.argmin(average)

        selected = []
        while len(selected) < len(filtered):
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
            column.append(filtered[index])

        return self.processed_channels

