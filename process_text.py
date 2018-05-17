from konlpy.tag import Twitter
import re

class TextFilter:

    def __init__(self):
        self.__twitter = Twitter()
        self.__text = ""
        self.__text_list = []
        self.__processed_text = []

    def set_text(self, text):
        self.__text = text
        self.__processed_text = text
        self.process_text(self.__processed_text)

    def process_text(self, text):
        self.__text_list = self.__twitter.pos(text, norm=True, stem=True)
        self.__processed_text_list = self.__text_list
        return str(self)

    def regex_from_text(self, removable_regex, refresh=False):
        if refresh:
            self.__processed_text = self.__text

        self.__processed_text = re.sub(removable_regex, "", self.__processed_text)

        self.process_text(self.__processed_text)

        return str(self)

    def remove_texts_from_text(self,
                                removable_text=['</b>', '<b>', '&quot;', '&apos;', '…'],
                                refresh=False):
        if refresh:
            self.__processed_text = self.__text

        for ch in removable_text:
            self.__processed_text = self.__processed_text.replace(ch, "")

        self.process_text(self.__processed_text)

        return str(self)


    def remove_pumsas_from_list(self,
                      removable_pumsas = ["Josa", "Eomi", "Punctuation", "URL", "Unknown"],
                      refresh=False):

        if refresh:
            self.__processed_text_list = self.__text_list

        processed_text_temp = []

        for word, pumsa in self.__processed_text_list:
            if not pumsa in removable_pumsas:
                processed_text_temp.append((word, pumsa))

        self.__processed_text_list = processed_text_temp

        return str(self)

    def remove_texts_from_list(self,
                     removable_texts = ['SBS', 'JTBC', 'MBC', 'TVCHOSUN', 'KBS', '뉴스', 'News', '뉴스투데이', '뉴스데스크'],
                     refresh=False):

        if refresh:
            self.__processed_text_list = self.__text_list

        processed_text_temp = []

        for word, pumsa in self.__processed_text_list:
            if not word in removable_texts:
                processed_text_temp.append((word, pumsa))

        self.__processed_text_list = processed_text_temp

        return str(self)

    def get_texts_from_list(self, pumsa, refresh=False):

        if refresh:
            self.__processed_text_list = self.__text_list

        text_nouns = []

        for _word, _pumsa in self.__processed_text_list:
            if _pumsa in pumsa:
                text_nouns.append(_word)

        return text_nouns

    def __str__(self):
        return " ".join(str(x[0]) for x in self.__processed_text_list)