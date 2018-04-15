from konlpy.tag import Twitter


class TextFilter:

    def __init__(self):
        self.__twitter = Twitter()
        self.__text = ""
        self.__text_list = []
        self.__processed_text = []

    def set_text(self, text):
        self.__text = text
        self.__text_list = self.__twitter.pos(self.__text, norm=True, stem=True)
        self.__processed_text = self.__text_list

    def remove_pumsas(self,
                      removable_pumsas = ["Josa", "Eomi", "Punctuation", "URL", "Unknown"],
                      refresh=False):

        if refresh:
            self.__processed_text = self.__text_list

        processed_text_temp = []

        for word, pumsa in self.__processed_text:
            if not pumsa in removable_pumsas:
                processed_text_temp.append((word, pumsa))

        self.__processed_text = processed_text_temp

        return str(self)

    def remove_texts(self,
                     removable_texts = ['SBS', 'JTBC', 'MBC', 'TVCHOSUN', 'KBS', '뉴스', 'News'],
                     refresh=False):

        if refresh:
            self.__processed_text = self.__text_list

        processed_text_temp = []

        for word, pumsa in self.__processed_text:
            if not pumsa in removable_texts:
                processed_text_temp.append((word, pumsa))

        self.__processed_text = processed_text_temp

        return str(self)

    def get_texts_pumsa(self, pumsa, refresh=False):

        if refresh:
            self.__processed_text = self.__text_list

        text_nouns = []

        for _word, _pumsa in self.__processed_text:
            if _pumsa in pumsa:
                text_nouns.append(_word)

        return text_nouns

    def __str__(self):
        return " ".join(str(x[0]) for x in self.__processed_text)