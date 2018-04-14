import urllib.request
import urllib.parse
import models_trainable
from bs4 import BeautifulSoup

from dateutil.parser import parse
models_trainable.initialized()

# Default query keywords
queries = ["테러", "사고", "건강", "일본", "북미",
           "한미", "정상회담", "선거", "김기식", "외교", "국방", "국회",
           "청화대", "비핵화", "자유한국당", "더불어민주당", "개헌", "문재인", "대통령",
           "이명박", "암호화폐", "핵무기",
           "날씨", "중국", "미국", "북한",
           "FTA", "경제", "부동산",
           "미투", "박근혜"]

# Default unknown queries
queries_unknown = ["쓰레기", "보이스피싱", "야구", "농구",
                   "스포츠", "개임", "자율주행",
                   "UAE", "졸음운전", "몰래카메라", "골프", "스마트폰", "전자발찌", "커피"
                                                                 "술", "마약", "폭력"]

#
def crawl_naver(is_go=True, keywords=[], keywords_unknown=[]):

    defaultURL = 'https://openapi.naver.com/v1/search/news.xml?'

    keyword_models = models_trainable.Keyword.select().where(models_trainable.Keyword.t_type == 1)

    for keyword_model in keyword_models:

        print(keyword_model.name)

        for i in range(10):
            sort = 'sort=sim'
            start = '&start={}'.format(i*100+1)
            display = '&display=100'
            query = '&query=' + urllib.parse.quote_plus(str(keyword_model.name))
            fullURL = defaultURL + sort + start + display + query

            headers = {
                'Host' : 'openapi.naver.com',
                'User-Agent' : 'curl/7.43.0',
                'Accept' : '*/*',
                'Content-Type' : 'application/xml',
                'X-Naver-Client-Id' : 'ID 넣으세',
                'X-Naver-Client-Secret' : 'Key 넣으세'
            }

            req = urllib.request.Request(fullURL, headers=headers)
            f = urllib.request.urlopen(req)
            result_xml = f.read()
            xml_soup = BeautifulSoup(result_xml, 'html.parser')
            items = xml_soup.find_all('item')

            store_naver_items(items, is_go, keyword_model)


def store_naver_items(items, is_go, keyword_model):

    for item in items:
        video_model = None
        try:

            video_model = models_trainable.Video.create(
                publishedAt=parse(item.pubdate.get_text(strip=True)),
                title=item.title.get_text(strip=True),
                content=item.description.get_text(strip=True),
                originalLink=item.originallink.get_text(strip=True)
            )

        except models_trainable.IntegrityError:
            print("Video not created")
            if not is_go:
                print("time : " + item.title.get_text(strip=True) + item.pubdate.get_text(strip=True))
                break

        if not video_model:
            video_model = models_trainable.Video.select().where(
                models_trainable.Video.originalLink ** item.originallink.get_text(strip=True)).get()

        try:
            print(video_model, keyword_model)
            models_trainable.Relationship.create(
                from_keyword=keyword_model,
                to_video=video_model
            )

        except models_trainable.IntegrityError:
            print("Relationship Already Exists")
            pass


if __name__ == '__main__':
    crawl_naver(keywords_unknown=[])