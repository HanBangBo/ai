import re
import requests
from bs4 import BeautifulSoup

headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) '
                        'Chrome/102.0.0.0 Safari/537.36'}


def extract_korean_news(title, url):
    try:
        get_url = requests.get(url=url, headers=headers)
    except:
        return
    bs = BeautifulSoup(get_url.text, 'lxml')
    content = bs.find('article', id='dic_area')
    if content is None:
        return
    unexpectes = [content.find('strong')] + content.find_all('span', {'class': 'end_photo_org'})
    for exception in unexpectes:
        if exception is not None:
            exception.extract()
    content = content.text
    for i in range(2):
        content = re.sub('  ', ' ', content)
    content = re.sub('\n|\t|\xa0', ' ', content)
    return {'title': title, 'content': content}


print(extract_korean_news("title", "https://n.news.naver.com/mnews/article/658/0000098548"))
