from datetime import datetime, timedelta
from typing import List

import requests
import tqdm

from news.article import NewsArticle, NewsSite, Country
from news.parsers.manager import ParsersManager


class NewsGetter:
    __API_URL = 'https://newsapi.org/v2'

    def __init__(self, api_key: str):
        self.__session = requests.Session()
        self.__session.headers.update({'Authorization': f'{api_key}'})
        self.__parsers_manager = ParsersManager()

    def get_articles(self, country: Country) -> List[NewsArticle]:
        url = f'{NewsGetter.__API_URL}/everything'
        parameters = {
            'pageSize': 100,
            'sources': ','.join(self.__parsers_manager.get_news_sites(country.code)),
            'from': datetime.now() - timedelta(days=1),
            'language': country.language,
            'sortBy': 'popularity',
        }

        response = self.__session.get(url, params=parameters)
        if response.status_code == requests.codes.ok:
            response = response.json()
            articles = []
            for article in tqdm.tqdm(response['articles'], desc='Parsing articles'):
                news_site = NewsSite(article['source']['name'], article['source']['id'], country)
                content = self.__parsers_manager.get_article_content(country.code, news_site.code, article['url'])
                if len(content) == 0:  # Skip if article content can not be extracted
                    continue

                articles.append(NewsArticle(
                    title=article['title'],
                    content=' '.join(content),
                    article_url=article['url'],
                    article_date=datetime.fromisoformat(article['publishedAt'][:-1]),
                    news_site=news_site,
                    image_url=article['urlToImage']
                ))

            return articles

        else:
            print(f'HTTP {response.status_code} code when getting news articles. Error message: {response.text}.')
