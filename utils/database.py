from collections import Counter
from typing import List, Tuple, Dict

import mysql.connector as mysql

from news.article import NewsSite, Country, NewsArticle
from utils.singleton import Singleton


class DatabaseConnector(metaclass=Singleton):
    def __init__(self):
        self.__database = mysql.connect(
            host='localhost',
            user='root',
            password='12345678',
            database='news'
        )
        self.__cursor = self.__database.cursor()

    def get_countries(self) -> List[Country]:
        query = 'SELECT c.name, c.code, l.code FROM countries c INNER JOIN languages l on c.language_id = l.id'
        self.__cursor.execute(query)

        return [Country(name, code, language) for name, code, language in self.__cursor.fetchall()]

    def get_news_sites(self, country: Country) -> List[NewsSite]:
        query = 'SELECT news_sites.name, news_sites.code FROM news_sites INNER JOIN countries ON ' \
                'news_sites.country_id = countries.id WHERE countries.code = %s'
        self.__cursor.execute(query, (country.code,))

        return [NewsSite(name, code, country) for name, code in self.__cursor.fetchall()]

    def add_new_article(self, article: NewsArticle, tokens: List[str]) -> None:
        query = 'SELECT id FROM news_sites WHERE code = %s'
        self.__cursor.execute(query, (article.news_site.code,))
        site_id = self.__cursor.fetchone()[0]

        query = 'INSERT INTO news_articles VALUES (0, %s, %s, %s, %s, %s, %s, %s)'
        self.__cursor.execute(query, (article.title, article.content, article.article_url,
                                      article.article_date, site_id, article.image_url, article.summary))
        self.__database.commit()
        article_id = self.__cursor.lastrowid
        for position, tag in article.named_entities.items():
            tag_id = self.__get_tag_id(tag)
            word = tokens[position]
            query = 'INSERT INTO article_tag_map VALUES (0, %s, %s, %s, %s)'
            self.__cursor.execute(query, (article_id, tag_id, word, position))
            self.__database.commit()

    def get_tag_count(self, dataset_name: str) -> int:
        query = 'SELECT COUNT(*) FROM tags INNER JOIN datasets d on tags.dataset_id = d.id WHERE name = %s'
        self.__cursor.execute(query, (dataset_name,))

        return self.__cursor.fetchone()[0]

    def __get_tag_id(self, tag: str) -> int:
        query = 'SELECT id FROM tags WHERE tag = %s'
        self.__cursor.execute(query, (tag,))

        return self.__cursor.fetchone()[0]

    # TODO refactor (chodzi o te data)
    def get_articles(self) -> List[NewsArticle]:
        query = 'SELECT * FROM news_articles'
        self.__cursor.execute(query)
        articles = []
        for data in self.__cursor.fetchall():
            news_site = self.__get_news_site(data[5])
            article = NewsArticle(data[1], data[2], data[3], data[4], news_site, data[6], data[7])
            for position, tag in self.__get_named_entities(data[0]):
                article.named_entities[position] = tag
            articles.append(article)

        return articles

    def get_article_tags_count(self) -> Dict[str, Counter]:
        query = 'SELECT DISTINCT article_id, word FROM article_tag_map map INNER JOIN tags t on map.tag_id = t.id ' \
                'WHERE fullname = %s'
        tag_names = self.__get_tag_names()
        tag_counts = {}
        for tag in tag_names:
            self.__cursor.execute(query, (tag,))
            words = [word for _, word in self.__cursor.fetchall()]
            tag_counts[tag] = Counter(words)

        return tag_counts

    def __get_tag_names(self) -> List[str]:
        query = 'SELECT fullname FROM tags'

        self.__cursor.execute(query)
        return [name[0] for name in self.__cursor.fetchall()]

    def __get_news_site(self, site_id: int) -> NewsSite:
        query = 'SELECT news_sites.name, news_sites.code FROM news_sites INNER JOIN countries ON ' \
                'news_sites.country_id = countries.id WHERE news_sites.id = %s'
        self.__cursor.execute(query, (site_id,))
        name, code = self.__cursor.fetchone()

        return NewsSite(name, code, Country('United States', 'us', 'en'))

    def __get_named_entities(self, article_id: int) -> List[Tuple[int, str]]:
        query = 'SELECT position, tag_short FROM article_tag_map map INNER JOIN tags t on map.tag_id = t.id ' \
                'WHERE article_id = %s'
        self.__cursor.execute(query, (article_id,))
        return self.__cursor.fetchall()
