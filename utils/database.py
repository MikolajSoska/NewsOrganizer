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

    def add_new_article(self, article: NewsArticle, dataset_name: str) -> None:
        query = 'SELECT id FROM news_sites WHERE code = %s'
        self.__cursor.execute(query, (article.news_site.code,))
        site_id = self.__cursor.fetchone()[0]

        query = 'INSERT INTO news_articles VALUES (0, %s, %s, %s, %s, %s, %s, %s)'
        self.__cursor.execute(query, (article.title, article.content, article.article_url,
                                      article.article_date, site_id, article.image_url, article.summary))

        article_id = self.__cursor.lastrowid
        tag_category_dict = self.__get_tag_category_dict(dataset_name)
        query = 'INSERT INTO article_tag_map VALUES (0, %s, %s, %s, %s, %s)'
        for category_name, position, length, words in article.named_entities:
            self.__cursor.execute(query, (article_id, tag_category_dict[category_name], position, length, words))

        self.__database.commit()

    def get_tag_count(self, dataset_name: str) -> int:
        query = 'SELECT COUNT(*) FROM tags t INNER JOIN tag_categories tc ON t.category_id = tc.id INNER JOIN ' \
                'datasets d on tc.dataset_id = d.id WHERE name = %s'
        self.__cursor.execute(query, (dataset_name,))

        return self.__cursor.fetchone()[0]

    def get_tags_dict(self, dataset_name: str) -> Dict[int, Tuple[str, str]]:
        query = 'SELECT tag_label, tag, category_name FROM tags INNER JOIN tag_categories tc ON ' \
                'tags.category_id = tc.id INNER JOIN datasets d on tc.dataset_id = d.id WHERE d.name = %s'
        self.__cursor.execute(query, (dataset_name,))

        return {tag_label: (tag, category_name) for tag_label, tag, category_name in self.__cursor.fetchall()}

    def __get_tag_category_dict(self, dataset_name: str) -> Dict[str, int]:
        query = 'SELECT category_name, tc.id FROM tag_categories tc INNER JOIN datasets d ON tc.dataset_id = d.id ' \
                'WHERE d.name = %s'
        self.__cursor.execute(query, (dataset_name,))

        return dict(self.__cursor.fetchall())

    def get_articles(self) -> List[NewsArticle]:
        query = 'SELECT * FROM news_articles'
        self.__cursor.execute(query)
        articles = []
        for article_id, title, content, url, date, site_id, image_url, summary in self.__cursor.fetchall():
            news_site = self.__get_news_site(site_id)
            article = NewsArticle(title, content, url, date, news_site, image_url, summary)
            article.named_entities = self.__get_named_entities(article_id)
            articles.append(article)

        return articles

    def get_article_tags_count(self) -> Dict[str, Counter]:
        query = 'SELECT DISTINCT article_id, words FROM article_tag_map map INNER JOIN tag_categories tc ON ' \
                'map.tag_category_id = tc.id WHERE category_name = %s'
        categories_name = self.__get_tag_categories()
        tag_counts = {}
        for category in categories_name:
            self.__cursor.execute(query, (category,))
            words = [word for _, word in self.__cursor.fetchall()]
            tag_counts[category] = Counter(words)

        return tag_counts

    def __get_tag_categories(self) -> List[str]:
        query = 'SELECT category_name FROM tag_categories'

        self.__cursor.execute(query)
        return [name[0] for name in self.__cursor.fetchall()]

    def __get_news_site(self, site_id: int) -> NewsSite:
        query = 'SELECT news_sites.name, news_sites.code FROM news_sites INNER JOIN countries ON ' \
                'news_sites.country_id = countries.id WHERE news_sites.id = %s'
        self.__cursor.execute(query, (site_id,))
        name, code = self.__cursor.fetchone()

        return NewsSite(name, code, Country('United States', 'us', 'en'))

    def __get_named_entities(self, article_id: int) -> List[Tuple[str, int, int, str]]:
        query = 'SELECT category_name, position, length, words FROM article_tag_map map INNER JOIN tag_categories tc ' \
                'ON map.tag_category_id = tc.id WHERE article_id = %s'
        self.__cursor.execute(query, (article_id,))
        return self.__cursor.fetchall()
