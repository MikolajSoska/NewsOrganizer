import pickle
from collections import Counter
from collections import defaultdict
from typing import List, Tuple, Dict

import mysql.connector as mysql

from news.article import NewsSite, Country, NewsArticle, NamedEntity, NewsModel
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

    def get_datasets_identifiers(self, task_name: str) -> List[str]:
        query = 'SELECT id_name FROM datasets d INNER JOIN tasks t on d.task_id = t.id WHERE t.name = %s'
        self.__cursor.execute(query, (task_name,))

        return [dataset[0] for dataset in self.__cursor.fetchall()]

    def get_models_by_dataset(self, dataset_identifier: str) -> List[NewsModel]:
        query = 'SELECT nm.id, model_name, model_identifier, class_name, constructor_args,dataset_args, batch_size ' \
                'FROM news_models nm INNER JOIN models m ON nm.model_id = m.id INNER JOIN datasets d ON ' \
                'nm.dataset_id = d.id WHERE d.id_name = %s'
        self.__cursor.execute(query, (dataset_identifier,))

        models = []
        data = self.__cursor.fetchall()
        for model_id, model_name, name_identifier, class_name, constructor_args, dataset_args, batch_size in data:
            model = NewsModel(
                model_id=model_id,
                fullname=model_name,
                class_name=class_name,
                name_identifier=name_identifier,
                constructor_args=pickle.loads(constructor_args),
                dataset_args=pickle.loads(dataset_args),
                batch_size=batch_size,
                dataset_name=dataset_identifier
            )
            models.append(model)

        return models

    def get_models_by_task(self, task_name: str, full_dataset_name: bool = False) -> List[NewsModel]:
        dataset_name_column = 'full_name' if full_dataset_name else 'id_name'
        query = f'SELECT nm.id, model_name, model_identifier, class_name, constructor_args, dataset_args, ' \
                f'batch_size, d.{dataset_name_column} FROM news_models nm INNER JOIN models m ON nm.model_id = m.id ' \
                f'INNER JOIN datasets d ON nm.dataset_id = d.id INNER JOIN tasks t on d.task_id = t.id ' \
                f'WHERE t.name = %s ORDER BY m.model_name'
        self.__cursor.execute(query, (task_name,))

        models = []
        data = self.__cursor.fetchall()
        for model_id, model_name, name_id, class_name, constructor_args, dataset_args, batch_size, dataset_name in data:
            model = NewsModel(
                model_id=model_id,
                fullname=model_name,
                class_name=class_name,
                name_identifier=name_id,
                constructor_args=pickle.loads(constructor_args),
                dataset_args=pickle.loads(dataset_args),
                batch_size=batch_size,
                dataset_name=dataset_name
            )
            models.append(model)

        return models

    def add_new_article(self, article: NewsArticle) -> int:
        query = 'SELECT id FROM news_sites WHERE code = %s'
        self.__cursor.execute(query, (article.news_site.code,))
        site_id = self.__cursor.fetchone()[0]

        query = 'INSERT INTO news_articles VALUES (0, %s, %s, %s, %s, %s, %s)'
        self.__cursor.execute(query, (article.title, article.content, article.article_url,
                                      article.article_date, site_id, article.image_url, article.summaries))
        self.__database.commit()
        return self.__cursor.lastrowid

    def add_article_summaries(self, article_id: int, summaries: Dict[int, str]) -> None:
        query = 'INSERT INTO summaries VALUES (0, %s, %s, %s)'
        for model_id, summary in summaries.items():
            self.__cursor.execute(query, (summary, article_id, model_id))
        self.__database.commit()

    def add_article_names_entities(self, article_id: int, named_entities: Dict[int, List[NamedEntity]]) -> None:
        query = 'INSERT INTO article_tag_map VALUES (0, %s, %s, %s, %s, %s, %s)'
        for model_id, entities in named_entities.items():
            dataset_name = self.__get_model_dataset_name(model_id)
            tag_category_dict = self.__get_tag_category_dict(dataset_name)
            for entity in entities:
                self.__cursor.execute(query,
                                      (tag_category_dict[entity.full_name], model_id, article_id, entity.position,
                                       entity.length, entity.words))
        self.__database.commit()

    def add_new_model(self, model: NewsModel) -> None:
        query = 'INSERT INTO news_models VALUES (0, %s, %s, %s, %s, %s)'
        params = (
            self.__get_model_id_or_add_new(model),
            self.__get_dataset_id(model.dataset_name),
            pickle.dumps(model.constructor_args),
            pickle.dumps(model.dataset_args),
            model.batch_size
        )
        self.__cursor.execute(query, params)
        self.__database.commit()

    def __get_model_id_or_add_new(self, model: NewsModel) -> int:
        query = 'SELECT id from models WHERE model_name = %s AND model_identifier = %s AND class_name = %s'
        self.__cursor.execute(query, (model.fullname, model.name_identifier, model.class_name))
        result = self.__cursor.fetchone()
        if result is None:
            query = 'INSERT INTO models VALUES (0, %s, %s, %s)'
            self.__cursor.execute(query, (model.fullname, model.name_identifier, model.class_name))
            self.__database.commit()
            return self.__cursor.lastrowid
        else:
            return result[0]

    def get_tag_count(self, dataset_name: str) -> int:
        query = 'SELECT COUNT(*) FROM tags t INNER JOIN tag_categories tc ON t.category_id = tc.id INNER JOIN ' \
                'datasets d on tc.dataset_id = d.id WHERE id_name = %s'
        self.__cursor.execute(query, (dataset_name,))

        return self.__cursor.fetchone()[0]

    def get_tags_dict(self, dataset_name: str) -> Dict[int, Tuple[str, str]]:
        query = 'SELECT tag_label, tag, category_name FROM tags INNER JOIN tag_categories tc ON ' \
                'tags.category_id = tc.id INNER JOIN datasets d on tc.dataset_id = d.id WHERE d.id_name = %s'
        self.__cursor.execute(query, (dataset_name,))

        return {tag_label: (tag, category_name) for tag_label, tag, category_name in self.__cursor.fetchall()}

    def __get_model_dataset_name(self, model_id: int) -> str:
        query = 'SELECT id_name FROM datasets INNER JOIN news_models nm on datasets.id = nm.dataset_id WHERE nm.id = %s'
        self.__cursor.execute(query, (model_id,))

        return self.__cursor.fetchone()[0]

    def __get_tag_category_dict(self, dataset_name: str) -> Dict[str, int]:
        query = 'SELECT category_name, tc.id FROM tag_categories tc INNER JOIN datasets d ON tc.dataset_id = d.id ' \
                'WHERE d.id_name = %s'
        self.__cursor.execute(query, (dataset_name,))

        return dict(self.__cursor.fetchall())

    def __get_dataset_id(self, dataset_name: str) -> int:
        query = 'SELECT id FROM datasets WHERE id_name = %s'
        self.__cursor.execute(query, (dataset_name,))

        return self.__cursor.fetchone()[0]

    def get_articles(self, model_id: int = None, tags: Tuple[str, ...] = ()) -> List[NewsArticle]:
        if model_id is None:
            query = 'SELECT * FROM news_articles'
            self.__cursor.execute(query)

        else:
            tags_str = ','.join(f'"{tag}"' for tag in tags)
            query = f'SELECT DISTINCT na.id, title, content, article_url, article_date, site_id, image_url ' \
                    f'FROM news_articles na INNER JOIN article_tag_map atm ON na.id = atm.article_id WHERE ' \
                    f'atm.model_id = %s AND words IN ({tags_str})'
            self.__cursor.execute(query, (model_id,))

        articles = []
        for article_id, title, content, url, date, site_id, image_url in self.__cursor.fetchall():
            news_site = self.__get_news_site(site_id)
            article = NewsArticle(article_id, title, content, url, date, news_site, image_url)
            articles.append(article)

        return articles

    def get_single_article(self, article_id: int) -> NewsArticle:
        query = 'SELECT * FROM news_articles WHERE id = %s'
        self.__cursor.execute(query, (article_id,))
        article_id, title, content, url, date, site_id, image_url = self.__cursor.fetchone()
        news_site = self.__get_news_site(site_id)

        return NewsArticle(article_id, title, content, url, date, news_site, image_url)

    def get_articles_summaries(self, model_id: int) -> Dict[int, str]:
        query = 'SELECT article_id, content FROM summaries WHERE model_id = %s'
        self.__cursor.execute(query, (model_id,))

        return {article_id: summary for article_id, summary in self.__cursor.fetchall()}

    def get_articles_named_entities(self, model_id: int) -> Dict[int, List[NamedEntity]]:
        query = 'SELECT article_id, category_name, short_name, position, length, words FROM article_tag_map map ' \
                'INNER JOIN tag_categories tc ON map.tag_category_id = tc.id WHERE model_id = %s'
        named_entities = defaultdict(list)
        self.__cursor.execute(query, (model_id,))
        for article_id, category_name, short_name, position, length, words in self.__cursor.fetchall():
            named_entity = NamedEntity(category_name, short_name, position, length, words)
            named_entities[article_id].append(named_entity)

        return named_entities

    def get_articles_top_named_entities(self, model_id: int, top: int = 10) -> Dict[int, List[Tuple[str, str]]]:
        named_entities = self.get_articles_named_entities(model_id)
        top_named_entities = {}
        for article_id, entities in named_entities.items():
            counter = Counter((entity.words, entity.short_name) for entity in entities)
            top_named_entities[article_id] = [entity for entity, _ in counter.most_common(top)]

        return top_named_entities

    def get_articles_tags_count(self, model_id: int, top: int = 10) -> Dict[str, Tuple[str, List[Tuple[str, int]]]]:
        query = 'SELECT DISTINCT article_id, words FROM article_tag_map map INNER JOIN tag_categories tc ON ' \
                'map.tag_category_id = tc.id WHERE category_name = %s AND model_id = %s'
        categories_name = self.__get_tag_categories(model_id)
        tag_counts = {}
        for category, short_name in categories_name:
            self.__cursor.execute(query, (category, model_id))
            words = [word for _, word in self.__cursor.fetchall()]
            tag_counts[short_name] = (category, Counter(words).most_common(top))

        return tag_counts

    def __get_tag_categories(self, model_id: int) -> List[Tuple[str, str]]:
        query = 'SELECT category_name, short_name FROM tag_categories tc INNER JOIN datasets d ON ' \
                'tc.dataset_id = d.id INNER JOIN news_models nm on d.id = nm.dataset_id WHERE nm.id = %s ' \
                'ORDER BY short_name'

        self.__cursor.execute(query, (model_id,))
        return self.__cursor.fetchall()

    def __get_news_site(self, site_id: int) -> NewsSite:
        query = 'SELECT news_sites.name, news_sites.code FROM news_sites INNER JOIN countries ON ' \
                'news_sites.country_id = countries.id WHERE news_sites.id = %s'
        self.__cursor.execute(query, (site_id,))
        name, code = self.__cursor.fetchone()

        return NewsSite(name, code, Country('United States', 'us', 'en'))
