import importlib.util
from typing import List, Dict, Optional

from news.parsers.base import BaseNewsParser
from utils.database import DatabaseConnector


class ParsersManager:
    def __init__(self):
        self.__parsers_dict = self.__create_parsers_dict()

    def get_news_sites(self, country: str) -> List[str]:
        return list(self.__parsers_dict[country].keys())

    def get_article_content(self, country: str, news_site: str, article_url: str) -> Optional[List[str]]:
        parser = self.__parsers_dict[country][news_site]
        return parser.get_article_content(article_url)

    @staticmethod
    def __create_parsers_dict() -> Dict[str, Dict[str, BaseNewsParser]]:
        connector = DatabaseConnector()
        parsers_dict = {}
        for country in connector.get_countries():
            country_dict = {}
            for site in connector.get_news_sites(country):
                package_name = site.code.replace('-', '_')
                parser_path = f'news.parsers.{country.code}.{package_name}'
                parser_name = site.name.replace(' ', '')
                parser = getattr(importlib.import_module(parser_path), f'{parser_name}Parser')
                country_dict[site.code] = parser()
            parsers_dict[country.code] = country_dict

        return parsers_dict
