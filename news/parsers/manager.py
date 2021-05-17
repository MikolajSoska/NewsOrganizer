import importlib.util
from typing import Dict

from database import DatabaseConnector
from news.parsers.base import BaseNewsParser


class ParsersManager:
    def __init__(self):
        self.__parsers_dict = self.__create_parsers_dict()

    @staticmethod
    def __create_parsers_dict() -> Dict[str, Dict[str, BaseNewsParser]]:
        connector = DatabaseConnector()
        parsers_dict = {}
        for country in connector.get_countries():
            country_dict = {}
            for site in connector.get_news_sites(country.code):
                package_name = site.code.replace('-', '_')
                parser_path = f'news.parsers.{country.code}.{package_name}'
                parser_name = site.name.replace(' ', '')
                parser = getattr(importlib.import_module(parser_path), f'{parser_name}Parser')
                country_dict[site.code] = parser()
            parsers_dict[country.code] = country_dict

        return parsers_dict
