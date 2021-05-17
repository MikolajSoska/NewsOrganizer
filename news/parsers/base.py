import abc
from typing import List, Optional

import bs4
import requests


class BaseNewsParser(abc.ABC):
    @abc.abstractmethod
    def parse_article_content(self, url: str) -> Optional[List[str]]:
        pass

    @staticmethod
    def _get_article_site(url: str) -> Optional[bs4.BeautifulSoup]:
        response = requests.get(url)
        if response.status_code == requests.codes.ok:
            return bs4.BeautifulSoup(response.content, 'html.parser')
        else:
            print(f'HTTP {response.status_code} code when accessing article to parse. Error message: {response.text}.'
                  f' Article URL: {url}')
            return None
