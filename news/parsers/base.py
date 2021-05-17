import abc
from typing import List, Optional

import bs4
import requests


class BaseNewsParser(abc.ABC):
    def get_article_content(self, url: str) -> Optional[List[str]]:
        article_site = self._get_article_site(url)
        if article_site is None:
            return None

        return self._parse_article_content(article_site)

    @abc.abstractmethod
    def _parse_article_content(self, content: bs4.BeautifulSoup) -> List[str]:
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
