import re
from typing import List, Optional

import bs4
import requests


class CNBCParser:
    def parse_article_content(self, url: str) -> Optional[List[str]]:
        article_site = self.__get_article_site(url)
        if article_site is None:
            return None

        article = []
        content = article_site.find('div', attrs={'id': re.compile('^.*RegularArticle-ArticleBody-5.*$')})
        for group in content.find_all('div', attrs={'class': 'group'}):
            for text in group.find_all('p'):
                article.append(text.text)

        return article

    @staticmethod
    def __get_article_site(url: str) -> Optional[bs4.BeautifulSoup]:
        response = requests.get(url)
        if response.status_code == requests.codes.ok:
            return bs4.BeautifulSoup(response.content, 'html.parser')
        else:
            print(f'HTTP {response.status_code} code when accessing article to parse. Error message: {response.text}.'
                  f' Article URL: {url}')
            return None
