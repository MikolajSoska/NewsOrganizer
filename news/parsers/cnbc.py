import re
from typing import List

import bs4

from news.parsers.base import BaseNewsParser


class CNBCParser(BaseNewsParser):
    def _parse_article_content(self, content: bs4.BeautifulSoup) -> List[str]:
        article = []
        article_body = content.find('div', attrs={'id': re.compile('^.*RegularArticle-ArticleBody-5.*$')})
        for group in article_body.find_all('div', attrs={'class': 'group'}):
            for paragraph in group.find_all('p'):
                if paragraph.text:
                    article.append(paragraph.text)

        return article
