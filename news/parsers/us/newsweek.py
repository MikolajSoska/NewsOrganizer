from typing import List

import bs4

from news.parsers.base import BaseNewsParser


class NewsweekParser(BaseNewsParser):
    def __init__(self):
        super().__init__(headers={'User-Agent': ''})

    def _parse_article_content(self, content: bs4.BeautifulSoup) -> List[str]:
        article = []
        article_body = content.find('div', attrs={'class': 'article-body'})
        for paragraph in article_body.find_all('p'):
            if paragraph.text:
                article.append(paragraph.text)

        return article
