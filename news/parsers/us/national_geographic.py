from typing import List

import bs4

from news.parsers.base import BaseNewsParser


class NationalGeographicParser(BaseNewsParser):
    def _parse_article_content(self, content: bs4.BeautifulSoup) -> List[str]:
        article = []
        article_body = content.find('section', attrs={'class': 'Article__Content'})
        for paragraph in article_body.find_all('p'):
            if paragraph.text:
                article.append(paragraph.text)

        return article
