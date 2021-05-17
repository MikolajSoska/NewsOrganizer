from typing import List

import bs4

from news.parsers.base import BaseNewsParser


class CNNParser(BaseNewsParser):
    def _parse_article_content(self, content: bs4.BeautifulSoup) -> List[str]:
        article = []
        for paragraph in content.find_all('div', attrs={'class': 'zn-body__paragraph'}):
            if paragraph.text:
                article.append(paragraph.text)

        return article
