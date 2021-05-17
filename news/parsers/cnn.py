from typing import List, Optional

from news.parsers.base import BaseNewsParser


class CNNParser(BaseNewsParser):
    def parse_article_content(self, url: str) -> Optional[List[str]]:
        article_site = self._get_article_site(url)
        if article_site is None:
            return None

        article = []
        for paragraph in article_site.find_all('div', attrs={'class': 'zn-body__paragraph'}):
            if paragraph.text:
                article.append(paragraph.text)

        return article
