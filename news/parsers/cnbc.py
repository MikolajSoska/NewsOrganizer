import re
from typing import List, Optional

from news.parsers.base import BaseNewsParser


class CNBCParser(BaseNewsParser):
    def parse_article_content(self, url: str) -> Optional[List[str]]:
        article_site = self._get_article_site(url)
        if article_site is None:
            return None

        article = []
        content = article_site.find('div', attrs={'id': re.compile('^.*RegularArticle-ArticleBody-5.*$')})
        for group in content.find_all('div', attrs={'class': 'group'}):
            for text in group.find_all('p'):
                article.append(text.text)

        return article
