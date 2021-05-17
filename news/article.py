from dataclasses import dataclass
from datetime import datetime
from typing import List


@dataclass(init=True, repr=True, eq=False)
class NewsArticle:
    title: str
    content: List[str]
    article_url: str
    article_date: datetime
    site_name: str
    image_url: str
    country: str


@dataclass(init=True, repr=True, eq=False)
class NewsSite:
    name: str
    code: str
    country: str


@dataclass(init=True, repr=True, eq=False)
class Country:
    name: str
    code: str
