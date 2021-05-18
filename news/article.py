from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional


@dataclass(init=True, repr=True, eq=False)
class Country:
    name: str
    code: str
    language: str


@dataclass(init=True, repr=True, eq=False)
class NewsSite:
    name: str
    code: str
    country: Country


@dataclass(init=True, repr=True, eq=False)
class NewsArticle:
    title: str
    content: List[str]
    article_url: str
    article_date: datetime
    news_site: NewsSite
    image_url: str
    summary: Optional[str] = None
