from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Optional


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
    content: str  # TODO: change back to list with mechanism to restore division into paragraphs
    article_url: str
    article_date: datetime
    news_site: NewsSite
    image_url: str
    summary: Optional[str] = None
    named_entities: Dict[int, str] = field(default_factory=lambda: defaultdict(lambda: 'O'))
