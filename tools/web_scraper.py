"""
tools/web_scraper.py
--------------------
Autonomous web scraping tool for the Trend Agent.

Uses crawl4ai (Playwright-based) to browse trend sources with zero API keys.
No Reddit developer account. No Twitter API. Just a headless browser.

Sources scraped:
  1. Hypebeast         — https://hypebeast.com/sneakers
  2. Reddit r/Sneakers — https://old.reddit.com/r/Sneakers/hot/
  3. Highsnobiety      — https://www.highsnobiety.com/tag/sneakers/
  4. Sneaker News      — https://sneakernews.com (RSS fallback)

All raw content is cached in ChromaDB (trend_archive collection) so the
system degrades gracefully when a site is unreachable or blocks the crawler.
"""

import os
import hashlib
import asyncio
from datetime import datetime
from typing import Optional

import feedparser
from dotenv import load_dotenv

load_dotenv()

CACHE_TTL_HOURS = 6

# ── Target URLs ───────────────────────────────────────────────────────────────
SCRAPE_TARGETS = [
    {
        "name": "hypebeast_sneakers",
        "url": "https://hypebeast.com/sneakers",
        "css_selector": "article, .post-title, h2, h3",
        "max_items": 20,
    },
    {
        "name": "reddit_sneakers",
        "url": "https://old.reddit.com/r/Sneakers/hot/.json?limit=25",
        "css_selector": None,   # JSON endpoint — parsed directly
        "max_items": 25,
        "is_json": True,
    },
    {
        "name": "reddit_streetwear",
        "url": "https://old.reddit.com/r/streetwear/hot/.json?limit=15",
        "css_selector": None,
        "max_items": 15,
        "is_json": True,
    },
    {
        "name": "highsnobiety",
        "url": "https://www.highsnobiety.com/tag/sneakers/",
        "css_selector": "article h2, article h3, .article-title",
        "max_items": 15,
    },
]

HYPEBEAST_RSS = "https://hypebeast.com/feed"
SNEAKERNEWS_RSS = "https://sneakernews.com/feed/"


def _cache_key(source: str) -> str:
    return hashlib.md5(source.encode()).hexdigest()[:12]


# ── crawl4ai async scraper ────────────────────────────────────────────────────

async def _scrape_url_async(url: str, css_selector: Optional[str] = None) -> str:
    """
    Use crawl4ai to fetch and extract clean text from a URL.
    Returns markdown-formatted content.
    """
    try:
        from crawl4ai import AsyncWebCrawler, CrawlerRunConfig, CacheMode
        from crawl4ai.extraction_strategy import CSSExtractionStrategy
        import json

        config = CrawlerRunConfig(
            cache_mode=CacheMode.BYPASS,
            page_timeout=15000,
            wait_until="domcontentloaded",
            verbose=False,
        )

        async with AsyncWebCrawler(headless=True, verbose=False) as crawler:
            result = await crawler.arun(url=url, config=config)

            if not result.success:
                return ""

            # Return clean markdown text (crawl4ai converts HTML->markdown)
            return result.markdown or result.cleaned_html or ""

    except Exception as e:
        print(f"[web_scraper] crawl4ai failed for {url}: {e}")
        return ""


def _scrape_url(url: str, css_selector: Optional[str] = None) -> str:
    """Sync wrapper around the async scraper."""
    try:
        return asyncio.run(_scrape_url_async(url, css_selector))
    except RuntimeError:
        # Event loop already running (e.g. in Jupyter/Streamlit)
        import nest_asyncio
        nest_asyncio.apply()
        loop = asyncio.get_event_loop()
        return loop.run_until_complete(_scrape_url_async(url, css_selector))
    except Exception as e:
        print(f"[web_scraper] Sync wrapper error: {e}")
        return ""


def _scrape_reddit_json(url: str, max_items: int = 25) -> list[dict]:
    """
    Fetch Reddit's JSON API directly (no auth needed for public subs).
    old.reddit.com/.json endpoints are public and don't require OAuth.
    """
    import httpx

    headers = {
        "User-Agent": "Mozilla/5.0 (compatible; ASBOS-TrendAgent/1.0)",
        "Accept": "application/json",
    }

    try:
        with httpx.Client(timeout=10, headers=headers, follow_redirects=True) as client:
            resp = client.get(url)
            resp.raise_for_status()
            data = resp.json()

        posts = data.get("data", {}).get("children", [])
        items = []
        for post in posts[:max_items]:
            p = post.get("data", {})
            title = p.get("title", "")
            score = p.get("score", 0)
            subreddit = p.get("subreddit", "")
            flair = p.get("link_flair_text", "")
            if title:
                items.append({
                    "title": title,
                    "score": score,
                    "subreddit": subreddit,
                    "flair": flair,
                })
        return items

    except Exception as e:
        print(f"[web_scraper] Reddit JSON fetch failed: {e}")
        return []


# ── Hypebeast RSS (fast, reliable fallback) ───────────────────────────────────

def _fetch_rss(url: str, max_items: int = 15) -> list[str]:
    """Parse an RSS feed and return title+summary strings."""
    try:
        import httpx
        with httpx.Client(timeout=10, follow_redirects=True) as client:
            resp = client.get(url)
            resp.raise_for_status()
            feed_content = resp.text
            
        feed = feedparser.parse(feed_content)
        items = []
        for entry in feed.entries[:max_items]:
            title = entry.get("title", "")
            summary = entry.get("summary", "")[:200]
            if title:
                items.append(f"{title}. {summary}".strip())
        return items
    except Exception as e:
        print(f"[web_scraper] RSS fetch failed for {url}: {e}")
        return []


# ── Main public functions ─────────────────────────────────────────────────────

def fetch_hypebeast_trends(max_items: int = 15) -> list[dict]:
    """
    Scrape Hypebeast for sneaker trend content.

    Strategy:
      1. Try crawl4ai headless browser scrape
      2. Fall back to RSS feed
      3. Fall back to ChromaDB cache
    """
    from tools.rag_retriever import retrieve, upsert

    items = []
    doc_ids, docs, metas = [], [], []

    # Strategy 1: crawl4ai
    print("[web_scraper] Scraping Hypebeast via crawl4ai...")
    content = _scrape_url("https://hypebeast.com/sneakers")

    if content and len(content) > 200:
        # Extract headline-like lines from the markdown
        lines = [
            line.strip("# *").strip()
            for line in content.split("\n")
            if len(line.strip()) > 20 and len(line.strip()) < 200
            and not line.startswith("http")
            and not line.startswith("![")
        ]
        for line in lines[:max_items]:
            items.append({"title": line, "summary": "", "published": str(datetime.now())})
            cid = _cache_key(f"hypebeast_{line}")
            doc_ids.append(cid)
            docs.append(line)
            metas.append({"source": "hypebeast_crawl", "fetched_at": str(datetime.now())})

    # Strategy 2: RSS fallback
    if not items:
        print("[web_scraper] crawl4ai empty — trying Hypebeast RSS...")
        rss_items = _fetch_rss(HYPEBEAST_RSS, max_items)
        for text in rss_items:
            items.append({"title": text[:100], "summary": text, "published": str(datetime.now())})
            cid = _cache_key(f"hbrss_{text[:40]}")
            doc_ids.append(cid)
            docs.append(text)
            metas.append({"source": "hypebeast_rss", "fetched_at": str(datetime.now())})

    # Strategy 3: cache fallback
    if not items:
        print("[web_scraper] All Hypebeast sources failed — using cache.")
        cached = retrieve(
            "sneaker trend aesthetic style",
            collection="trend_archive",
            n=max_items,
            where={"source": "hypebeast_crawl"},
        )
        return [{"title": r["document"][:100], "summary": r["document"], "published": ""} for r in cached]

    # Cache new results
    if doc_ids:
        upsert("trend_archive", ids=doc_ids, documents=docs, metadatas=metas)

    print(f"[web_scraper] Hypebeast: {len(items)} items collected.")
    return items[:max_items]


def fetch_reddit_trends(max_posts: int = 30) -> list[dict]:
    """
    Fetch trending posts from sneaker/fashion subreddits.

    Strategy:
      1. Reddit public JSON API (no auth, no PRAW, no API key)
      2. crawl4ai browser scrape of old.reddit.com
      3. ChromaDB cache fallback
    """
    from tools.rag_retriever import retrieve, upsert

    all_items = []
    doc_ids, docs, metas = [], [], []

    subreddits = [
        ("Sneakers", 20),
        ("streetwear", 15),
        ("malefashionadvice", 10),
    ]

    # Strategy 1: Reddit public JSON API
    print("[web_scraper] Fetching Reddit via public JSON API...")
    for subreddit, limit in subreddits:
        url = f"https://old.reddit.com/r/{subreddit}/hot/.json?limit={limit}"
        posts = _scrape_reddit_json(url, limit)
        for post in posts:
            all_items.append(post)
            cid = _cache_key(f"reddit_{subreddit}_{post['title'][:30]}")
            doc_text = f"[r/{post['subreddit']}] {post['title']}"
            if post.get("flair"):
                doc_text += f" [{post['flair']}]"
            doc_ids.append(cid)
            docs.append(doc_text)
            metas.append({
                "source": "reddit_json",
                "subreddit": subreddit,
                "score": str(post.get("score", 0)),
                "fetched_at": str(datetime.now()),
            })

    # Strategy 2: crawl4ai browser scrape (if JSON returned nothing)
    if not all_items:
        print("[web_scraper] Reddit JSON empty — trying crawl4ai browser scrape...")
        content = _scrape_url("https://old.reddit.com/r/Sneakers/hot/")
        if content and len(content) > 200:
            lines = [
                line.strip("# *").strip()
                for line in content.split("\n")
                if 20 < len(line.strip()) < 250
                and not line.startswith("http")
            ]
            for line in lines[:max_posts]:
                all_items.append({"title": line, "score": 0, "subreddit": "Sneakers", "flair": ""})
                cid = _cache_key(f"reddit_crawl_{line[:30]}")
                doc_ids.append(cid)
                docs.append(f"[r/Sneakers] {line}")
                metas.append({"source": "reddit_crawl", "subreddit": "Sneakers",
                              "score": "0", "fetched_at": str(datetime.now())})

    # Strategy 3: cache fallback
    if not all_items:
        print("[web_scraper] All Reddit sources failed — using cache.")
        return _get_reddit_cache()

    # Cache new results
    if doc_ids:
        upsert("trend_archive", ids=doc_ids, documents=docs, metadatas=metas)

    # Sort by score (highest engagement first)
    all_items.sort(key=lambda x: x.get("score", 0), reverse=True)
    print(f"[web_scraper] Reddit: {len(all_items)} posts collected.")
    return all_items[:max_posts]


def fetch_highsnobiety_trends(max_items: int = 10) -> list[dict]:
    """
    Scrape Highsnobiety for premium sneaker trend signals.
    Uses crawl4ai — no API needed.
    """
    from tools.rag_retriever import retrieve, upsert

    print("[web_scraper] Scraping Highsnobiety via crawl4ai...")
    content = _scrape_url("https://www.highsnobiety.com/tag/sneakers/")

    items = []
    if content and len(content) > 100:
        lines = [
            line.strip("# *").strip()
            for line in content.split("\n")
            if 20 < len(line.strip()) < 200
            and not line.startswith("http")
            and not line.startswith("![")
        ]
        for line in lines[:max_items]:
            items.append({"title": line, "source": "highsnobiety"})

        if items:
            upsert(
                "trend_archive",
                ids=[_cache_key(f"highsnob_{i['title'][:30]}") for i in items],
                documents=[i["title"] for i in items],
                metadatas=[{"source": "highsnobiety", "fetched_at": str(datetime.now())} for _ in items],
            )

    if not items:
        # Fallback to cache
        cached = retrieve("sneaker premium luxury trend", collection="trend_archive",
                          n=max_items, where={"source": "highsnobiety"})
        items = [{"title": r["document"][:100], "source": "highsnobiety_cache"} for r in cached]

    print(f"[web_scraper] Highsnobiety: {len(items)} items collected.")
    return items


def _get_reddit_cache() -> list[dict]:
    """Return cached Reddit posts from ChromaDB."""
    from tools.rag_retriever import retrieve
    cached = retrieve("sneaker hype trending drop", collection="trend_archive", n=15)
    return [{"title": r["document"], "score": 0, "subreddit": "cached", "flair": ""} for r in cached]


def compile_trend_summary(
    hypebeast_items: list[dict],
    reddit_items: list[dict],
    highsnobiety_items: Optional[list[dict]] = None,
) -> str:
    """
    Merge all scraped trend data into a single prompt-ready summary
    for Gemini to analyse and extract aesthetic keywords from.
    """
    hb_block = "\n".join(
        f"- {item.get('title', '')}"
        for item in hypebeast_items[:12]
        if item.get('title')
    ) or "No Hypebeast data available."

    reddit_block = "\n".join(
        f"- [r/{item.get('subreddit', '?')}] {item.get('title', '')} "
        f"(upvotes: {item.get('score', 0)})"
        for item in sorted(reddit_items, key=lambda x: x.get("score", 0), reverse=True)[:12]
        if item.get('title')
    ) or "No Reddit data available."

    highsnob_block = ""
    if highsnobiety_items:
        highsnob_block = "\n\nHIGHSNOBIETY HEADLINES:\n" + "\n".join(
            f"- {item.get('title', '')}"
            for item in highsnobiety_items[:8]
            if item.get('title')
        )

    return (
        f"HYPEBEAST SNEAKER HEADLINES:\n{hb_block}\n\n"
        f"REDDIT HOT POSTS:\n{reddit_block}"
        f"{highsnob_block}"
    )

