import re
from crawl4ai import AsyncWebCrawler
from crawl4ai.extraction_strategy import NoExtractionStrategy
from crawl4ai.chunking_strategy import RegexChunking

def extract_urls(text):
    url_pattern = re.compile(r'(https?://\S+)')
    return url_pattern.findall(text)

async def crawl_urls(urls):
    async with AsyncWebCrawler(verbose=True) as crawler:
        results = await crawler.arun_many(
            urls=urls,
            word_count_threshold=10,
            extraction_strategy=NoExtractionStrategy(),
            chunking_strategy=RegexChunking(),
            bypass_cache=True
        )
        return [result.markdown for result in results if result.success]