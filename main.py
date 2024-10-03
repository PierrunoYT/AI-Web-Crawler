from crawl4ai import WebCrawler
import asyncio

async def main():
    crawler = WebCrawler()
    result = await crawler.basic_crawl("https://example.com")
    print(result)

if __name__ == "__main__":
    asyncio.run(main())
