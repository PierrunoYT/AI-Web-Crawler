import asyncio
from flask import Flask
from app import create_app
from crawler import WebCrawler

async def main():
    crawler = WebCrawler()
    app = create_app(crawler)
    
    # Run a sample crawl
    result = await crawler.advanced_configurable_crawl("https://example.com")
    print("Sample crawl result:", result)
    
    # Run the Flask app
    app.run(debug=True)

if __name__ == "__main__":
    asyncio.run(main())
