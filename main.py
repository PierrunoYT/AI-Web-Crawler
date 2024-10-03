import asyncio
from flask import Flask
from app import create_app
from crawler import WebCrawler

async def main():
    crawler = WebCrawler()
    app = create_app(crawler)
    
    print("\n--- Running Sample Crawl ---")
    # Run a sample crawl
    result = await crawler.advanced_configurable_crawl("https://example.com")
    print("Sample crawl result:")
    print(f"URL: {result['url']}")
    print(f"HTML Length: {result['html_length']}")
    print("Extracted Content:")
    print(result['extracted_content'])
    print("--- Sample Crawl Complete ---\n")
    
    print("Starting Flask application...")
    # Run the Flask app
    app.run(debug=True)

if __name__ == "__main__":
    asyncio.run(main())
