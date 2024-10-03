# Crawl4AI: Advanced Web Crawling and Data Extraction Library

Crawl4AI is a powerful and versatile Python library for web crawling, data extraction, and content analysis. It provides a wide range of features to handle various web scraping scenarios, from basic crawling to advanced session-based extractions and AI-powered content analysis.

## Features

- **Basic Crawling**: Simple URL fetching and content extraction
- **Screenshot Capture**: Capture screenshots of web pages
- **Chunked Crawling**: Split content into manageable chunks
- **Structured Data Extraction**: Extract data using CSS selectors and JSON schemas
- **LLM-based Extraction**: Leverage Language Models for intelligent content extraction
- **Advanced JavaScript Handling**: Execute custom JavaScript for dynamic content
- **Session-based Crawling**: Maintain state across multiple requests
- **Page Summarization**: Generate concise summaries of web pages
- **Multi-page Analysis**: Crawl and analyze multiple pages concurrently
- **Research Assistant**: AI-powered content analysis and question answering
- **Proxy Rotation**: Rotate through a list of proxies for distributed crawling
- **Custom Extraction Strategies**: Implement your own extraction logic
- **Flexible Content Chunking**: Various strategies for content segmentation

## Installation

```bash
pip install crawl4ai
```

## Quick Start

```python
from crawl4ai import WebCrawler

async def main():
    crawler = WebCrawler()
    result = await crawler.basic_crawl("https://example.com")
    print(result)

import asyncio
asyncio.run(main())
```

## Advanced Usage

### Structured Data Extraction

```python
schema = {
    "name": "Product Catalog",
    "baseSelector": "div.product",
    "fields": [
        {"name": "name", "selector": "h2.product-name", "type": "text"},
        {"name": "price", "selector": "span.price", "type": "text"},
        {"name": "description", "selector": "p.description", "type": "text"}
    ]
}

async def extract_products():
    crawler = WebCrawler()
    products = await crawler.extract_with_json_css_strategy("https://example.com/products", schema)
    print(products)

asyncio.run(extract_products())
```

### AI-Powered Content Analysis

```python
async def analyze_content():
    crawler = WebCrawler()
    summary = await crawler.summarize("https://example.com/article")
    print(summary)

asyncio.run(analyze_content())
```

## Configuration

Crawl4AI can be configured using environment variables:

- `OPENAI_API_KEY`: Your OpenAI API key for LLM-based features
- `GROQ_API_KEY`: Your Groq API key for alternative LLM provider

## Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for more details.

## License

Crawl4AI is released under the [MIT License](LICENSE).

## Documentation

For full documentation, visit [https://crawl4ai.readthedocs.io](https://crawl4ai.readthedocs.io).

## Support

For questions and support, please open an issue on our [GitHub repository](https://github.com/yourusername/crawl4ai/issues).
