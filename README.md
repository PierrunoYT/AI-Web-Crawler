# AIWebCrawler: Advanced Web Crawling and Data Extraction Library

**Note: This project is based on [https://github.com/unclecode/crawl4ai](https://github.com/unclecode/crawl4ai). It is currently not fully functional and should be used as a template or starting point for your own implementation.**

AIWebCrawler is a powerful and versatile Python library for web crawling, data extraction, and content analysis. It provides a wide range of features to handle various web scraping scenarios, from basic crawling to advanced session-based extractions and AI-powered content analysis.

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
- **Complex Product Data Extraction**: Extract detailed product information from e-commerce sites
- **Advanced Session Crawling**: Perform sophisticated multi-page crawls with custom hooks
- **Wait-for Parameter Crawling**: Efficiently crawl dynamic content with customizable wait conditions

## Installation

1. Create a virtual environment:
   ```bash
   python -m venv aiwebcrawler_env
   ```

2. Activate the virtual environment:
   - On Windows:
     ```bash
     aiwebcrawler_env\Scripts\activate
     ```
   - On macOS and Linux:
     ```bash
     source aiwebcrawler_env/bin/activate
     ```

3. Install requirements:
   ```bash
   pip install -r requirements.txt
   ```

4. Install AIWebCrawler:
   ```bash
   pip install aiwebcrawler[all]
   ```

5. Download required models:
   ```bash
   aiwebcrawler-download-models
   ```

Note: The `requirements.txt` file includes the necessary dependencies for the project. If you prefer to install dependencies manually or want more control over the versions, you can install them individually instead of using the requirements file.

## Quick Start

```python
from aiwebcrawler import WebCrawler

async def main():
    crawler = WebCrawler()
    result = await crawler.basic_crawl("https://example.com")
    print(result)

import asyncio
asyncio.run(main())
```

## Running the Script

To run the script, follow these steps:

1. Make sure you're in the project directory and your virtual environment is activated.

2. Create a new Python file, for example `main.py`, and paste the following code:

   ```python
   from aiwebcrawler import WebCrawler
   import asyncio

   async def main():
       crawler = WebCrawler()
       result = await crawler.basic_crawl("https://example.com")
       print(result)

   if __name__ == "__main__":
       asyncio.run(main())
   ```

3. Run the script using Python:

   ```bash
   python main.py
   ```

   This will execute the basic crawl and print the results.

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

### Complex Product Data Extraction

```python
async def extract_complex_product_data():
    crawler = WebCrawler()
    product_data = await crawler.extract_complex_product_data("https://example.com/product")
    print(product_data)

asyncio.run(extract_complex_product_data())
```

### Advanced Session Crawling

```python
async def advanced_session_crawl():
    crawler = WebCrawler()
    results = await crawler.advanced_session_crawl_with_hooks(
        "https://example.com/page1",
        num_pages=3,
        content_selector="div.content-item",
        next_page_selector="a.next-page"
    )
    print(results)

asyncio.run(advanced_session_crawl())
```

## Configuration

AIWebCrawler can be configured using environment variables:

On Windows, you can set these variables using the `set` or `setx` commands:

```
set OPENAI_API_KEY=your_openai_api_key
set GROQ_API_KEY=your_groq_api_key
```

Or for persistent settings:

```
setx OPENAI_API_KEY your_openai_api_key
setx GROQ_API_KEY your_groq_api_key
```

On macOS and Linux, you can set these variables in your shell configuration file (e.g., `.bashrc`, `.zshrc`):

```
export OPENAI_API_KEY=your_openai_api_key
export GROQ_API_KEY=your_groq_api_key
```

The required environment variables are:

- `OPENAI_API_KEY`: Your OpenAI API key for LLM-based features
- `GROQ_API_KEY`: Your Groq API key for alternative LLM provider

## Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for more details.

## License

AIWebCrawler is released under the [MIT License](LICENSE).

## Documentation

For full documentation, visit [https://aiwebcrawler.readthedocs.io](https://aiwebcrawler.readthedocs.io).

## Support

For questions and support, please open an issue on our [GitHub repository](https://github.com/yourusername/aiwebcrawler/issues).
