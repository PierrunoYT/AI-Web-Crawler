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
- **Complex Product Data Extraction**: Extract detailed product information from e-commerce sites
- **Advanced Session Crawling**: Perform sophisticated multi-page crawls with custom hooks
- **Wait-for Parameter Crawling**: Efficiently crawl dynamic content with customizable wait conditions

## Installation

1. Create a virtual environment:
   ```bash
   python -m venv crawl4ai_env
   ```

2. Activate the virtual environment:
   - On Windows:
     ```bash
     crawl4ai_env\Scripts\activate
     ```
   - On macOS and Linux:
     ```bash
     source crawl4ai_env/bin/activate
     ```

3. Install Crawl4AI:
   ```bash
   pip install crawl4ai[all]
   ```

4. Download required models:
   ```bash
   crawl4ai-download-models
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

Crawl4AI can be configured using environment variables. On Windows, you can set these variables using the `set` or `setx` commands:

```
set OPENAI_API_KEY=your_openai_api_key
set GROQ_API_KEY=your_groq_api_key
```

Or for persistent settings:

```
setx OPENAI_API_KEY your_openai_api_key
setx GROQ_API_KEY your_groq_api_key
```

The required environment variables are:

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

## Development Guidelines

### SEARCH/REPLACE Block Rules

When contributing code changes, please follow these rules for SEARCH/REPLACE blocks:

1. The FULL file path alone on a line, verbatim. No bold asterisks, no quotes around it, no escaping of characters, etc.
2. The opening fence and code language, e.g., <source>python
3. The start of search block: <<<<<<< SEARCH
4. A contiguous chunk of lines to search for in the existing source code
5. The dividing line: =======
6. The lines to replace into the source code
7. The end of the replace block: >>>>>>> REPLACE
8. The closing fence: </source>

- Use the FULL file path, as shown by the maintainer.
- Every SEARCH section must EXACTLY MATCH the existing file content, character for character, including all comments, docstrings, etc.
- SEARCH/REPLACE blocks will replace ALL matching occurrences.
- Include enough lines to make the SEARCH blocks uniquely match the lines to change.
- Keep SEARCH/REPLACE blocks concise.
- Break large SEARCH/REPLACE blocks into a series of smaller blocks that each change a small portion of the file.
- Include just the changing lines, and a few surrounding lines if needed for uniqueness.
- Do not include long runs of unchanging lines in SEARCH/REPLACE blocks.
- Only create SEARCH/REPLACE blocks for files that have been explicitly added to the discussion.
- To move code within a file, use 2 SEARCH/REPLACE blocks: 1 to delete it from its current location, 1 to insert it in the new location.
- Pay attention to which filenames are intended for editing, especially when creating new files.
- For new files, use a SEARCH/REPLACE block with an empty SEARCH section and the new file's contents in the REPLACE section.

Remember: ONLY EVER RETURN CODE IN A SEARCH/REPLACE BLOCK!

For file system operations like renaming files, use appropriate shell commands at the end of your response.
