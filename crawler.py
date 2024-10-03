import asyncio
import base64
import json
import csv
import time
import os
from typing import List, Dict, Any, Optional, Callable
from crawl4ai import AsyncWebCrawler
from crawl4ai.extraction_strategy import JsonCssExtractionStrategy, LLMExtractionStrategy, CosineStrategy, NoExtractionStrategy
from typing import Dict, Any, Optional
import json
from crawl4ai.chunking_strategy import (
    RegexChunking,
    NlpSentenceChunking,
    TopicSegmentationChunking,
    FixedLengthWordChunking,
    SlidingWindowChunking
)
from crawl4ai.async_crawler_strategy import AsyncPlaywrightCrawlerStrategy
from playwright.async_api import Page, Browser
from openai import AsyncOpenAI
from tenacity import retry, stop_after_attempt, wait_exponential
from aiohttp import ClientSession
from models import PageSummary
from utils import extract_urls, crawl_urls

class WebCrawler:
    """
    A comprehensive web crawling and data extraction class that utilizes AsyncWebCrawler.
    
    This class provides various methods for crawling web pages, extracting structured data,
    summarizing content, and performing research tasks. It supports advanced features such as
    custom JavaScript execution, pagination handling, and integration with language models
    for content analysis and summarization.
    """

    def __init__(self, proxy_list: Optional[List[str]] = None):
        crawler_strategy = AsyncPlaywrightCrawlerStrategy(verbose=True)
        crawler_strategy.set_hook('on_browser_created', self.on_browser_created)
        crawler_strategy.set_hook('before_goto', self.before_goto)
        crawler_strategy.set_hook('after_goto', self.after_goto)
        crawler_strategy.set_hook('on_execution_started', self.on_execution_started)
        crawler_strategy.set_hook('before_return_html', self.before_return_html)
        self.crawler = AsyncWebCrawler(verbose=True, crawler_strategy=crawler_strategy)
        self.proxy_list = proxy_list or []
        
        self.summarization_strategy = LLMExtractionStrategy(
            provider="openai/gpt-4o",
            api_token=os.getenv('OPENAI_API_KEY'),
            schema=PageSummary.model_json_schema(),
            extraction_type="schema",
            apply_chunking=False,
            instruction=(
                "From the crawled content, extract the following details: "
                "1. Title of the page "
                "2. Summary of the page, which is a detailed summary "
                "3. Brief summary of the page, which is a paragraph text "
                "4. Keywords assigned to the page, which is a list of keywords. "
                'The extracted JSON format should look like this: '
                '{ "title": "Page Title", "summary": "Detailed summary of the page.", '
                '"brief_summary": "Brief summary in a paragraph.", "keywords": ["keyword1", "keyword2", "keyword3"] }'
            )
        )
        
        self.client = AsyncOpenAI(base_url="https://api.groq.com/openai/v1", api_key=os.getenv("GROQ_API_KEY"))

    def get_default_llm_settings(self) -> Dict[str, Any]:
        return {
            "model": "llama3-8b-8192",
            "temperature": 0.5,
            "max_tokens": 500,
            "top_p": 1,
            "frequency_penalty": 0,
            "presence_penalty": 0,
        }

    async def on_browser_created(self, browser: Browser):
        print("[HOOK] on_browser_created")
        context = await browser.new_context(viewport={'width': 1920, 'height': 1080})
        page = await context.new_page()
        await context.add_cookies([{'name': 'test_cookie', 'value': 'cookie_value', 'url': 'https://example.com'}])
        await page.close()
        await context.close()

    async def before_goto(self, page: Page):
        print("[HOOK] before_goto")
        await page.set_extra_http_headers({'X-Test-Header': 'test'})

    async def after_goto(self, page: Page):
        print("[HOOK] after_goto")
        print(f"Current URL: {page.url}")

    async def on_execution_started(self, page: Page):
        print("[HOOK] on_execution_started")
        await page.evaluate("console.log('Custom JS executed')")

    async def before_return_html(self, page: Page, html: str):
        print("[HOOK] before_return_html")
        print(f"HTML length: {len(html)}")
        return page

    def set_custom_hooks(self, hooks: Dict[str, Callable[..., Any]]):
        for hook_name, hook_function in hooks.items():
            self.crawler.crawler_strategy.set_hook(hook_name, hook_function)

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    async def basic_crawl(self, url: str) -> str:
        extraction_strategy = NoExtractionStrategy()
        chunking_strategy = RegexChunking()
        async with self.crawler as crawler:
            result = await crawler.arun(
                url=url,
                word_count_threshold=10,
                extraction_strategy=extraction_strategy,
                chunking_strategy=chunking_strategy,
                bypass_cache=False,
                css_selector=None,
                screenshot=False,
                user_agent=None,
                verbose=True,
                only_text=False
            )
            if result.success:
                return result.markdown[:500]  # Return first 500 characters
            else:
                return f"Error: {result.error_message}"

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    async def take_screenshot(self, url: str) -> str:
        async with self.crawler as crawler:
            result = await crawler.arun(url=url, screenshot=True)
            return base64.b64encode(result.screenshot).decode('utf-8')

    async def save_screenshot(self, url: str, filename: str = "screenshot.png") -> None:
        screenshot_base64 = await self.take_screenshot(url)
        with open(filename, "wb") as f:
            f.write(base64.b64decode(screenshot_base64))
        print(f"Screenshot saved to '{filename}'!")

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    async def chunked_crawl(self, url: str) -> str:
        async with self.crawler as crawler:
            result = await crawler.arun(
                url=url,
                chunking_strategy=RegexChunking(patterns=["\n\n"])
            )
            return result.extracted_content[:200]

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    async def extract_structured_data(self, url: str, schema: Dict[str, Any]) -> Dict[str, Any]:
        extraction_strategy = JsonCssExtractionStrategy(schema, verbose=True)

        async with self.crawler as crawler:
            result = await crawler.arun(
                url=url,
                extraction_strategy=extraction_strategy,
                bypass_cache=True
            )
            
            if not result.success:
                return {"error": "Failed to extract data"}

            extracted_data = json.loads(result.extracted_content)
            return extracted_data[:10]  # Return first 10 items

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    async def extract_with_llm(self, url: str, instruction: str) -> Dict[str, Any]:
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            return {"error": "OPENAI_API_KEY environment variable is not set"}
        
        extraction_strategy = LLMExtractionStrategy(
            provider="openai/gpt-4o",
            api_token=api_key,
            instruction=instruction
        )

        async with self.crawler as crawler:
            result = await crawler.arun(
                url=url,
                extraction_strategy=extraction_strategy,
                bypass_cache=True
            )

            if not result.success:
                return {"error": "Failed to extract data"}

            extracted_data = json.loads(result.extracted_content)
            return extracted_data

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    async def advanced_crawl(self, url: str) -> Dict[str, Any]:
        js_code = """
        const loadMoreButton = Array.from(document.querySelectorAll('button')).find(button => button.textContent.includes('Load More'));
        if (loadMoreButton) {
            loadMoreButton.click();
            // Wait for new content to load
            await new Promise(resolve => setTimeout(resolve, 2000));
        }
        """

        wait_for = """
        () => {
            const articles = document.querySelectorAll('article.tease-card');
            return articles.length > 10;
        }
        """

        async with self.crawler as crawler:
            result = await crawler.arun(
                url=url,
                js_code=js_code,
                wait_for=wait_for,
                css_selector="article.tease-card",
                extraction_strategy=CosineStrategy(
                    semantic_filter="technology",
                ),
                chunking_strategy=RegexChunking(),
            )

        return {
            "extracted_content": result.extracted_content,
            "html_length": len(result.html) if result.html else 0,
            "url": result.url,
            "media": result.media,
            "links": result.links
        }

    async def extract_media_and_links(self, url: str) -> Dict[str, Any]:
        async with self.crawler as crawler:
            result = await crawler.arun(url=url)
        return {
            "media": result.media,
            "links": result.links
        }

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    async def advanced_configurable_crawl(self, url: str, **kwargs) -> Dict[str, Any]:
        default_settings = {
            "word_count_threshold": 10,
            "extraction_strategy": NoExtractionStrategy(),
            "chunking_strategy": RegexChunking(),
            "bypass_cache": False,
            "css_selector": None,
            "screenshot": False,
            "user_agent": None,
            "verbose": True,
            "only_text": False,
            "js_code": None,
            "wait_for": None,
            "session_id": None
        }

        # Update default settings with provided kwargs
        settings = {**default_settings, **kwargs}

        async with self.crawler as crawler:
            try:
                result = await crawler.arun(
                    url=url,
                    **settings
                )
                if result.success:
                    return {
                        "extracted_content": result.extracted_content,
                        "html_length": len(result.html) if result.html else 0,
                        "url": result.url,
                        "screenshot": base64.b64encode(result.screenshot).decode('utf-8') if result.screenshot else None
                    }
                else:
                    return {"error": f"Crawl failed: {result.error_message}"}
            except Exception as e:
                return {"error": f"An error occurred: {str(e)}"}

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    async def custom_session_crawl(self, url: str) -> Dict[str, Any]:
        js_code = """
        const loadMoreButton = Array.from(document.querySelectorAll('button')).find(button => button.textContent.includes('Load More'));
        if (loadMoreButton) {
            loadMoreButton.click();
            // Wait for new content to load
            await new Promise(resolve => setTimeout(resolve, 2000));
        }
        """

        wait_for = """
        () => {
            const articles = document.querySelectorAll('article.tease-card');
            return articles.length > 10;
        }
        """

        async with self.crawler as crawler:
            result1 = await crawler.arun(
                url=url,
                js_code=js_code,
                wait_for=wait_for,
                css_selector="article.tease-card",
                session_id="business_session"
            )

            result2 = await crawler.crawler_strategy.execute_js(
                session_id="business_session",
                js_code="window.scrollTo(0, document.body.scrollHeight);",
                wait_for_js="() => window.innerHeight + window.scrollY >= document.body.offsetHeight"
            )

        return {
            "initial_crawl": result1.extracted_content,
            "additional_js": result2.html
        }

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    async def summarize(self, url: str) -> Dict[str, Any]:
        try:
            async with self.crawler as crawler:
                result = await crawler.arun(
                    url=url,
                    word_count_threshold=1,
                    extraction_strategy=self.summarization_strategy,
                    chunking_strategy=RegexChunking(),
                    bypass_cache=True,
                    verbose=True,
                    only_text=True,
                    wait_for="body"
                )

            if result.success:
                page_summary = json.loads(result.extracted_content)
                return page_summary
            else:
                return {"error": f"Failed to summarize the page. Error: {result.error_message}"}
        except json.JSONDecodeError:
            return {"error": "Failed to parse the extracted content as JSON"}
        except Exception as e:
            return {"error": f"An unexpected error occurred: {str(e)}"}

    # ... (rest of the code remains unchanged)

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    async def summarize_multiple(self, urls: List[str], use_proxy: bool = False) -> List[Dict[str, Any]]:
        async with self.crawler as crawler:
            tasks = [crawler.arun(
                url=url,
                word_count_threshold=1,
                extraction_strategy=self.summarization_strategy,
                chunking_strategy=RegexChunking(),
                bypass_cache=True
            ) for url in urls
            ]
            results = await asyncio.gather(*tasks)

        summaries = []
        for i, result in enumerate(results):
            if result.success:
                page_summary = json.loads(result.extracted_content)
                summaries.append(page_summary)
            else:
                summaries.append({"error": f"Failed to summarize URL {i+1}. Error: {result.error_message}"})

        return summaries

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    async def crawl_multiple(self, urls: List[str], **kwargs) -> List[Dict[str, Any]]:
        """
        Crawl multiple URLs concurrently using the same configuration.
        
        :param urls: List of URLs to crawl
        """
        async with self.crawler as crawler:
            tasks = [crawler.arun(url=url, **kwargs) for url in urls]
            results = await asyncio.gather(*tasks)

        return [
            {
                "url": result.url,
                "success": result.success,
                "extracted_content": result.extracted_content if result.success else None,
                "error": result.error_message if not result.success else None
            }
            for result in results
        ]

    async def clear_cache(self) -> None:
        """Clear the crawler's cache."""
        async with self.crawler as crawler:
            await crawler.aclear_cache()

    async def flush_cache(self) -> None:
        """Completely flush the crawler's cache."""
        async with self.crawler as crawler:
            await crawler.aflush_cache()

    async def get_cache_size(self) -> int:
        """Get the current size of the cache."""
        async with self.crawler as crawler:
            return await crawler.aget_cache_size()

    async def paginated_crawl(self, url: str, max_pages: int = 5, scroll_delay: int = 2, **kwargs) -> List[str]:
        """
        Perform a paginated crawl by scrolling and waiting for new content.
        
        :param url: URL to crawl
        :param max_pages: Maximum number of pages to crawl
        :param scroll_delay: Delay in seconds between scrolls
        :param kwargs: Additional parameters for the crawler
        """
        if 'js_code' in kwargs or 'wait_for' in kwargs:
            print("Warning: 'js_code' and 'wait_for' parameters will be overwritten for pagination handling.")
        
        kwargs['bypass_cache'] = kwargs.get('bypass_cache', True)  # Ensure fresh content for each page
        results = []
        async with self.crawler as crawler:
            for page in range(max_pages):
                result = await crawler.arun(url=url, **kwargs)
                if result.success:
                    results.append(result.extracted_content)
                else:
                    break

                # Scroll to bottom and wait for new content
                js_code = "window.scrollTo(0, document.body.scrollHeight);"
                await crawler.crawler_strategy.execute_js(
                    js_code=js_code,
                    wait_for_js="() => window.innerHeight + window.scrollY >= document.body.offsetHeight"
                )
                await asyncio.sleep(scroll_delay)

        return results

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    async def research_assistant(self, user_message: str, context: Dict[str, Any], llm_settings: Optional[Dict[str, Any]] = None) -> str:
        """
        Perform research based on user message, crawl URLs, and generate a response using an LLM.
        
        :param user_message: User's input message
        :param context: Dictionary to store context information
        :param llm_settings: Custom LLM settings (optional)
        """
        settings = llm_settings or self.get_default_llm_settings()
        urls = extract_urls(user_message)
        
        if urls:
            crawled_contents = await crawl_urls(urls)
            for url, content in zip(urls, crawled_contents):
                ref_number = f"REF_{len(context) + 1}"
                context[ref_number] = {
                    "url": url,
                    "content": content
                }

        context_messages = [
            f'<appendix ref="{ref}">\n{data["content"]}\n</appendix>'
            for ref, data in context.items()
        ]
        system_message = {
            "role": "system",
            "content": (
                "You are a helpful bot. Use the following context for answering questions. "
                "Refer to the sources using the REF number in square brackets, e.g., [1], only if the source is given in the appendices below.\n\n"
                "If the question requires any information from the provided appendices or context, refer to the sources. "
                "If not, there is no need to add a references section. "
                "At the end of your response, provide a reference section listing the URLs and their REF numbers only if sources from the appendices were used.\n\n"
                "\n\n".join(context_messages)
            ) if context_messages else "You are a helpful assistant."
        }

        messages = [system_message, {"role": "user", "content": user_message}]

        stream = await self.client.chat.completions.create(
            messages=messages,
            stream=True,
            **settings
        )

        assistant_response = ""
        async for part in stream:
            if token := part.choices[0].delta.content:
                assistant_response += token

        if context:
            reference_section = "\n\nReferences:\n"
            for ref, data in context.items():
                reference_section += f"[{ref.split('_')[1]}]: {data['url']}\n"
            assistant_response += reference_section

        return assistant_response

    async def handle_dynamic_content(self, url: str, scroll_interval: int = 2, max_scrolls: int = 10, **kwargs) -> Optional[str]:
        """
        Handle dynamic content loading (e.g., infinite scrolling) more generally.
        
        :param url: URL to crawl
        :param scroll_interval: Time to wait between scrolls (in seconds)
        :param max_scrolls: Maximum number of scrolls to perform
        :param kwargs: Additional parameters for the crawler
        """
        try:
            async with self.crawler as crawler:
                result = await crawler.arun(url=url, **kwargs)
                if not result.success:
                    return None
                initial_content = result.extracted_content

                for _ in range(max_scrolls):
                    js_code = "window.scrollTo(0, document.body.scrollHeight);"
                    await crawler.crawler_strategy.execute_js(
                        js_code=js_code,
                        wait_for_js="() => window.innerHeight + window.scrollY >= document.body.offsetHeight"
                    )
                    await asyncio.sleep(scroll_interval)

                    new_result = await crawler.arun(url=url, **kwargs)
                    if not new_result.success or new_result.extracted_content == initial_content:
                        break
                    initial_content = new_result.extracted_content
        except Exception as e:
            print(f"Error handling dynamic content: {str(e)}")
            return None
        return initial_content

    async def export_data(self, data: List[Dict[str, Any]], export_format: str = 'json', filename: str = 'crawled_data') -> str:
        """
        Export crawled data in various formats (JSON or CSV).
        
        :param data: List of dictionaries containing crawled data
        :param export_format: Format to export data ('json' or 'csv')
        :param filename: Name of the file to save the exported data
        """
        if export_format == 'json':
            with open(f"{filename}.json", 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            return f"Data exported to {filename}.json"
        elif export_format == 'csv':
            keys = set().union(*(d.keys() for d in data))
            with open(f"{filename}.csv", 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=keys)
                writer.writeheader()
                writer.writerows(data)
            return f"Data exported to {filename}.csv"
        else:
            return "Invalid export format. Use 'json' or 'csv'."

    def get_next_proxy(self) -> Optional[str]:
        return self.proxy_list.pop(0) if self.proxy_list else None

    async def rotate_proxy(self, url: str, **kwargs) -> Dict[str, Any]:
        """
        Rotate through available proxies when crawling a URL.
        
        :param url: URL to crawl
        :param kwargs: Additional parameters for the crawler
        """
        while True:
            proxy = self.get_next_proxy()
            if not proxy:
                return {"error": "No more proxies available"}
            
            kwargs['proxy'] = proxy
            result = await self.advanced_configurable_crawl(url, **kwargs)
            if result.get('error') is None:
                return result

    async def basic_session_crawl(self, url: str, num_pages: int = 3, load_more_selector: str = '.load-more-button', content_selector: str = '.content-item') -> List[Dict[str, Any]]:
        """
        Perform a basic session-based crawl on a dynamic website.
        
        :param url: URL to crawl
        :param num_pages: Number of pages to crawl
        :param load_more_selector: CSS selector for the 'Load More' button
        :param content_selector: CSS selector for the content items
        :return: List of extracted content items
        """
        session_id = f"session_{int(time.time())}"
        all_items = []

        async with self.crawler as crawler:
            for page in range(num_pages):
                result = await crawler.arun(
                    url=url,
                    session_id=session_id,
                    js_code=f"document.querySelector('{load_more_selector}').click();" if page > 0 else None,
                    css_selector=content_selector,
                    bypass_cache=True
                )

                if result.success:
                    items = result.extracted_content.split(content_selector)
                    all_items.extend(items[1:])  # Skip the first empty item
                    print(f"Page {page + 1}: Found {len(items) - 1} items")
                else:
                    print(f"Error on page {page + 1}: {result.error_message}")
                    break

            await crawler.crawler_strategy.kill_session(session_id)

        return all_items

    async def extract_with_cosine_strategy(self, url: str, semantic_filter: Optional[str] = None, word_count_threshold: int = 20, max_dist: float = 0.2, linkage_method: str = 'ward', top_k: int = 3, model_name: str = 'BAAI/bge-small-en-v1.5') -> str:
        """
        Extract content using CosineStrategy.

        :param url: URL to crawl
        :param semantic_filter: Keywords for filtering relevant documents before clustering
        :param word_count_threshold: Minimum number of words per cluster
        :param max_dist: Maximum cophenetic distance on the dendrogram to form clusters
        :param linkage_method: Linkage method for hierarchical clustering
        :param top_k: Number of top categories to extract
        :param model_name: Model name for embedding generation
        :return: Extracted content
        """
        strategy = CosineStrategy(
            semantic_filter=semantic_filter,
            word_count_threshold=word_count_threshold,
            max_dist=max_dist,
            linkage_method=linkage_method,
            top_k=top_k,
            model_name=model_name
        )

        async with self.crawler as crawler:
            result = await crawler.arun(url=url, extraction_strategy=strategy)
            return result.extracted_content if result.success else f"Error: {result.error_message}"

    async def extract_with_llm_strategy(self, url: str, provider: str = 'openai', api_token: Optional[str] = None, instruction: Optional[str] = None) -> str:
        """
        Extract content using LLMExtractionStrategy.

        :param url: URL to crawl
        :param provider: Provider for language model completions
        :param api_token: API token for the provider
        :param instruction: Instructions to guide the LLM on how to perform the extraction
        :return: Extracted content
        """
        strategy = LLMExtractionStrategy(
            provider=provider,
            api_token=api_token or os.getenv('OPENAI_API_KEY'),
            instruction=instruction
        )

        async with self.crawler as crawler:
            result = await crawler.arun(url=url, extraction_strategy=strategy)
            return result.extracted_content if result.success else f"Error: {result.error_message}"

    async def extract_with_json_css_strategy(self, url: str, schema: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract content using JsonCssExtractionStrategy.

        :param url: URL to crawl
        :param schema: A dictionary defining the extraction schema
        :return: Extracted structured data
        """
        strategy = JsonCssExtractionStrategy(schema, verbose=True)

        async with self.crawler as crawler:
            result = await crawler.arun(url=url, extraction_strategy=strategy)
            
            if not result.success:
                return {"error": f"Failed to crawl the page: {result.error_message}"}

            try:
                extracted_data = json.loads(result.extracted_content)
                return extracted_data
            except json.JSONDecodeError:
                return {"error": "Failed to parse the extracted content as JSON"}

    async def chunk_content(self, url: str, chunking_strategy: str = 'regex', **kwargs) -> List[str]:
        """
        Crawl a URL and chunk the content using the specified chunking strategy.

        :param url: URL to crawl
        :param chunking_strategy: Strategy to use for chunking ('regex', 'nlp', 'topic', 'fixed', 'sliding')
        :param kwargs: Additional parameters for the chunking strategy
        :return: List of content chunks
        """
        async with self.crawler as crawler:
            result = await crawler.arun(url=url, bypass_cache=True)

            if not result.success:
                return [f"Error: {result.error_message}"]

            content = result.extracted_content or result.html

            if chunking_strategy == 'regex':
                patterns = kwargs.get('patterns', ['\n\n'])
                chunker = RegexChunking(patterns=patterns)
            elif chunking_strategy == 'nlp':
                chunker = NlpSentenceChunking()
            elif chunking_strategy == 'topic':
                num_keywords = kwargs.get('num_keywords', 3)
                chunker = TopicSegmentationChunking(num_keywords=num_keywords)
            elif chunking_strategy == 'fixed':
                chunk_size = kwargs.get('chunk_size', 100)
                chunker = FixedLengthWordChunking(chunk_size=chunk_size)
            elif chunking_strategy == 'sliding':
                window_size = kwargs.get('window_size', 100)
                step = kwargs.get('step', 50)
                chunker = SlidingWindowChunking(window_size=window_size, step=step)
            else:
                return ["Error: Invalid chunking strategy"]

            chunks = chunker.chunk(content)
            return chunks

    async def extract_complex_product_data(self, url: str) -> Dict[str, Any]:
        """
        Extract complex product data using JsonCssExtractionStrategy.

        :param url: URL to crawl
        :return: Extracted product data
        """
        """
        Extract complex product data using JsonCssExtractionStrategy.

        :param url: URL to crawl
        :return: Extracted product data
        """
        schema = {
            "name": "E-commerce Product Catalog",
            "baseSelector": "div.category",
            "fields": [
                {
                    "name": "category_name",
                    "selector": "h2.category-name",
                    "type": "text"
                },
                {
                    "name": "products",
                    "selector": "div.product",
                    "type": "nested_list",
                    "fields": [
                        {
                            "name": "name",
                            "selector": "h3.product-name",
                            "type": "text"
                        },
                        {
                            "name": "price",
                            "selector": "p.product-price",
                            "type": "text"
                        },
                        {
                            "name": "details",
                            "selector": "div.product-details",
                            "type": "nested",
                            "fields": [
                                {
                                    "name": "brand",
                                    "selector": "span.brand",
                                    "type": "text"
                                },
                                {
                                    "name": "model",
                                    "selector": "span.model",
                                    "type": "text"
                                }
                            ]
                        },
                        {
                            "name": "features",
                            "selector": "ul.product-features li",
                            "type": "list",
                            "fields": [
                                {
                                    "name": "feature",
                                    "type": "text"
                                }
                            ]
                        },
                        {
                            "name": "reviews",
                            "selector": "div.review",
                            "type": "nested_list",
                            "fields": [
                                {
                                    "name": "reviewer",
                                    "selector": "span.reviewer",
                                    "type": "text"
                                },
                                {
                                    "name": "rating",
                                    "selector": "span.rating",
                                    "type": "text"
                                },
                                {
                                    "name": "comment",
                                    "selector": "p.review-text",
                                    "type": "text"
                                }
                            ]
                        },
                        {
                            "name": "related_products",
                            "selector": "ul.related-products li",
                            "type": "list",
                            "fields": [
                                {
                                    "name": "name",
                                    "selector": "span.related-name",
                                    "type": "text"
                                },
                                {
                                    "name": "price",
                                    "selector": "span.related-price",
                                    "type": "text"
                                }
                            ]
                        }
                    ]
                }
            ]
        }

        extraction_strategy = JsonCssExtractionStrategy(schema, verbose=True)

        async with self.crawler as crawler:
            result = await crawler.arun(
                url=url,
                extraction_strategy=extraction_strategy,
                bypass_cache=True,
            )

            if not result.success:
                return {"error": f"Failed to crawl the page: {result.error_message}"}

            try:
                product_data = json.loads(result.extracted_content)
                return product_data
            except json.JSONDecodeError:
                return {"error": "Failed to parse the extracted content as JSON"}

    async def advanced_session_crawl_with_hooks(self, url: str, num_pages: int = 3, content_selector: str = 'li.commit-item', next_page_selector: str = 'a.pagination-next') -> List[Dict[str, Any]]:
        """
        Perform an advanced session-based crawl with custom execution hooks.
        
        :param url: URL to crawl
        :param num_pages: Number of pages to crawl
        :param content_selector: CSS selector for the content items
        :param next_page_selector: CSS selector for the 'Next Page' button
        :return: List of extracted content items
        """
        first_item = ""

        async def on_execution_started(page):
            nonlocal first_item
            try:
                while True:
                    await page.wait_for_selector(content_selector)
                    item = await page.query_selector(f"{content_selector} h4")
                    item_text = await item.evaluate("(element) => element.textContent")
                    item_text = item_text.strip()
                    if item_text and item_text != first_item:
                        first_item = item_text
                        break
                    await asyncio.sleep(0.5)
            except Exception as e:
                print(f"Warning: New content didn't appear after JavaScript execution: {e}")

        async with self.crawler as crawler:
            crawler.crawler_strategy.set_hook("on_execution_started", on_execution_started)

            session_id = f"session_{int(time.time())}"
            all_items = []

            js_next_page = f"""
            const button = document.querySelector('{next_page_selector}');
            if (button) button.click();
            """

            for page in range(num_pages):
                result = await crawler.arun(
                    url=url,
                    session_id=session_id,
                    css_selector=content_selector,
                    js_code=js_next_page if page > 0 else None,
                    bypass_cache=True,
                    js_only=page > 0
                )

                if result.success:
                    items = result.extracted_content.split(content_selector)
                    all_items.extend(items[1:])  # Skip the first empty item
                    print(f"Page {page + 1}: Found {len(items) - 1} items")
                else:
                    print(f"Error on page {page + 1}: {result.error_message}")
                    break

            await crawler.crawler_strategy.kill_session(session_id)

        return all_items

    async def wait_for_parameter_crawl(self, url: str, num_pages: int = 3, content_selector: str = 'li.commit-item', next_page_selector: str = 'a.pagination-next') -> List[Dict[str, Any]]:
        """
        Perform a session-based crawl using the wait_for parameter.
        
        :param url: URL to crawl
        :param num_pages: Number of pages to crawl
        :param content_selector: CSS selector for the content items
        :param next_page_selector: CSS selector for the 'Next Page' button
        :return: List of extracted content items
        """
        async with self.crawler as crawler:
            session_id = f"session_{int(time.time())}"
            all_items = []

            js_next_page = f"""
            const items = document.querySelectorAll('{content_selector} h4');
            if (items.length > 0) {{
                window.lastItem = items[0].textContent.trim();
            }}
            const button = document.querySelector('{next_page_selector}');
            if (button) button.click();
            """

            wait_for = f"""() => {{
                const items = document.querySelectorAll('{content_selector} h4');
                if (items.length === 0) return false;
                const firstItem = items[0].textContent.trim();
                return firstItem !== window.lastItem;
            }}"""

            for page in range(num_pages):
                result = await crawler.arun(
                    url=url,
                    session_id=session_id,
                    css_selector=content_selector,
                    js_code=js_next_page if page > 0 else None,
                    wait_for=wait_for if page > 0 else None,
                    js_only=page > 0,
                    bypass_cache=True
                )

                if result.success:
                    items = result.extracted_content.split(content_selector)
                    all_items.extend(items[1:])  # Skip the first empty item
                    print(f"Page {page + 1}: Found {len(items) - 1} items")
                else:
                    print(f"Error on page {page + 1}: {result.error_message}")
                    break

            await crawler.crawler_strategy.kill_session(session_id)

        return all_items
