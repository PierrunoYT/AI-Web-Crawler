import asyncio
import base64
import json
import os
from crawl4ai import AsyncWebCrawler
from crawl4ai.extraction_strategy import JsonCssExtractionStrategy, LLMExtractionStrategy, CosineStrategy, NoExtractionStrategy
from crawl4ai.chunking_strategy import RegexChunking
from crawl4ai.async_crawler_strategy import AsyncPlaywrightCrawlerStrategy
from playwright.async_api import Page, Browser
from openai import AsyncOpenAI
from models import PageSummary
from utils import extract_urls, crawl_urls

class WebCrawler:
    def __init__(self):
        crawler_strategy = AsyncPlaywrightCrawlerStrategy(verbose=True)
        crawler_strategy.set_hook('on_browser_created', self.on_browser_created)
        crawler_strategy.set_hook('before_goto', self.before_goto)
        crawler_strategy.set_hook('after_goto', self.after_goto)
        crawler_strategy.set_hook('on_execution_started', self.on_execution_started)
        crawler_strategy.set_hook('before_return_html', self.before_return_html)
        self.crawler = AsyncWebCrawler(verbose=True, crawler_strategy=crawler_strategy)
        
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
        
        self.settings = {
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

    async def basic_crawl(self, url):
        async with self.crawler as crawler:
            result = await crawler.arun(url=url)
            return result.markdown[:500]  # Return first 500 characters

    async def take_screenshot(self, url):
        async with self.crawler as crawler:
            result = await crawler.arun(url=url, screenshot=True)
            return base64.b64encode(result.screenshot).decode('utf-8')

    async def chunked_crawl(self, url):
        async with self.crawler as crawler:
            result = await crawler.arun(
                url=url,
                chunking_strategy=RegexChunking(patterns=["\n\n"])
            )
            return result.extracted_content[:200]

    async def extract_structured_data(self, url, schema):
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

    async def extract_with_llm(self, url, instruction):
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

    async def advanced_crawl(self, url):
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
            "url": result.url
        }

    async def custom_session_crawl(self, url):
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

    async def summarize(self, url):
        async with self.crawler as crawler:
            result = await crawler.arun(
                url=url,
                word_count_threshold=1,
                extraction_strategy=self.summarization_strategy,
                chunking_strategy=RegexChunking(),
                bypass_cache=True
            )

        if result.success:
            page_summary = json.loads(result.extracted_content)
            return page_summary
        else:
            return {"error": f"Failed to summarize the page. Error: {result.error_message}"}

    async def summarize_multiple(self, urls):
        async with self.crawler as crawler:
            tasks = [crawler.arun(
                url=url,
                word_count_threshold=1,
                extraction_strategy=self.summarization_strategy,
                chunking_strategy=RegexChunking(),
                bypass_cache=True
            ) for url in urls]
            results = await asyncio.gather(*tasks)

        summaries = []
        for i, result in enumerate(results):
            if result.success:
                page_summary = json.loads(result.extracted_content)
                summaries.append(page_summary)
            else:
                summaries.append({"error": f"Failed to summarize URL {i+1}. Error: {result.error_message}"})

        return summaries

    async def research_assistant(self, user_message, context):
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
            **self.settings
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