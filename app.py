import asyncio
import json
from flask import Flask, render_template, request, jsonify
from crawler import WebCrawler

app = Flask(__name__)
crawler = WebCrawler()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/crawl', methods=['POST'])
def crawl():
    url = request.form['url']
    crawl_type = request.form['type']

    if crawl_type == 'basic':
        result = asyncio.run(crawler.basic_crawl(url))
    elif crawl_type == 'screenshot':
        result = asyncio.run(crawler.take_screenshot(url))
    elif crawl_type == 'chunked':
        result = asyncio.run(crawler.chunked_crawl(url))
    elif crawl_type == 'structured':
        schema = {
            "name": "Generic Article Extractor",
            "baseSelector": "article",
            "fields": [
                {"name": "title", "selector": "h1,h2,h3", "type": "text"},
                {"name": "content", "selector": "p", "type": "text"},
                {"name": "date", "selector": "time", "type": "text"},
                {"name": "author", "selector": ".author", "type": "text"},
            ]
        }
        result = asyncio.run(crawler.extract_structured_data(url, schema))
    elif crawl_type == 'llm':
        instruction = "Extract and summarize the main points of the content"
        result = asyncio.run(crawler.extract_with_llm(url, instruction))
    elif crawl_type == 'advanced':
        result = asyncio.run(crawler.advanced_crawl(url))
    elif crawl_type == 'custom_session':
        result = asyncio.run(crawler.custom_session_crawl(url))
    elif crawl_type == 'summarize':
        result = asyncio.run(crawler.summarize(url))
    elif crawl_type == 'summarize_multiple':
        urls = request.form.getlist('urls[]')
        result = asyncio.run(crawler.summarize_multiple(urls))
    elif crawl_type == 'research_assistant':
        user_message = request.form['message']
        context = request.form.get('context', '{}')
        context = json.loads(context)
        result = asyncio.run(crawler.research_assistant(user_message, context))
    else:
        return jsonify({'error': 'Invalid crawl type'})

    return jsonify({'result': result})

if __name__ == '__main__':
    app.run(debug=True)