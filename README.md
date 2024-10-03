# Web Crawler Project

This project is a versatile web crawler with various functionalities, including basic crawling, screenshot capture, content extraction, and more. It uses Flask to provide a web interface for interacting with the crawler.

## Project Structure

- `web_crawler.py`: Main entry point of the application
- `app.py`: Flask application and routes
- `crawler.py`: WebCrawler class with all crawling functionalities
- `models.py`: Pydantic models for data structures
- `utils.py`: Utility functions for URL extraction and crawling

## Setup

1. Ensure you have Python 3.7+ installed.
2. Install the required packages:
   ```
   pip install -r requirements.txt
   ```
3. Set up the necessary environment variables:
   - `OPENAI_API_KEY`: Your OpenAI API key
   - `GROQ_API_KEY`: Your Groq API key

## Running the Application

1. Run the following command in the project directory:
   ```
   python web_crawler.py
   ```
2. Open a web browser and navigate to `http://localhost:5000` to access the web interface.

## Features

- Basic crawling
- Screenshot capture
- Chunked crawling
- Structured data extraction
- LLM-based extraction
- Advanced crawling with JavaScript execution
- Custom session crawling
- Page summarization
- Multi-page summarization
- Research assistant functionality

## API Endpoints

- `/`: Home page
- `/crawl` (POST): Endpoint for all crawling operations

## Contributing

Feel free to submit issues or pull requests if you have any improvements or bug fixes.

## License

[MIT License](LICENSE)