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
2. Set up a virtual environment (see "Virtual Environment Setup" section below).
3. Install the required packages:
   ```
   pip install -r requirements.txt
   ```
4. Set up the necessary environment variables:
   - `OPENAI_API_KEY`: Your OpenAI API key
   - `GROQ_API_KEY`: Your Groq API key

## Virtual Environment Setup

### Windows

1. Open Command Prompt or PowerShell
2. Navigate to your project directory
3. Create a virtual environment:
   ```
   python -m venv venv
   ```
4. Activate the virtual environment:
   ```
   venv\Scripts\activate
   ```

### macOS and Linux

1. Open Terminal
2. Navigate to your project directory
3. Create a virtual environment:
   ```
   python3 -m venv venv
   ```
4. Activate the virtual environment:
   ```
   source venv/bin/activate
   ```

After activating the virtual environment, your command prompt should change to indicate that the virtual environment is active. You can now proceed with installing the required packages.

## Running the Application

1. Ensure your virtual environment is activated.
2. Run the following command in the project directory:
   ```
   python web_crawler.py
   ```
3. Open a web browser and navigate to `http://localhost:5000` to access the web interface.

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