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

   ### Setting Environment Variables

   #### Windows
   Using Command Prompt:
   ```
   set OPENAI_API_KEY=your_openai_api_key_here
   set GROQ_API_KEY=your_groq_api_key_here
   ```
   Or to set them permanently:
   ```
   setx OPENAI_API_KEY your_openai_api_key_here
   setx GROQ_API_KEY your_groq_api_key_here
   ```
   Note: After using `setx`, you'll need to restart your command prompt for the changes to take effect.

   #### macOS and Linux
   In Terminal:
   ```
   export OPENAI_API_KEY=your_openai_api_key_here
   export GROQ_API_KEY=your_groq_api_key_here
   ```
   To make these permanent, add these lines to your `~/.bashrc`, `~/.zshrc`, or equivalent shell configuration file.

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

## Handling of Extracted Content

The extracted content from web crawling operations is processed in-memory and is not automatically saved to a file or database. Instead, it is returned as part of the JSON response to API requests. The handling of the extracted content varies depending on the type of crawl:

1. Basic and chunked crawls return a portion of the extracted content.
2. Structured data extraction returns a list of extracted items.
3. LLM-based extraction and summarization return processed content as a Python dictionary.
4. Advanced and custom session crawls return extracted content along with additional metadata.
5. Multi-page summarization returns a list of summaries for multiple URLs.
6. The research assistant functionality uses the extracted content to generate a response, which is then returned.

If persistent storage of the extracted content is required, you would need to implement additional functionality to save the results to a file or database after receiving the API response.

## Contributing

Feel free to submit issues or pull requests if you have any improvements or bug fixes.

## License

[MIT License](LICENSE)