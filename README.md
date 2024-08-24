1. **Install Dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

2. **Set up Google API Key:**
   - Obtain a Google API key and set it in the `.env` file.

   ```bash
   GOOGLE_API_KEY=your_api_key_here
   ```

3. **Run the Application:**

   ```bash
   streamlit run app.py
   ```

5. **Upload PDFs:**
   - Upload PDF files.
   - Click on "Submit & Process" to extract text and generate embeddings.

6. **Chat Interface:**
   - Chat with the AI in the main interface.

## Project Structure

- `app.py`: Main application script.
- `.env`: file which will contain your environment variable.
- `requirements.txt`: Python packages required for working of the app.
- `README.md`: Project documentation.

## Dependencies

- PyPDF2
- langchain
- Streamlit
- google.generativeai
- dotenv
