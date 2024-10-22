# Document Processing and Testset Generation using LangChain and OpenAI

This project is designed to process a PDF document, split its content, create embeddings, and generate a test set for a question-answering chatbot. It uses the following tools:
- LangChain (for document loading, text splitting, and embedding generation)
- OpenAI API (for embedding documents)
- Giskard's RAG (for generating test sets from a knowledge base)
- Pandas (for handling document content and test set generation)

## Features

- **Load and split PDF documents**: Load a PDF and split it into chunks of text.
- **Create vectorstore**: Generate embeddings from the text chunks and store them in a searchable vectorstore.
- **Generate test sets**: Automatically create a test set of questions and answers from the document content.
- **Save outputs**: Store the generated test set in JSONL and CSV formats.
- **Display questions**: Print a few sample questions and answers generated from the content.

## Requirements

- Python 3.x
- OpenAI API Key
- Libraries: 
  - langchain-community
  - fillpdf
  - pandas
  - dotenv
  - giskard

## Installation

1. **Clone the repository**:
    ```bash
    git clone https://github.com/your-repo/document-processor.git
    cd document-processor
    ```

2. **Create a virtual environment** (optional but recommended):
    ```bash
    python -m venv venv
    source venv/bin/activate  # For Windows use: venv\Scripts\activate
    ```

3. **Install the required packages**:
    ```bash
    pip install -r requirements.txt
    ```

4. **Set up environment variables**:
    - Create a `.env` file in the root directory with your OpenAI API key:
      ```
      OPENAI_API_KEY=your_openai_api_key_here
      ```

## Usage

1. **Load a PDF and generate a testset**:
   Edit the `pdf_path` in the `DocumentProcessor` initialization to the path of the PDF you want to process.

2. **Run the script**:
    ```bash
    python app.py
    ```

3. **View output**:
    - A CSV (`out.csv`) and a JSONL file (`testset.jsonl`) will be generated with the test set.
    - The questions and answers will be displayed in the console.

4. **Display more questions**:
   Modify the number of questions passed to `processor.display_questions(num_questions)` to display more.

## Customization

- **Change document splitting**: You can adjust the `chunk_size` and `chunk_overlap` in the `RecursiveCharacterTextSplitter` to control how the document is split.
- **Testset generation**: Modify the `description` or `num_questions` to customize the generated test set.
- **Save formats**: The script saves both a CSV and JSONL format of the test set, but you can add or remove formats as needed.

## Troubleshooting

- Ensure that the OpenAI API key
