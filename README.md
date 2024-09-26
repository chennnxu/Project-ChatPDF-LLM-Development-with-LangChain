# PDF Question Answering with LangChain and DeepSeek AI

This project demonstrates how to build a question-answering system for PDF documents using LangChain, DeepSeek AI, and Baichuan embeddings. The system can read a PDF file, process its content, and answer questions based on the document's information.

## Features

- PDF document loading and processing
- Text splitting for efficient processing
- Vector store creation using Chroma and Baichuan embeddings
- Question answering using DeepSeek AI language model
- Retrieval-augmented generation (RAG) for improved answer accuracy

## Prerequisites

- Python 3.9+
- DeepSeek API key
- Baichuan API key

## Installation

1. Clone this repository:
   ```
   git clone https://github.com/chennnxu/Project-QnAsystem-LLM-Langchain.git
   cd pdf-qa-langchain-deepseek
   ```

2. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

3. Create a `.env` file in the project root and add your API keys:
   ```
   DEEPSEEK_API_KEY=your_deepseek_api_key_here
   BAICHUAN_AUTH_TOKEN=your_baichuan_api_key_here
   ```

## Usage

1. Place your PDF file in the `book/` directory.

2. Run the Jupyter notebook `QnA-deepseek.ipynb` or `deepseek-qa.py` to process the PDF and set up the question-answering system.

3. Use the `rag_chain.invoke()` function to ask questions about the PDF content:
   ```python
   results = rag_chain.invoke({"input": "Your question here?"})
   print(results["answer"])
   ```

## How it Works

1. The PDF is loaded and split into smaller chunks.
2. Baichuan embeddings are used to create vector representations of the text chunks.
3. The embeddings are stored in a Chroma vector store for efficient retrieval.
4. When a question is asked, relevant text chunks are retrieved from the vector store.
5. The DeepSeek AI model generates an answer based on the retrieved context and the question.

## Customization

- Adjust the `chunk_size` and `chunk_overlap` parameters in the text splitter to optimize for your specific PDF content.
- Modify the system prompt in the `ChatPromptTemplate` to change the behavior of the question-answering system.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [LangChain](https://python.langchain.com/) for the document processing and RAG implementation
- [DeepSeek AI](https://deepseek.com/) for the language model
- [Baichuan](https://www.baichuan-ai.com/) for text embeddings
