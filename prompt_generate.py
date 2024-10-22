import os
import pandas as pd
from dotenv import load_dotenv
from langchain_community.document_loaders import WebBaseLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import DocArrayInMemorySearch
from giskard.rag import KnowledgeBase, generate_testset

class DocumentProcessor:
    def __init__(self, pdf_path, openai_api_key=None):
        self.pdf_path = pdf_path
        self.openai_api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        self.model = "gpt-3.5-turbo"
        self.documents = []
        self.vectorstore = None
        self.knowledge_base = None
        self.testset = None

        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=20)

    def load_pdf(self):
        """Load PDF and split it into documents."""
        loader = PyPDFLoader(self.pdf_path)
        self.documents = loader.load_and_split(self.text_splitter)
        print(f"Loaded {len(self.documents)} documents from {self.pdf_path}.")

    def create_vectorstore(self):
        """Create a vector store from the loaded documents."""
        self.vectorstore = DocArrayInMemorySearch.from_documents(
            self.documents, embedding=OpenAIEmbeddings()
        )

    def generate_testset(self, num_questions=15, description="A chatbot answering questions about the Legal Payment of Taxes"):
        """Generate a test set based on the knowledge base."""
        self.knowledge_base = KnowledgeBase(pd.DataFrame([d.page_content for d in self.documents], columns=["text"]))
        self.testset = generate_testset(
            self.knowledge_base,
            num_questions=num_questions,
            agent_description=description
        )
        print(f"Generated testset with {num_questions} questions.")

    def save_outputs(self):
        """Save testset and output DataFrame to files, including IDs."""
        self.testset.save("testset.jsonl")
        
        # Convert the testset to a DataFrame and include the ID
        test_set_df = self.testset.to_pandas()
        # Include 'id' in the DataFrame, assuming your testset contains it
        test_set_df['id'] = test_set_df.index  # Adding index as ID if not already present
        
        test_set_df.to_csv('out.csv', index=False)
        print("Saved testset and output DataFrame to 'testset.jsonl' and 'out.csv'.")

    def display_questions(self, num_questions=3):
        """Print the specified number of questions and answers."""
        test_set_df = self.testset.to_pandas()
        for index, row in enumerate(test_set_df.head(num_questions).iterrows()):
            print(f"Question {index + 1}: {row[1]['question']}")
            print(f"Reference answer: {row[1]['reference_answer']}")
            print("Reference context:")
            print(row[1]['reference_context'])
            print("******************", end="\n\n")

# Usage example
if __name__ == "__main__":
    load_dotenv()  # Load environment variables
    processor = DocumentProcessor(pdf_path="GSTSGCH13.pdf", openai_api_key=os.getenv("OPENAI_API_KEY"))
    
    processor.load_pdf()
    processor.create_vectorstore()
    processor.generate_testset(num_questions=1)
    processor.save_outputs()
    processor.display_questions(num_questions=3)
