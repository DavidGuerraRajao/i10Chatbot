from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings
from langchain_core.documents import Document


def update_database_from_txt(file_path: str, db_path: str = "db_directory"):
    try:
        # Read the text file
        with open(file_path, 'r', encoding='utf-8') as f:
            raw_text = f.read()

        # Create documents
        documents = [Document(page_content=raw_text)]

        # Split text
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        splits = text_splitter.split_documents(documents)

        # Initialize embeddings
        embeddings = OllamaEmbeddings(
            model="deepseek-r1:32b",
            base_url="http://localhost:11434"
        )

        # Try to load existing DB
        try:
            db = FAISS.load_local(
                db_path,
                embeddings,
                allow_dangerous_deserialization=True
            )
            db.add_documents(splits)
            operation = "updated"
        except:
            # Create new DB if none exists
            db = FAISS.from_documents(splits, embeddings)
            operation = "created"

        # Save the database
        db.save_local(db_path)
        return f"Database {operation} com sucesso de {file_path} → {db_path}"

    except FileNotFoundError:
        return f"Erro: File {file_path} não encontrada"
    except Exception as e:
        return f"Erro updating a database: {str(e)}"


# Example usage:
if __name__ == "__main__":
    filepath = input('A onde fica o txt para atualizar a Pesquisa Manual AI do i10Chatbot?: ')
    result = update_database_from_txt(
        file_path=filepath,
        db_path="db_directory"
    )
    print(result)
