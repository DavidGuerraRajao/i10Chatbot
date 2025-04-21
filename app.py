from flask import Flask, render_template, request, jsonify, redirect, url_for
from langchain_ollama import OllamaEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama.llms import OllamaLLM
from langchain_community.vectorstores import FAISS
import re
import os
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

app = Flask(__name__)

# Initialize the models and templates
model = OllamaLLM(model="deepseek-r1:32b")

# Main template (from Main.py)
main_template = """
Você é um chatbot chamado David que responde perguntas sobre a Biblioteca virtual, i10 Bibliotecas.
Aqui estão algumas partes da documentação para te ajudar a responder ao usuario:
{context}

Aqui está a pergunta do usuário:
{question}

Tome muito cuidado ao responder o usuário para não alucinar, não falar Português ou falar Inglês.
Aliás, se o que o usuário perguntou não está na documentação ou dentro do tema responda com preucaucao, mas mesmo assim você deve fornecer alguma resposta sem exceção!
Se possivel resuma sua resposta em no máximo 3 parágrafos com cada um contendo no máximo 2 frases.
Mas se voce ter que criar uma resposta mais longa fassa.

Responda aqui:
"""

# FAQ template (from model2.py)
faq_template = """
Você é um chatbot chamado David que responde perguntas sobre a Biblioteca virtual, i10 Bibliotecas.
Aqui estão algumas perguntas e respostas frequentes que tem aver com a pergunta do usuario para te ajudar a responder ao usuario:
{context}

Aqui está a pergunta do usuário:
{question}

Tome muito cuidado ao responder o usuário para não alucinar, não falar Português ou falar Inglês.
Aliás, se o que o usuário perguntou não está na documentação ou dentro do tema responda com preucaucao, mas mesmo assim você deve fornecer alguma resposta sem exceção!
Se possivel resuma sua resposta em no máximo 3 parágrafos com cada um contendo no máximo 2 frases.
Mas se voce ter que criar uma resposta mais longa fassa.

Responda aqui:
"""

# File paths
FAQ_FILE = "fileij"
DB_DIRECTORY = "db_directory"


def retrieve_docs(db, query, k=10):
    return db.similarity_search_with_score(query, k)


def question_pdf(question, documents):
    document_texts = [doc[0].page_content for doc in documents]
    context = "\n\n\n".join(document_texts)
    prompt = ChatPromptTemplate.from_template(main_template)
    chain = prompt | model
    response = chain.invoke({"question": question, "context": context})
    return re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL)


def load_faq_documents():
    if os.path.exists(FAQ_FILE):
        with open(FAQ_FILE, 'r', encoding='utf-8') as f:
            content = f.read()
        return [line.strip() for line in content.split('//') if line.strip()]
    return []


def get_faq_response(question):
    dirty_docs = load_faq_documents()

    def cleaner():
        with open(FAQ_FILE, 'r', encoding='utf-8') as file:
            content = file.read()
        pattern = r'Pergunta:\s*(.*?\?)'
        perguntas = re.findall(pattern, content)
        return [pergunta.strip() for pergunta in perguntas]

    cleaned_docs = cleaner()
    if not cleaned_docs:
        return "Desculpe, não há perguntas frequentes disponíveis no momento."

    vectorizer = TfidfVectorizer().fit(cleaned_docs)
    doc_vectors = vectorizer.transform(cleaned_docs)
    query_vector = vectorizer.transform([question])
    similarities = cosine_similarity(query_vector, doc_vectors).flatten()
    sorted_indices = np.argsort(-similarities)

    search_docs = ''
    n = 0
    for idx in sorted_indices:
        n += 1
        if similarities[idx] < 10.0:
            search_docs += f'\n\n{dirty_docs[idx]}\n\n\n'
        if n == 8:
            break

    prompt = ChatPromptTemplate.from_template(faq_template)
    chain = prompt | model
    response = chain.invoke({"question": question, "context": search_docs})
    return re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL)

print('Interface dos Admins: http://localhost:5000/admin')
print('Interface dos Usuarios: http://127.0.0.1:5000')

@app.route('/')
def home():
    return render_template('index.html')


@app.route('/ask', methods=['POST'])
def ask():
    question = request.form['question']
    use_faq = request.form.get('use_faq', 'true') == 'true'

    if use_faq:
        response = get_faq_response(question)
        return jsonify({
            'response': response,
            'method': 'FAQ'
        })
    else:
        embeddings = OllamaEmbeddings(model="qwq:32b")
        db = FAISS.load_local(
            DB_DIRECTORY,
            embeddings,
            allow_dangerous_deserialization=True
        )
        related_documents = retrieve_docs(db, question)
        response = question_pdf(question, related_documents)
        return jsonify({
            'response': response,
            'method': 'Document Search'
        })


# Admin routes
@app.route('/admin')
def admin():
    return render_template('admin.html')


@app.route('/admin/add_faq', methods=['POST'])
def add_faq():
    new_entry = request.form['faq_entry']

    # Validate the format
    if "Pergunta:" not in new_entry or "Resposta:" not in new_entry:
        return jsonify({"success": False, "message": "Formato inválido. Use 'Pergunta: ... Resposta: ...'"})

    # Add to FAQ file
    with open(FAQ_FILE, 'a', encoding='utf-8') as f:
        f.write(f"\n{new_entry}\n//")

    return jsonify({"success": True, "message": "FAQ adicionada com sucesso!"})


@app.route('/admin/add_document', methods=['POST'])
def add_document():
    new_doc = request.form['document_text']

    # Create document
    documents = [Document(page_content=new_doc)]

    # Split text
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    splits = text_splitter.split_documents(documents)

    # Initialize embeddings
    embeddings = OllamaEmbeddings(model="deepseek-r1:32b")

    try:
        # Try to load existing DB
        db = FAISS.load_local(
            DB_DIRECTORY,
            embeddings,
            allow_dangerous_deserialization=True
        )
        # Add new documents to existing DB
        db.add_documents(splits)
    except:
        # If DB doesn't exist, create new
        db = FAISS.from_documents(splits, embeddings)

    # Save the updated DB
    db.save_local(DB_DIRECTORY)

    return jsonify({"success": True, "message": "Documento adicionado ao banco de dados com sucesso!"})


@app.route('/admin/view_faqs')
def view_faqs():
    if os.path.exists(FAQ_FILE):
        with open(FAQ_FILE, 'r', encoding='utf-8') as f:
            content = f.read()
        faqs = [faq.strip() for faq in content.split('//') if faq.strip()]
        return jsonify({"faqs": faqs})
    return jsonify({"faqs": []})


if __name__ == '__main__':
    # Create necessary files if they don't exist
    if not os.path.exists(FAQ_FILE):
        with open(FAQ_FILE, 'w', encoding='utf-8') as f:
            f.write("")

    app.run(debug=True)
