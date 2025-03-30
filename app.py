from flask import Flask, render_template, request, jsonify
from langchain_ollama import OllamaEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama.llms import OllamaLLM
from langchain_community.vectorstores import FAISS
import re
import os

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


def retrieve_docs(db, query, k=10):
    # Returns list of (Document, score) tuples
    return db.similarity_search_with_score(query, k)


def question_pdf(question, documents):
    # Extract text from each document (ignoring scores)
    document_texts = [doc[0].page_content for doc in documents]
    context = "\n\n\n".join(document_texts)
    prompt = ChatPromptTemplate.from_template(main_template)
    chain = prompt | model
    response = chain.invoke({"question": question, "context": context})
    return re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL)


def load_faq_documents():
    # Load FAQ documents (you'll need to implement this based on your file structure)
    # This should return a list of documents similar to your model2.py implementation
    with open('fileij', 'r', encoding='utf-8') as f:
        content = f.read()
    return [line.strip() for line in content.split('//')]


def get_faq_response(question):
    # Implement the FAQ response logic from model2.py
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    import numpy as np

    def cleaner():
        with open('fileij', 'r', encoding='utf-8') as file:
            content = file.read()
        pattern = r'Pergunta:\s*(.*?\?)'
        perguntas = re.findall(pattern, content)
        return [pergunta.strip() for pergunta in perguntas]

    dirty_docs = load_faq_documents()
    cleaned_docs = cleaner()
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


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/ask', methods=['POST'])
def ask():
    question = request.form['question']
    use_faq = request.form.get('use_faq', 'true') == 'true'

    if use_faq:
        # First try with FAQ
        response = get_faq_response(question)
        return jsonify({
            'response': response,
            'method': 'FAQ'
        })
    else:
        # Fall back to document search
        embeddings = OllamaEmbeddings(model="qwq:32b")
        db = FAISS.load_local(
            "db_directory",
            embeddings,
            allow_dangerous_deserialization=True
        )
        related_documents = retrieve_docs(db, question)
        response = question_pdf(question, related_documents)
        return jsonify({
            'response': response,
            'method': 'Document Search'
        })


# Add this right before if __name__ == '__main__':

if __name__ == '__main__':
    app.run(debug=True)
