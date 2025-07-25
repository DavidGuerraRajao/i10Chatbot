#Importacoes
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

#Inicialiazando o app local
app = Flask(__name__)

# Definindo o modelo de IA
modelo = "deepseek-r1:32b"
model = OllamaLLM(model=f"{modelo}")

# Interfaces da IA. É aqui onde a IA é dita oque é para ela vai fazer, como ele vai fazer e o contesto  dos documentos
iaManual_template = """
Você é um chatbot chamado David que responde perguntas sobre a Biblioteca virtual, i10 Bibliotecas.
Aqui estão algumas partes da documentação para te ajudar a responder ao usuario:
{contesto}

Aqui está a pergunta do usuário:
{pergunta}

Tome muito cuidado ao responder o usuário para não alucinar, não falar Português ou falar Inglês.
Aliás, se o que o usuário perguntou não está na documentação ou dentro do tema responda com preucaucao, mas mesmo assim você deve fornecer alguma resposta sem exceção!
Se possivel resuma sua resposta em no máximo 3 parágrafos com cada um contendo no máximo 2 frases.
Mas se voce ter que criar uma resposta mais longa fassa.

Responda aqui:
"""

pf_template = """
Você é um chatbot chamado David que responde perguntas sobre a Biblioteca virtual, i10 Bibliotecas.
Aqui estão algumas perguntas e respostas frequentes que tem aver com a pergunta do usuario para te ajudar a responder ao usuario:
{contesto}

Aqui está a pergunta do usuário:
{pergunta}

Tome muito cuidado ao responder o usuário para não alucinar, não falar Português ou falar Inglês.
Aliás, se o que o usuário perguntou não está na documentação ou dentro do tema responda com preucaucao, mas mesmo assim você deve fornecer alguma resposta sem exceção!
Se possivel resuma sua resposta em no máximo 3 parágrafos com cada um contendo no máximo 2 frases.
Mas se voce ter que criar uma resposta mais longa fassa.

Responda aqui:
"""

# Aonde esta as databases
PF_DataBase = "PF_DataBase"
Ai_Manual_DataBase = "Ai_Manual_DataBase"

# Esta funcao serve para filtrar os dados da DataBase da IA Manual
# mais relevantes em relacao a pergunta
def dados_relevantes(database, pergunta, k=10):
    return database.similarity_search_with_score(pergunta, k)

# A funcao AiManual serve para organizar os documentos filtrados para
# o chatbot da AiManual e retornar a resposta dele
def AiManual(question, documents):
    document_texts = [doc[0].page_content for doc in documents]
    context = "\n\n\n".join(document_texts)
    prompt = ChatPromptTemplate.from_template(iaManual_template)
    chain = prompt | model
    response = chain.invoke({"pergunta": question, "contesto": context})
    return re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL)

# Nesta parte se organiza e carrega os documentos do modo Perguntas Frequentes
def load_pf_documents():
    if os.path.exists(PF_DataBase):
        with open(PF_DataBase, 'r', encoding='utf-8') as f:
            content = f.read()
        return [line.strip() for line in content.split('//') if line.strip()]
    return []

# Nessa funcao calcula todas as variaveis do Perguntas Frequentes e gera uma resposta
def PF_response(question):
    documentos_semilimpos = load_pf_documents()

    def vassora():
        with open(PF_DataBase, 'r', encoding='utf-8') as file:
            content = file.read()
        pattern = r'Pergunta:\s*(.*?\?)'
        perguntas = re.findall(pattern, content)
        return [pergunta.strip() for pergunta in perguntas]

    documentos_limpos = vassora()
    if not documentos_limpos:
        return "Desculpe, não há perguntas frequentes disponíveis no momento."

    # Vetorizando os documentos limpos e filtrandos
    # os documentos mais parecidos com a pergunta para o search_docs
    vectorizer = TfidfVectorizer().fit(documentos_limpos)
    doc_vectors = vectorizer.transform(documentos_limpos)
    query_vector = vectorizer.transform([question])
    similarities = cosine_similarity(query_vector, doc_vectors).flatten()
    sorted_indices = np.argsort(-similarities)

    search_docs = ''
    n = 0
    for idx in sorted_indices:
        n += 1
        if similarities[idx] < 10.0:
            search_docs += f'\n\n{documentos_semilimpos[idx]}\n\n\n'
        if n == 8:
            break

    # Enviando todas as variaveis para o chatbot e finalizando por inviar o output fina
    prompt = ChatPromptTemplate.from_template(pf_template)
    chain = prompt | model
    resposta = chain.invoke({"pergunta": question, "contesto": search_docs})
    resposta_filtrada = re.sub(r'<think>.*?</think>', '', resposta, flags=re.DOTALL)
    return resposta_filtrada

print('Interface dos Admins: http://localhost:5000/admin')
print('Interface dos Usuarios: http://127.0.0.1:5000')

# Interface "Web" do chatbot:


# Interface do Usuario:

# Marcando a file de html
@app.route('/')
def home():
    return render_template('index.html')

# Funcao de perguntar para o chatbot
@app.route('/ask', methods=['POST'])
def ask():
    question = request.form['question']
    use_faq = request.form.get('use_faq', 'true') == 'true'

    if use_faq:
        response = PF_response(question)
        return jsonify({
            'response': response,
            'method': 'FAQ'
        })
    else:
        embeddings = OllamaEmbeddings(model=f"{modelo}")
        db = FAISS.load_local(
            Ai_Manual_DataBase,
            embeddings,
            allow_dangerous_deserialization=True
        )
        relevantes = dados_relevantes(db, question)
        response = AiManual(question, relevantes)
        return jsonify({
            'response': response,
            'method': 'Document Search'
        })


# Interface do Admin :

#Marcando localizacao do html
@app.route('/admin')
def admin():
    return render_template('admin.html')

# Interface de onde se adiciona mais PFs
@app.route('/admin/add_faq', methods=['POST'])
def add_faq():
    new_entry = request.form['faq_entry']

    # Validate the format
    if "Pergunta:" not in new_entry or "Resposta:" not in new_entry:
        return jsonify({"success": False, "message": "Formato inválido. Use 'Pergunta: ... Resposta: ...'"})

    # Add to FAQ file
    with open(PF_DataBase, 'a', encoding='utf-8') as f:
        f.write(f"\n{new_entry}\n//")

    return jsonify({"success": True, "message": "FAQ adicionada com sucesso!"})

# Interface onde se adiciona mais documentos para a IA Manual
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
    embeddings = OllamaEmbeddings(model=f"{modelo}")

    try:
        # Try to load existing DB
        db = FAISS.load_local(
            Ai_Manual_DataBase,
            embeddings,
            allow_dangerous_deserialization=True
        )
        # Add new documents to existing DB
        db.add_documents(splits)
    except:
        # If DB doesn't exist, create new
        db = FAISS.from_documents(splits, embeddings)

    with open("dbBackUp.txt", "w", encoding='utf-8') as f:
        f.write(new_doc)

    # Save the updated DB
    db.save_local(Ai_Manual_DataBase)

    return jsonify({"success": True, "message": "Documento adicionado ao banco de dados com sucesso!"})

# Vizualizacao das Perguntas Frequentes
@app.route('/admin/view_faqs')
def view_faqs():
    if os.path.exists(PF_DataBase):
        with open(PF_DataBase, 'r', encoding='utf-8') as f:
            content = f.read()
        faqs = [faq.strip() for faq in content.split('//') if faq.strip()]
        return jsonify({"faqs": faqs})
    return jsonify({"faqs": []})

# Rodar o codigo
if __name__ == '__main__':
    if not os.path.exists(PF_DataBase):
        with open(PF_DataBase, 'w', encoding='utf-8') as f:
            f.write("")

    app.run(debug=True)
