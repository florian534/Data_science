from flask import Flask, render_template, request
from langchain import hub
from langchain.chains import RetrievalQA
from langchain.llms import Ollama
from langchain.embeddings.ollama import OllamaEmbeddings
from langchain.vectorstores import Chroma
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory

app = Flask(__name__)

# Configuration du modèle et des données
FILEPATH = "Thèse_matmut.pdf"  # Remplace par le chemin de ton fichier PDF
LOCAL_MODEL = "llama2"
EMBEDDING = "nomic-embed-text"

# Chargement du document PDF et préparation des embeddings
loader = PyPDFLoader(FILEPATH)
data = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=100)
all_splits = text_splitter.split_documents(data)

persist_directory = 'data'
vectorstore = Chroma.from_documents(
    documents=all_splits,
    embedding=OllamaEmbeddings(model=EMBEDDING),
    persist_directory=persist_directory
)

llm = Ollama(base_url="http://localhost:11434", model=LOCAL_MODEL, verbose=True)

retriever = vectorstore.as_retriever()

prompt = """ Vous êtes un chatbot expérimenté, là pour répondre aux questions de l'utilisateur. Votre ton doit être professionnel et informatif. Vous devez absolument répondre en français.

    Contexte : {context}
    Historique : {history}

    Utilisateur : {question}
    Chatbot (répond toujours en français) :
"""

prompt_template = PromptTemplate(input_variables=["history", "context", "question"], template=prompt)

memory = ConversationBufferMemory(memory_key="history", return_messages=True, input_key="question")

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type='stuff',
    retriever=retriever,
    verbose=True,
    chain_type_kwargs={
        "verbose": True,
        "prompt": prompt_template,
        "memory": memory,
    }
)

@app.route('/')
def home():
    return render_template('index.html')  # Page d'accueil

@app.route('/ask', methods=['POST'])
def ask():
    user_query = request.form['question']  # Récupère la question posée par l'utilisateur
    query = f"Donne moi la réponse en français à partir des informations du PDF: {user_query}"
    
    # Obtenir la réponse du modèle
    response_rag = qa_chain.invoke({"query": query})
    return render_template('index.html', user_query=user_query, response=response_rag['result'])

if __name__ == '__main__':
    app.run(debug=True)
