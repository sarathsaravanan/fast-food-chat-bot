import http.server
import socketserver
import json
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import ChatPromptTemplate
from langchain.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_community.llms.octoai_endpoint import OctoAIEndpoint
from langchain_text_splitters import RecursiveCharacterTextSplitter, HTMLHeaderTextSplitter
from dotenv import load_dotenv
import os

load_dotenv()
OCTOAI_API_TOKEN = os.environ["eyJhbGciOiJSUzI1NiIsInR5cCI6IkpXVCIsImtpZCI6IjNkMjMzOTQ5In0.eyJzdWIiOiIyZWY4MGFjYi1jM2M0LTQzZjMtYjVmOC0yYzQyYTQyYWVjNTYiLCJ0eXBlIjoidXNlckFjY2Vzc1Rva2VuIiwidGVuYW50SWQiOiI2OWEyZjhmZS1mM2FiLTQ2OGMtOTc4YS1lNzEwN2YyMDk1ZDQiLCJ1c2VySWQiOiJmZDFmYjIyZS1iODk5LTRjMTEtODYzZi1lNWE5Y2E1ZTU2N2EiLCJhcHBsaWNhdGlvbklkIjoiYTkyNmZlYmQtMjFlYS00ODdiLTg1ZjUtMzQ5NDA5N2VjODMzIiwicm9sZXMiOlsiRkVUQ0gtUk9MRVMtQlktQVBJIl0sInBlcm1pc3Npb25zIjpbIkZFVENILVBFUk1JU1NJT05TLUJZLUFQSSJdLCJhdWQiOiIzZDIzMzk0OS1hMmZiLTRhYjAtYjdlYy00NmY2MjU1YzUxMGUiLCJpc3MiOiJodHRwczovL2lkZW50aXR5Lm9jdG8uYWkiLCJpYXQiOjE3MTcyNjY5ODl9.s2ZrU9iW4BF9efsPinbNOEfVOVF7ETr2kicDaKM8DpyO_7l5Q_sa19atNALrAGj37cNaSU4SqDAYDmMlDnrmygK14niWTnvIEPh66fIsgQ6GPGteT4qOKZKc8InbaGWcTKJaMBgUoBkl-QfTEFoc-dUUORRAxphAChJVZlIFWu63jT5DT7MK8O2WAOVTg5eIjVeD1YkfYHiId_z-Zom_4lgmu_cHxR3gzMeN2xEmlFbyP9GtIHlD5TAvaX3zO5G7i3HvjvTMs5Ky_yoUysp3gSX5C9EclV_dxyvjEsHsBmEfIVw0nFJsPR9-n1ANJKth4KVXIbVtWVhV5VQFA5zOTg"]
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]

PORT = 8000

# Load and split text from URL
url = "https://en.wikipedia.org/wiki/Star_Wars"
headers_to_split_on = [
    ("h1", "Header 1"),
    ("h2", "Header 2"),
    ("h3", "Header 3"),
    ("h4", "Header 4"),
    ("div", "Divider")
]

html_splitter = HTMLHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
html_header_splits = html_splitter.split_text_from_url(url)

chunk_size = 1024
chunk_overlap = 128
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=chunk_size,
    chunk_overlap=chunk_overlap,
)
splits = text_splitter.split_documents(html_header_splits)

# Initialize LLM and vector store
llm = OctoAIEndpoint(
    model="llama-2-13b-chat-fp16",
    max_tokens=1024,
    presence_penalty=0,
    temperature=0.1,
    top_p=0.9,
)
embeddings = OpenAIEmbeddings()
vector_store = FAISS.from_documents(
    splits,
    embedding=embeddings
)
retriever = vector_store.as_retriever()

template = """You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.
Question: {question} 
Context: {context} 
Answer:"""
prompt = ChatPromptTemplate.from_template(template)

chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

class MyHandler(http.server.SimpleHTTPRequestHandler):
    def do_POST(self):
        if self.path == '/ask':
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            data = json.loads(post_data)
            question = data['question']
            answer = chain.invoke(question)
            response = {'answer': answer}

            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps(response).encode())

with socketserver.TCPServer(("", PORT), MyHandler) as httpd:
    print("serving at port", PORT)
    httpd.serve_forever()
