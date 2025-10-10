from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_chroma import Chroma 
from langchain_huggingface import HuggingFaceEndpoint
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents.stuff import create_stuff_documents_chain
from langchain.prompts import ChatPromptTemplate 
from dotenv import load_dotenv
import os

load_dotenv("D:\Soft. Engr\DeepLearning\RAG\.env")
#os.environ["HUGGINGFACEHUB_API_TOKEN"] = 
print(os.getenv("HUGGINGFACEHUB_API_TOKEN")) 


# Reading the File.  
with open("cleaned_text_file.txt", "r", encoding="utf-8") as f: 
    text = f.read()

# Splitting the text into chunks. 
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000,
                                               chunk_overlap=100)

chunks = text_splitter.split_text(text)

print(f"Total Chunks {len(chunks)}")


# Embedding the chunks created by splitting. 
model_name = "BAAI/bge-small-en"
model_kwargs = {"device": "cpu"}
encode_kwargs = {"normalize_embeddings": True}
hf = HuggingFaceBgeEmbeddings(
    model_name=model_name, model_kwargs=model_kwargs, encode_kwargs=encode_kwargs
)
# vector = hf.embed_documents(chunks)

# print(f"Vector representation of some chunks: {vector[:5]}")
# print(f"Embedded chunks successfully!")

# Storing Embeddings in VectorDB 
vector_store = Chroma.from_texts(
    chunks,
    hf,
    persist_directory="chroma_db"
)

# Retriever creation
retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 3})

# Loading LLM 
llm = HuggingFaceEndpoint(
    repo_id="mistralai/Mixtral-8x7B-Instruct-v0.1",
    task="conversational",
    temperature=0.3,
    max_new_tokens=512,
    huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN")
)

# Defining Prompts 
prompt = ChatPromptTemplate.from_template(
    """
    Answer the following question based only on the provided context:

    <context>
    {context}
    </context>

    Question: {input}
    """
)

#Combining Multiple Chunks retrieved from the Vector Database. 
combine_docs_chain = create_stuff_documents_chain(
    llm=llm,
    prompt=prompt
)

# Creating RetrievalQA Chain 
retrieval_chain = create_retrieval_chain(
    retriever=retriever, 
    combine_docs_chain=combine_docs_chain 
) 

# Querying 
query = "Summarize main ideas from the provided text."
response = retrieval_chain.invoke({"input": query})
print("Answer", response["answer"])



