from typing import List
import chromadb
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.llms import Ollama
from langchain_community.document_loaders import DirectoryLoader, UnstructuredMarkdownLoader

documents = [
  "Llamas are members of the camelid family meaning they're pretty closely related to vicuñas and camels",
  "Llamas were first domesticated and used as pack animals 4,000 to 5,000 years ago in the Peruvian highlands",
  "Llamas can grow as much as 6 feet tall though the average llama between 5 feet 6 inches and 5 feet 9 inches tall",
  "Llamas weigh between 280 and 450 pounds and can carry 25 to 30 percent of their body weight",
  "Llamas are vegetarians and have very efficient digestive systems",
  "Llamas live to be about 20 years old, though some only live for 15 years and others live to be 30 years old",
]

class RagWithDB:
    def __init__(self):
        self.embedding = OllamaEmbeddings(
            temperature=0,
            model="nomic-embed-text",
        )
        self.llm = Ollama(
            # base_url='http://localhost:11434',
            model="yi",
        )
        self.collection = None

    def get_collection(self, collection_name="vector_store"):
        client = chromadb.HttpClient(host="localhost", port=8000)
        return client.get_or_create_collection(name=collection_name)

    def embed_doc_and_add_to_collection(self, documents: List[str]):
        for i, doc in enumerate(documents):
            embed_results = self.embedding.embed_documents([doc])
            self.collection.add(
                ids=[str(i)],
                embeddings=embed_results,
                documents=[doc]
            )
    def add_documents(self, documents: List[str]):
        if not self.collection:
            self.collection = self.get_collection()
        self.embed_doc_and_add_to_collection(documents)
    
    def add_directory(self, directory, ftype="md"):
        loader = DirectoryLoader(directory, glob="**/*."+ftype, loader_cls=UnstructuredMarkdownLoader)
        documents = [doc.page_content for doc in loader.load()]
        self.embed_doc_and_add_to_collection(documents)

    def ask_local_rag(self, query: str):
        if not self.collection:
            self.collection = self.get_collection()
        query_embed = self.embedding.embed_query(query)

        relevant_docs = self.collection.query(query_embeddings=[query_embed], n_results=2)["documents"][0]
        docs = "\n\n".join(relevant_docs)
        model_query = f"用户原问题：{query} - 请尽量简短、且分点回答这个问题，回答所用素材如下:\n{docs}"
        # print(model_query)
        for chunks in self.llm.stream(model_query):
            print(chunks, end="", flush=True) 

if __name__ == "__main__":
    rag = RagWithDB()
    # 添加测试文本列表
    rag.add_documents(documents)
    # 添加文件夹里的数据
    rag.add_directory("./docs")
    print("文档添加完成…")

    # 测试问答
    query1 = "What are llama's favorite activities?"
    query2 = "思维导图的另类使用是什么？"
    print("问题：", query1)
    rag.ask_local_rag(query1)
    print("\n---------------\n问题：", query2)
    rag.ask_local_rag("思维导图的另类使用是什么？")

    # collection = get_collection("vector_store")
    # embedding = get_embedding()
    # embed_doc_and_add_to_collection(collection, documents)

    # query = "What are llama's favorite activities?"
    # query_embedding = embedding.embed_query(query)
    # results = collection.query(
    #     query_embeddings=query_embedding,
    #     n_results=1,
    # )
    # print(results["documents"][0][0])
