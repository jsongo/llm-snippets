import os
import json
import chromadb
# from typing import List, Dict
from langchain_community.document_loaders import UnstructuredMarkdownLoader
from langchain_community.embeddings import OllamaEmbeddings
from sklearn.metrics.pairwise import cosine_similarity
from langchain_community.llms import Ollama

# 基于文件的 RAG
class RagWithFileStore:

    def __init__(self):
        self.directory = "vector_store"
        self.embedding = OllamaEmbeddings(
            temperature=0,
            model="nomic-embed-text",
        )
        self.llm = Ollama(
            # base_url='http://localhost:11434',
            model="yi",
        )
        self.data = []
    
    def _load_embeddings(self, path):
        full_path = f"{self.directory}/{path}.json"
        if os.path.exists(full_path):
            with open(full_path, "r") as file:
                return json.load(file)
        return None

    def _save_embeddings(self, path, embeddings):
        if not os.path.exists(self.directory):
            os.mkdir(self.directory)
        with open(f"{self.directory}/{path}.json", "w") as file:
            # 把 embeddings 存入文件
            json.dump(embeddings, file)

    def embed_docs(self, documents):
        return self.embedding.embed_documents(documents)[0]
    
    def embed_text(self, text):
        return self.embedding.embed_query(text)

    # 加载文件
    def load_embeddings(self, directory):
        for filename in os.listdir(directory):
            full_path = f"{directory}/{filename}"
            # 文件名把路径打平
            path = full_path.replace("/", "__")
            document = UnstructuredMarkdownLoader(full_path).load()
            embedding = self._load_embeddings(path)
            if not embedding:
                embedding = self.embed_docs(document)
                self._save_embeddings(path, embedding)
            self.data.append({
                "path": path,
                "text": document[0].page_content,
                "embedding": embedding
            })

    def find_simular(self, query: str, n_results = 2):
        if len(self.data) == 0:
            return []
        query_embeddings = [self.embed_text(query)]
        similarities=[]
        
        for d in self.data:
            similarity = cosine_similarity(query_embeddings, [d["embedding"]])[0][0]
            similarities.append({"data": d, "similarity": similarity})

        output = sorted(similarities, key=lambda x: x["similarity"], reverse=True)[:n_results]
        return [{"title": o["data"]["path"], "content": o["data"]["text"]} for o in output]

if __name__ == "__main__":
    query = "选择和评估一个软件或工具的 9 个指标有哪些"
    rag = RagWithFileStore()
    # 先加载资料文档
    rag.load_embeddings("./docs")

    # 用近似查找算法来找到最接近的
    result = rag.find_simular(query)
    if len(result) > 0:
        print([r["title"] for r in result])

    # 也可以借助 chromadb 的 collection 封装来实现
    client = chromadb.Client()
    collection = client.create_collection("vector_store")
    for d in rag.data:
        collection.add(
            ids=[d["path"]],
            embeddings=[d["embedding"]],
            documents=[d["text"]]
        )
    result = collection.query(
        query_embeddings=[rag.embed_text(query)],
        n_results=2,
    )
    print(result['ids'])
