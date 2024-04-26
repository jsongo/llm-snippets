import time
from langchain_community.document_loaders import DirectoryLoader, UnstructuredMarkdownLoader
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.llms import Ollama
from langchain.indexes import VectorstoreIndexCreator

def load_documents(directory):
    return DirectoryLoader(directory, glob="**/*.md", loader_cls=UnstructuredMarkdownLoader)

def get_embedding():
    return OllamaEmbeddings(
        temperature=0,
        model="nomic-embed-text",
    )

def get_llm():
    return Ollama(
        # base_url='http://localhost:11434',
        model="yi",
    )

if __name__ == "__main__":
    start_time = time.perf_counter()
    print("loading documents...")
    # 加载文档
    doc_results = load_documents("./docs/")
    print(f"done within {(time.perf_counter() - start_time)*1000}ms.")

    # 创建一个向量索引对象
    index = VectorstoreIndexCreator(
        embedding=get_embedding(),
    ).from_loaders([doc_results])

    llm = get_llm()

    print("start to query...")
    # 在向量中去查询
    result = index.query("选择和评估一个软件或工具的 9 个指标有哪些", llm=llm)
    print(f"结果：\n{result}。 \ntotal time cost: {(time.perf_counter() - start_time)*1000}ms.")
