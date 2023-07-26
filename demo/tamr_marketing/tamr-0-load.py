from llama_index import (
    SimpleDirectoryReader,
    GPTVectorStoreIndex,
    LLMPredictor,
    ServiceContext,
)

from llama_index.node_parser import SimpleNodeParser
from langchain import OpenAI
import os

os.environ["OPENAI_API_KEY"] = "sk-..."


# step 1 load documents
# documents = SimpleDirectoryReader('./data').load_data()

# # step 2 nodes
# parser = SimpleNodeParser()
# nodes = parser.get_nodes_from_documents(documents)


# # step 3 create index
# index = GPTVectorStoreIndex.from_documents(nodes)


# # step 4 customizing LLMs
# llm_predictor = LLMPredictor(llm=OpenAI(temperature=0, model_name = "text-davinci-003"))

# service_context = ServiceContext.from_defaults(llm_predictor=llm_predictor)

# index = GPTVectorStoreIndex.from_documents(
#     documents, service_context=service_context
# )

# # 5A - store index to disk
# index.storage_context.persist(persist_dir="./storage")

# 5B - load from disk
from llama_index import StorageContext, load_index_from_storage

storage_context = StorageContext.from_defaults(
    persist_dir="./demo/tamr_marketing/storage"
)

index = load_index_from_storage(storage_context)

# service_context = ServiceContext.from_defaults(llm_predictor=llm_predictor)


# 6 Querying
query_engine = index.as_query_engine()
query_input = "Why should I choose Tamr.  Keep response under 100 words"
response = query_engine.query(query_input)
print(response)
