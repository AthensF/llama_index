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

storage_context = StorageContext.from_defaults(persist_dir="./demo/bible/storage")

index = load_index_from_storage(storage_context)

# service_context = ServiceContext.from_defaults(llm_predictor=llm_predictor)


# 6 Querying
query_engine = index.as_query_engine()
query_input = "What would Jesus say about dating as a teenager in the modern world?  Keep response under 100 words"
response = query_engine.query(query_input)
print(response)

# Prompt: What would Jesus say about a situation where a reckless owner whose dog bit me?
# Keep response under 100 words
#
# Jesus would likely say that the owner should be held accountable
# for their dog's actions. He would emphasize the importance of taking responsibility
# for one's actions and the consequences of those actions.
# He would likely also emphasize the importance of
# loving one's neighbor and treating them with respect and kindness.
# He would likely encourage the owner to make amends and to
# do what is necessary to make the situation right.

# Prompt: "What would Jesus do or say about people who harmed us?  Keep response under 100 words"
# Jesus would say that we should not take revenge on those who have harmed us.
# Instead, we should forgive them and pray for them. He said, "Love your enemies,
# bless them that curse you, do good to them that hate you, and pray for them which
# despitefully use you, and persecute you" (Matthew 5:44).
# He also said, "If thy brother trespass against thee, rebuke him; and if he repent,
# forgive him" (Luke 17:3). Jesus taught us to forgive those who have wronged us,
# and to seek reconciliation with them. He also taught us to fast and pray for
# those who have hurt us, so that God can work in their hearts and bring them to repentance.

# Prompt: "What should we do about injustice?  Keep response under 100 words"
# We should strive to bring justice to those who have been wronged. We should work to ensure that those who have been wronged are given a fair and just outcome.
# We should also work to ensure that those who have committed injustices are held accountable for their actions.
# We should strive to create a society where everyone is treated with respect and dignity, and where everyone is given equal opportunities.
# We should also work to ensure that those who are in positions of power are held to a higher standard of accountability and justice.
#  Finally, we should strive to create a society where everyone is treated with fairness and justice, regardless of their background or identity.
