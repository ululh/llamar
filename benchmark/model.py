from llama_index import SimpleDirectoryReader
from llama_index.node_parser import SimpleNodeParser
from llama_index.schema import IndexNode
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from llama_index.embeddings import LangchainEmbedding
from llama_index.llms import LlamaCPP
from llama_index.llms.llama_utils import messages_to_prompt, completion_to_prompt
from llama_index import ServiceContext
from llama_index import VectorStoreIndex
from llama_index.prompts import PromptTemplate
from llama_index import ServiceContext
import glob

print("Chargement des documents")
documents = SimpleDirectoryReader(
    input_files=glob.glob("./input/*.txt")
).load_data()

node_parser = SimpleNodeParser.from_defaults(chunk_size=500)
base_nodes = node_parser.get_nodes_from_documents(documents)

# Name or path to sentence-transformers embedding model.
#  - Multilingual: paraphrase-multilingual-mpnet-base-v2, paraphrase-multilingual-MiniLM-L12-v2
#  - French: dangvantuan/sentence-camembert-base, dangvantuan/sentence-camembert-large
EMBEDDING_MODEL_NAME = 'dangvantuan/sentence-camembert-base'

print(f"Chargement du modéle Embedding: {EMBEDDING_MODEL_NAME} ...")

embedding_model = LangchainEmbedding(HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL_NAME,
    encode_kwargs = {"normalize_embeddings": False}
  )
)

LLM_MODEL = "../model/vigogne-2-7b-chat.Q5_K_M.gguf"
#LLM_MODEL = "../model/vigogne-2-7b-chat.Q8_0.gguf"

print(f"Chargement du modéle LLM: {LLM_MODEL} ...")

llm = LlamaCPP(
    # optionally, you can set the path to a pre-downloaded model instead of model_url
    model_path=LLM_MODEL,
    temperature=0.1,
    max_new_tokens=1024,
    generate_kwargs={},
    model_kwargs={
        "low_cpu_mem_usage": True,
    },
    messages_to_prompt=messages_to_prompt,
    completion_to_prompt=completion_to_prompt,
    verbose=True,
)


service_context=ServiceContext.from_defaults(
      llm=llm,
      embed_model=embedding_model
)

vectorstore_index = VectorStoreIndex(
    nodes=base_nodes,
    service_context=service_context
)
print("Sauvegarde de la base de données vectorielle")
vectorstore_index.storage_context.persist(persist_dir='llama_index')
print("Sauvegardée")

text_qa_template_str = (
  "<|system|>: Vous êtes un assistant IA qui répond à la question posée à la fin en utilisant le contexte suivant. Si vous ne connaissez pas la réponse, dites simplement que vous ne savez pas, n'essayez pas d'inventer une réponse. Veuillez répondre exclusivement en français.\n"
  "<|user|>: {context_str}\n"
  "Question: {query_str}\n"
  "<|assistant|>:"
)

text_qa_template = PromptTemplate(text_qa_template_str)

query_engine = vectorstore_index.as_query_engine(
    text_qa_template=text_qa_template,
    service_context=ServiceContext.from_defaults(
      llm=llm,
      embed_model=embedding_model,
      chunk_size=500,
    ),
)

# "Quel est l'avantage des engrais verts ?",
questions = [ 
              "Peux-tu m'indiquer des fixateurs d'azote ?"
            ]

for question in questions:
    print(f'Question: {question}')
    response = query_engine.query(question)
    print(f'Réponse: {response}')
    print()
