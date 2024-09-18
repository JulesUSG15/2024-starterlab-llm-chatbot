from collections.abc import Iterator
import time

from langchain_community.vectorstores.pgvecto_rs import PGVecto_rs
from langchain_community.chat_models import ChatOllama
from langchain_community.embeddings.ollama import OllamaEmbeddings
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.runnables.base import RunnableSerializable
from langchain_core.vectorstores.base import BaseRetriever, VectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from utils.args import init_args
from utils.prompt import Debugger, debug, debug_runnable_fn, prompt_session

# TODO: - 001 : Créer un retriever de la BDD
#       - 002 : Initialiser la chaine en utilisant le retriever de la BDD
#       - 003 : Initialiser la session de prompt
#       - 004 : Créer le template de prompt en associant le system prompt et le human prompt
#       - 005 : Créer la chaine de traitement
#       - 006 : Compléter la fonction init_retriever
#       - 007 : Compléter le retour de la fonction ask_bot
def format_docs(docs: list[Document]) -> str:
  """
  Met en forme le résultat d'une recherche de documents de contexte
  """

  return "\n\n".join(doc.page_content for doc in docs)


def init_chain(_model: BaseChatModel, _retriever: BaseRetriever) -> RunnableSerializable:
  """
  Initialise la chaîne d'appel au LLM
  """

  # Create the custom prompt
  # TODO 004 - Tips : utiliser la fonction ChatPromptTemplate.from_messages
  system_prompt = """
    Réponds en utilisant uniquement les données de contexte fournies.
    Lorsque le contexte ne fournit pas d'informations pour répondre à la question posée,
    réponds que tu n'as pas la réponse.
    """
  human_template = """
    Contexte: {context_data}
    Question: {question}
    Réponse:
    """
  custom_prompt = ChatPromptTemplate.from_messages(
    [
      SystemMessage(system_prompt),
      HumanMessagePromptTemplate.from_template(human_template),
    ]
  )
  # Create the chain
  return (
    {"context_data": _retriever | format_docs, "question": RunnablePassthrough()}
    | debug_runnable_fn("Données initiales")
    | custom_prompt
    | debug_runnable_fn("Prompt")
    | _model
    | StrOutputParser()
  )


def init_retriever(_postgres_url: str, _embeddings: str) -> BaseRetriever:
  """
  Initialise les données de contexte
  """

  # loading the PGVecto.rs store
  # TODO 006
  db = PGVecto_rs.from_collection_name(
    embedding=OllamaEmbeddings(model=_embeddings),
    collection_name="doc_embeddings",
    db_url=_postgres_url,
  )
  # Getting the retriever
  _retriever = db.as_retriever()

  return _retriever


def ask_bot(_chain: RunnableSerializable, question: str) -> Iterator[str]:
    """
    Implémentation de l'appel au bot
    """

    # TODO 007 - Tips : utiliser la fonction stream
    return _chain.stream({"question": question})


if __name__ == "__main__":
  args = init_args()

  # Activating the debug mode
  Debugger.debug_mode = args.debug

  # Initiating the retriever
  # TODO 001
  retriever = init_retriever(args.postgres_url, args.embeddings)

  # Instantiating the LLM chain
  # TODO 002
  model = ChatOllama(model=args.model, base_url=args.ollama_url)
  chain = init_chain(model, retriever)

  # Starting the prompt session
  # TODO 003
  prompt_session(lambda question: ask_bot(chain, question))
