from collections.abc import Iterator
import time

from langchain_community.vectorstores import FAISS
from langchain_community.chat_models import ChatOllama
from langchain_community.document_loaders import TextLoader
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


# TODO: - 001 : Créer et stocker les embeddings.
#       - 002 : Initialiser la chaine en utilisant le retriever de la BDD (FAISS)
#       - 003 : Initialiser la session de prompt
#       - 004 : Créer le template de prompt en associant le system prompt et le human prompt
#       - 005 : Créer la chaine de traitement
#       - 006 : Créér des chunks à partir du document champ_euro_football_2024.html
#       - 007 : Stocker les chunks dans la BDD
#       - 008 : Compléter le retour de la fonction ask_bot
def format_docs(docs: list[Document]) -> str:
  """
  Met en forme le résultat d'une recherche de documents de contexte
  """

  return "\n\n".join(doc.page_content for doc in docs)


def init_chain(model: BaseChatModel, retriever: BaseRetriever) -> RunnableSerializable:
  """
  Initialise la chaîne d'appel au LLM
  """

  # TODO 004 - Tips : utiliser la fonction ChatPromptTemplate.from_messages
  system_prompt = """
    Répond en 3 phrases maximum et utilise un ton neutre.
    Lorsque tu n'as pas d'informations pour répondre à la question posée, réponds que tu n'as la réponse.
    """
  human_template = "{question}"
  custom_prompt = ChatPromptTemplate.from_messages(
    [
      SystemMessage(system_prompt),
      HumanMessagePromptTemplate.from_template(human_template),
    ]
  )

  # TODO 005
  return (
    {"context_data": retriever | format_docs, "question": RunnablePassthrough()}
    | debug_runnable_fn("Données initiales")
    | custom_prompt
    | debug_runnable_fn("Prompt")
    | model
    | StrOutputParser()
  )


def init_data(embedding: Embeddings, _store: VectorStore, chunk_size: int, chunk_overlap: int) -> VectorStore:
  """
  Initialise les données de contexte
  """

  t0 = time.time()

  # Define the splitter
  # TODO 006 - Tips : utliser la méthode RecursiveCharacterTextSplitter et TextLoader
  text_splitter = RecursiveCharacterTextSplitter(
    chunk_size= chunk_size, chunk_overlap= chunk_overlap, add_start_index=True
  )
  all_splits = TextLoader(file_path="data/champ_euro_football_2024.wiki").load_and_split(text_splitter)

  debug(
    f"Contenu découpé en <ansigreen>{len(all_splits)} chunks</ansigreen> en <ansigreen>{(time.time() - t0):0.3} secondes</ansigreen>"
  )
  debug(f"Métadonnées du chunk n°<ansigreen>2</ansigreen> : {all_splits[2].metadata}")

  # Store the chunks
  t0 = time.time()
  # TODO 007 - Tips : utliser la méthode from_documents de l'objet store
  _store = _store.from_documents(documents=all_splits, embedding=embedding)
  debug(f"Génération et sauvegarde des embeddings en <ansigreen>{(time.time() - t0):0.3} secondes</ansigreen>")

  return _store


def ask_bot(chain: RunnableSerializable, question: str) -> Iterator[str]:
  """
  Implémentation de l'appel au bot
  """
  # TODO 008 - Tips : utiliser la fonction stream
  return chain.stream(question)


if __name__ == "__main__":
  # Getting arguments from CLI
  args = init_args()

  # Activating the debug mode
  Debugger.debug_mode = args.debug

  # Calculating and storing embeddings
  # TODO 001 -  Tips : Utiliser OllamaEmbeddings et la fonction init_data avec Chroma
  embeddings = OllamaEmbeddings(base_url=args.ollama_url, model=args.embeddings, temperature=args.temperature)
  store = init_data(embeddings, FAISS, args.chunk_size, args.chunk_overlap)

  # Instantiating the LLM chain
  # TODO 002
  model = ChatOllama(model=args.model, base_url=args.ollama_url)
  chain = init_chain(model, store.as_retriever())

  # Starting the prompt session
  # TODO 003
  prompt_session(lambda question: ask_bot(chain, question))
