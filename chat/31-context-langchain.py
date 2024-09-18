from collections.abc import Iterator

from langchain_community.chat_models import ChatOllama
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate
from langchain_core.runnables.base import RunnableSerializable
from utils.args import init_args
from utils.prompt import Debugger, debug_runnable_fn, prompt_session
from langchain_community.document_loaders import TextLoader

# TODO: - 001 : Initialiser la chaine en utilisant le retriever de la BDD
#       - 002 : Récupérer les données de contexte et démarrer la session de prompt
#       - 003 : Créer le template de prompt en associant le system prompt et le human prompt
#       - 004 : Créer la chaine de traitement
#       - 005 : Compléter la fonction init_data pour charger les données
#       - 006 : Compléter le retour de la fonction ask_bot
def init_chain(model: BaseChatModel) -> RunnableSerializable:
  """
  Initialise la chaîne d'appel au LLM
  """

  # Create the custom prompt
  # TODO 003 - Tips : utiliser la fonction ChatPromptTemplate.from_messages
  system_prompt = """
    Réponds en utilisant uniquement les données de contexte fournies entre triple backquotes.
    Lorsque le contexte ne fournit pas d'informations pour répondre à la question posée, réponds que tu n'as pas la réponse.
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
  # TODO 004
  return (custom_prompt | debug_runnable_fn("Prompt") | model | StrOutputParser())


def init_data() -> str:
  """
  Initialise les données de contexte
  """

  # TODO 005 - Tips : utliser la fonction UnstructuredHTMLLoader
  docs = TextLoader(file_path="data/champ_euro_football_2024.html").load()
  return docs[0].page_content


def ask_bot(chain: RunnableSerializable, question: str, context_data: str) -> Iterator[str]:
  """
  Implémentation de l'appel au bot
  """

  # TODO 006 - Tips : utiliser la fonction stream
  return chain.stream({"question": question, "context_data": context_data})



if __name__ == "__main__":
  # Getting arguments from CLI
  args = init_args()

  # Activating the debug mode
  Debugger.debug_mode = args.debug

  # Instantiating the LLM chain
  # TODO 001
  model = ChatOllama(model=args.model, base_url=args.ollama_url, temperature=args.temperature)
  chain = init_chain(model)

  # Getting context data
  # TODO 002
  context_data = init_data()
  # Starting the prompt session
  prompt_session(lambda question: ask_bot(chain, question, context_data))
