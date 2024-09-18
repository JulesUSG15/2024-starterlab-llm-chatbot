from collections.abc import Iterator
from pathlib import Path

from ollama import Client
from utils.args import init_args
from utils.prompt import Debugger, debug_label, prompt_session


# TODO: - 001 :  Charger les données de contexte (champ_euro_fottball_2024.txt)
#       - 002 :  Ajouter context_data dans ask_bot
#       - 003 :  Créer le system prompt et le user prompt pour n'utiliser que le contexte
#       - 004 : Appel au model via le client Ollama (en mode stream)
def init_data() -> str:
  """
  Initialise les données de contexte
  """

  # TODO 001
  return Path("data/champ_euro_football_2024.txt").read_text()


def ask_bot(model: str, client: Client, temperature: float, question: str, context_data: str) -> Iterator[str]:
  """
  Implémentation de l'appel au bot
  """

  # TODO 003
  system_prompt = """
    Tu es un chatbot qui répond à des questions en utilisant uniquement les données de contexte fournies et en utilisant un ton formel.
    Tu réponds toujours en français, quelque soit la langue dans laquelle la requête utilisateur est donnée.
    Lorsque le contexte ne fournit pas d'informations sur la question posée, réponds seulement que tu n'as pas la réponse.
    Par exemple, si la question concerne une recette de cuisine, tu dois répondre que tu n'as pas la réponse.
    """
  user_prompt = f"""
  Context : {context_data}
  Question : {question}
  Reponse :
  """

  messages = [
    { "role": "system", "content": system_prompt },
    { "role": "user", "content": user_prompt }
  ]

  debug_label("Prompt", messages)

  # TODO 004
  return map(
    lambda x: x["message"]["content"],
    client.chat(
      model=model,
      messages=messages,
      options={"temperature": temperature},
      stream=True,
    ),
  )


if __name__ == "__main__":
  # Getting arguments from CLI
  args = init_args()

  # Activating the debug mode
  Debugger.debug_mode = args.debug

  # Creating the model client
  client = Client(host=args.ollama_url)

  # TODO 002
  # Getting context data
  context_data = init_data()

  # Starting the prompt session
  prompt_session(lambda question: ask_bot(args.model, client, args.temperature, question, context_data))
