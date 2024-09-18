from collections.abc import Iterator

from ollama import Client
from utils.args import init_args
from utils.prompt import Debugger, debug_label, prompt_session
from datasets import load_dataset


# TODO: - 001 : Créer l'appel au client dans le main
#       - 002 : Charger le dataset python_code_instructions_18k_alpaca (train) et sélectionner les 100 premières lignes
#       - 003 : Formatter les données Créer un system prompt avec les exemples
#       - 004 : Construire le dict messages avec
#         - 1 prompt de rôle system
#         - 1 prompt de rôle user dont le contenu est question
#       - 005 : Appel au modèle (mode stream) et retourner le "message.content" de la réponse
def ask_bot(model: str, client: Client, temperature: float, question: str, n_rows: int = 100) -> Iterator[str]:
  """
  Implémentation de l'appel au bot
  """

  # load data using HuggingFace datasets API
  # TODO 002 - Charger le dataset python_code_instructions_18k_alpaca (train) et sélectionner les 100 premières lignes
  ds = load_dataset("python_code_instructions_18k_alpaca", split="train")[0:n_rows]
  
  ds = ''.join(["{Question :\n"+ds["instruction"][x] + "}\nAnswer:\n{"+ds["output"][x]+"}\n\n" for x in range(n_rows)])

  # TODO 003 - Formatter les données Créer un system prompt avec des exemples
  system_prompt = f"""
    Tu es un assistant répondant à des questions sur des instructions de code Python.
    Tu dois rédiger une réponse en une phrase à la question posée par l'utilisateur.
    Voici quelques exemples : 

    {ds}
    """

  # TODO 004 - Same as previous exercise with an enriched system_prompt
  messages = [
    {
      "role": "system",
      "content": system_prompt
    },
    {
      "role": "user",
      "content": question
    }
  ]

  debug_label("Prompt", messages)

  # TODO 005 - Tips : Use the chat method of the client object
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

  # Activation du mode debug
  Debugger.debug_mode = args.debug

  # Creating the model client
  client = Client(host=args.ollama_url)

  # Starting the prompt session
  prompt_session(lambda question: ask_bot(args.model, client, args.temperature, question))
