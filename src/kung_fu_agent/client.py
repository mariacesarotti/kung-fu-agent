import os
import sys

from google import genai
# Considerar isso caso algum importe quebre: as classes de erro da Interactions não são públicas na versão 2.31 do SDK.
from google.genai._gaos.lib.compat_errors import APIError


GEMINI_API_KEY = os.environ.get('GEMINI_API_KEY')

if not GEMINI_API_KEY:
    sys.exit("Chave API do Gemini não encontrada ou vazia.")

client = genai.Client()

try:
    interaction = client.interactions.create(
        model="gemini-3.5-flash-lite",
        input="what are the names for the katis/forms in kung fu choy lay fut?",
        generation_config={
            "thinking_level": "low"
        }
    )
    print(interaction.output_text)
except APIError as e:
    print(e.status_code)
    sys.exit(e.message)

client.close()
