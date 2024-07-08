from openai import OpenAI
import logging

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO
)
logging.getLogger("httpx").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

client = OpenAI(
    base_url='http://localhost:11434/v1/',
    api_key='ollama',
)

def llm(prompt):
    response = client.chat.completions.create(
        model="gemma:2b",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0
    )
    logger.info(f"Out tokens count: {response.usage.completion_tokens}\n\n")
    return response.choices[0].message.content


query = "What's the formula for energy?"

response = llm(query)
print(response)