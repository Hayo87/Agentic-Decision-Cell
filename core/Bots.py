from huggingface_hub import InferenceClient
from transformers import pipeline
from ollama import chat
import os
from dotenv import load_dotenv

# Load the environment variable from the .env file
load_dotenv()
HF_KEY = os.getenv("HF_KEY")

def load(provider: str, model: str):
    """
    Create a *stateless* inference function for a given provider + model.

    This returns a simple callable:  fn(prompt: str) -> str

    - No chat history or memory is stored.
    - No system prompt is used.
    - Each call is an independent prompt response interaction.
    """
    match provider:
        
        case "hf":
            client = InferenceClient(api_key=HF_KEY)
            return lambda prompt: client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=800,
            ).choices[0].message.content
        
        case "ollama":
            return lambda prompt: chat(model=model, messages=[{"role": "user", "content": prompt}]).message.content


def load_chat_agent(provider: str, model: str, system_prompt: str = ""):
    """
    Create a *stateful* chat function with optional system prompt and memory.

    Returns a callable:  ask(user_message: str) -> str

    Behavior:
    - Maintains full chat history across calls.
    - Prepends a system prompt when provided.
    - Sends (system + all previous messages + new message) to the model.
    - Appends the model's response to the chat history.
    """

    if provider == "hf":
        client = InferenceClient(api_key=HF_KEY)
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        def ask(user_content: str) -> str:
            messages.append({"role": "user", "content": user_content})
            resp = client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=250,          
                temperature=0.2,        
                top_p=0.9,
                stop=[
                    "\nQuestion:",
                    "\nObservation:",
                    "------------------------------------------------------------"
                ],
            ).choices[0].message.content
            messages.append({"role": "assistant", "content": resp})
            return resp or ""

        return ask

    if provider == "ollama":
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        def ask(user_content: str) -> str:
            messages.append({"role": "user", "content": user_content})
            resp = chat(model=model, messages=messages).message.content
            messages.append({"role": "assistant", "content": resp})
            return resp or ""

        return ask

    raise ValueError("Unknown provider")
