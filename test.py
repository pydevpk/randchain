import os
from sequential import Layer, Sequential
from typing import List
from config import EmailConfig, Whatsappconfig
from chat import ollama_chat
from utils import send_email, send_whatsapp, txt_writer, output_conversion
from decision_tree import DTCliassifier, LLM
from dotenv import load_dotenv


load_dotenv()


email_config = EmailConfig(
    sender_email=os.getenv("sender_email"),
    password=os.getenv("password"),
    recipient_email="<send_to_email>",
    subject="Automated Email from sequential chain",
    )

whatsapp_config = Whatsappconfig(
    account_sid=os.getenv("account_sid"),
    auth_token=os.getenv("auth_token"),
    from_number=os.getenv("from_number"),
    to_number=os.getenv("to_number")
)


age_prompt = """
    You are an insurance expert. Check user's age.
    - If age < 18, return 'Not Eligible'.
    - If 18 <= age <= 40, return 'Check Income'.
    - If age > 40, return 'Check Health'.
    Only return one of: Not Eligible, Check Income, Check Health. Do not return any preambles.
    """
income_prompt = """
    You are an insurance expert. Evaluate only income, ignore other factors:
    - If income < 300000 INR/year, return 'Low Eligibility'.
    - If income >= 300000 INR/year, return 'Eligible'.
    - User is already checked for health and age.
    Only return one of: Low Eligibility, Eligible. Do not return any preambles.
    """
health_prompt = """You are an insurance expert. Check user's health condition.
- If user has serious health issues like cancer, heart disease, chronic illness, return 'Not Eligible'.
- If user has minor health issues like mild diabetes, controlled hypertension, return 'Check Income'.
- If user has no health issues, return 'Check Income'.
Only return one of: Not Eligible, Check Income. Do not return any preambles."""

# Node: Age Check
age_node = DTCliassifier(
    "AgeAgent",
    LLM("Ollama", "qwen2.5:latest", age_prompt)
)

# Node: Income Check
income_node = DTCliassifier(
    "IncomeAgent",
    LLM("Ollama", "qwen2.5:latest", income_prompt)
)

# Node: Health Check
health_node = DTCliassifier(
    "HealthAgent",
    LLM("Ollama", "qwen2.5:latest", health_prompt),
    children=[
        {"condition": lambda d: "check income" in d.lower(), "node": income_node},
    ]
)

# Root Decision Tree
root = DTCliassifier(
    "RootInsuranceAgent",
    LLM("Ollama", "qwen2.5:latest", age_prompt),
    children=[
        {"condition": lambda d: "check income" in d.lower(), "node": income_node},
        {"condition": lambda d: "check health" in d.lower(), "node": health_node},
    ]
)

def make_a_decision(query):
    decision = root.decide(query)
    return decision

if __name__ == "__main__":
    chain = Sequential([
        Layer(make_a_decision),
        Layer(send_whatsapp, whatsapp_config=whatsapp_config),
    ])

    chain.run(["Hello, my name is Vijay. I am 45 years old, earn 10 lakh per year, and I have mild diabetes under control."])