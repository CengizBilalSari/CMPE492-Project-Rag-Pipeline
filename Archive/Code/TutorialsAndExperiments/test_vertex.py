import os
from dotenv import load_dotenv
from langchain_google_vertexai import ChatVertexAI

# 1. Load the .env file
load_dotenv()
PROJECT_ID = os.getenv("GCP_PROJECT_ID")
LOCATION = os.getenv("GCP_LOCATION", "us-central1")

if not PROJECT_ID:
    raise ValueError("❌ GCP_PROJECT_ID is missing! Please add it to your .env file.")

print(f"Pinging Vertex AI on project: {PROJECT_ID}...")

# 2. Initialize the lightweight model for testing
try:
    llm = ChatVertexAI(
    model="gemini-2.5-flash",
    project=PROJECT_ID,
    location=LOCATION,
    temperature=0,
    max_tokens=32,
)
    response = llm.invoke("Output exactly 4 characters: PONG")
    print("content repr:", repr(response.content))
    print("content stripped:", response.content.strip())
    print("full response:", response)
    print("response_metadata:", getattr(response, "response_metadata", None))
    print("usage_metadata:", getattr(response, "usage_metadata", None))
    print("additional_kwargs:", getattr(response, "additional_kwargs", None))

except Exception as e:
    print(f"\n❌ API ERROR: Could not connect to Vertex AI.")
    print("Make sure you ran 'gcloud auth application-default login' and that your GCP_PROJECT_ID is correct.")
    print(f"Details: {e}")