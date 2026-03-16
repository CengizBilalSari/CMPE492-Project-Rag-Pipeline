import openai
import os

client = openai.OpenAI(
    base_url="https://8000-01kktj2tz5cbeqf9chb77mn3qc.cloudspaces.litng.ai/v1", 
    api_key="not-needed"
)

model_name = "meta-llama/Llama-3.1-8B-Instruct"

def test_model():

    print(f"send request to {model_name}")
    
    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "I am building a GraphRAG pipeline using vLLM and Mistral/Llama models.The current challenge is extracting high-fidelity entities and relationships from a 50-page document about complex supply chain logistics. What do you think about that? "
       }
            ],
            temperature=0.7,
            max_tokens=100
        )
        
        # 3. Yanıtı Yazdır
        answer = response.choices[0].message.content
        print("\n✅ Answer from vLLM")
        print("-" * 30)
        print(answer)
        print("-" * 30)
        
    except Exception as e:
        print(f"\n❌ Problem: {e}")

if __name__ == "__main__":
    test_model()