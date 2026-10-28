import os
from dotenv import load_dotenv
from openai import OpenAI, NotFoundError
from tavily import TavilyClient

# Load environment variables
load_dotenv()

nebius_api_key = os.getenv("NEBIUS_API_KEY")
tavily_api_key = os.getenv("TAVILY_API_KEY")

print("=== Starting Connection Tests ===\n")

# 1. Nebius Token Factory Connection
print("--- Testing Nebius Token Factory (OpenAI API) ---")
if not nebius_api_key:
    print("[ERROR] NEBIUS_API_KEY is not set in .env")
else:
    try:
        client = OpenAI(
            base_url="https://api.tokenfactory.nebius.com/v1/",
            api_key=nebius_api_key,
        )
        model_name = "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B"
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {
                        "role": "user",
                        "content": "Reply with: Nebius Token Factory Connected!",
                    }
                ],
                max_tokens=250,
            )
            content = response.choices[0].message.content or ""
            print(f"[SUCCESS] Nebius response: {content.strip()}")
        except NotFoundError:
            print(f"[404] Model '{model_name}' not found. Fetching available nvidia/ models...")
            models = client.models.list()
            nvidia_models = [m.id for m in models.data if "nvidia" in m.id.lower()]
            print("Available nvidia/ models:")
            for m in nvidia_models:
                print(f" - {m}")
        except Exception as e:
            if "404" in str(e):
                print(f"[404] Model '{model_name}' not found. Fetching available nvidia/ models...")
                try:
                    models = client.models.list()
                    nvidia_models = [m.id for m in models.data if "nvidia" in m.id.lower()]
                    print("Available nvidia/ models:")
                    for m in nvidia_models:
                        print(f" - {m}")
                except Exception as list_err:
                    print(f"[ERROR] Could not list models: {list_err}")
            else:
                print(f"[ERROR] Nebius request failed: {e}")
    except Exception as e:
        print(f"[ERROR] Nebius client initialization failed: {e}")

print()

# 2. Tavily Client Connection
print("--- Testing Tavily Client ---")
if not tavily_api_key:
    print("[ERROR] TAVILY_API_KEY is not set in .env")
else:
    try:
        tavily_client = TavilyClient(api_key=tavily_api_key)
        query = "Legal Metrology Packaged Commodities Rules 2011 mandatory declarations"
        search_res = tavily_client.search(query=query)
        results = search_res.get("results", [])
        if results:
            print(f"[SUCCESS] Tavily first result title: {results[0].get('title')}")
        else:
            print("[WARNING] Tavily returned no results.")
    except Exception as e:
        print(f"[ERROR] Tavily search failed: {e}")

print("\n=== Connection Tests Finished ===")
