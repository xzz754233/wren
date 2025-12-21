from typing import Literal
from langchain_core.language_models import BaseChatModel
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from src.config import settings

def get_llm(mode: Literal["interview", "profile"] = "interview") -> BaseChatModel:
    """
    Factory function to return the configured LLM based on available API keys.
    Priority: Kimi -> OpenAI -> Gemini
    
    Args:
        mode: "interview" (needs speed) or "profile" (needs reasoning depth)
    """
    
    # 1. Priority: Kimi (Moonshot)
    if settings.moonshot_api_key:
        print(f"🔌 Using Kimi (Moonshot) - Mode: {mode}")
        model_name = "kimi-k2-thinking-turbo" if mode == "interview" else "kimi-k2-thinking"
        return ChatOpenAI(
            model=model_name,
            api_key=settings.moonshot_api_key,
            base_url=settings.moonshot_base_url,
            temperature=0.7,
            max_tokens=4000 if mode == "profile" else 1000,
        )

    # 2. Priority: OpenAI
    if settings.openai_api_key:
        print(f"🔌 Using OpenAI - Model: {settings.openai_model}")
        # o1/o3-mini models currently have specific restrictions (no system prompt sometimes, fixed temp)
        # But LangChain handles most of this adaptation.
        return ChatOpenAI(
            model=settings.openai_model, # e.g., "o3-mini" or "gpt-4o"
            api_key=settings.openai_api_key,
            temperature=1 if "o1" in settings.openai_model or "o3" in settings.openai_model else 0.7, 
            # Note: o1 models often require temperature=1
        )

    # 3. Priority: Google Gemini
    if settings.google_api_key:
        print(f"🔌 Using Google Gemini - Model: {settings.google_model}")
        return ChatGoogleGenerativeAI(
            model=settings.google_model, # e.g., "gemini-2.0-flash-exp"
            google_api_key=settings.google_api_key,
            temperature=0.7,
            convert_system_message_to_human=True # Helps with Gemini specific constraints
        )

    raise ValueError("No valid LLM configuration found.")