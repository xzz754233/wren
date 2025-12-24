from typing import Literal
from langchain_core.language_models import BaseChatModel
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI, HarmBlockThreshold, HarmCategory
from src.config import settings

def get_llm(mode: Literal["interview", "profile"] = "interview") -> BaseChatModel:
    
    provider = settings.llm_provider
    
    if provider == "auto":
        if settings.moonshot_api_key: provider = "moonshot"
        elif settings.google_api_key: provider = "gemini"
        elif settings.openai_api_key: provider = "openai"
    
    print(f"🔌 LLM Factory initializing: Provider={provider}, Mode={mode}")

    if provider in ["gemini", "google"]:
        if not settings.google_api_key:
            raise ValueError("Provider is Gemini but GOOGLE_API_KEY is missing")
        
        # [FIX] Explicit Safety Settings to prevent empty responses
        safety_settings = {
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
        }

        return ChatGoogleGenerativeAI(
            model=settings.google_model,
            google_api_key=settings.google_api_key,
            temperature=0.7,
            # [FIX] Do not force convert system to human, let Gemini handle it naturally
            # or handle it manually in the agent. Setting this to True often causes
            # "User, User" consecutive message errors.
            convert_system_message_to_human=False,
            safety_settings=safety_settings
        )

    if provider == "openai":
        if not settings.openai_api_key:
            raise ValueError("Provider is OpenAI but OPENAI_API_KEY is missing")
        return ChatOpenAI(
            model=settings.openai_model,
            api_key=settings.openai_api_key,
            temperature=0.7
        )

    if provider == "moonshot" or settings.moonshot_api_key:
        model_name = "kimi-k2-thinking-turbo" if mode == "interview" else "kimi-k2-thinking"
        return ChatOpenAI(
            model=model_name,
            api_key=settings.moonshot_api_key,
            base_url=settings.moonshot_base_url,
            temperature=0.7,
            max_tokens=4000 if mode == "profile" else 1000,
        )

    raise ValueError("Could not determine LLM provider. Check your API keys.")