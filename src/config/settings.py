import os
from typing import Optional
from dotenv import load_dotenv

load_dotenv()

class Settings:
    def __init__(self):
        self.moonshot_api_key = os.getenv("MOONSHOT_API_KEY")
        self.moonshot_base_url = os.getenv("MOONSHOT_BASE_URL", "https://api.moonshot.cn/v1")
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        self.openai_model = os.getenv("OPENAI_MODEL", "gpt-4o")
        self.google_api_key = os.getenv("GOOGLE_API_KEY")
        self.google_model = os.getenv("GOOGLE_MODEL", "gemini-2.0-flash-exp")
        self.redis_host = os.getenv("REDIS_HOST", "localhost")
        self.redis_port = int(os.getenv("REDIS_PORT", "6379"))
        self.redis_password = os.getenv("REDIS_PASSWORD")
        self.llm_provider = os.getenv("LLM_PROVIDER", "auto").lower()

    def validate(self) -> None:
        has_moonshot = self.moonshot_api_key is not None and len(self.moonshot_api_key) > 1
        has_openai = self.openai_api_key is not None and len(self.openai_api_key) > 1
        has_google = self.google_api_key is not None and len(self.google_api_key) > 1

        if not any([has_moonshot, has_openai, has_google]):
            raise ValueError(
                "No valid API keys found! Please set MOONSHOT_API_KEY, OPENAI_API_KEY, or GOOGLE_API_KEY."
            )

settings = Settings()

