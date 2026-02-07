# from langchain_openai import ChatOpenAI
from langchain_nvidia import ChatNVIDIA
from ds_agent.config import settings

class LLMFactory:
    """
    Factory for creating LLM instances with structured output and tool support.
    """
    def __init__(self, 
                 model_name: str = settings.model_name, 
                 temperature: float = settings.temperature,
                 thinking: bool = True, 
                 max_output_tokens: int = 2048,
                 max_retries: int = settings.max_retries):
        self.model_name = model_name
        self.temperature = temperature
        self.thinking = thinking
        self.max_output_tokens = max_output_tokens
        self.max_retries = max_retries

    def create(self):
        """
        Creates and returns a configured ChatNVIDIA instance.
        """
        return ChatNVIDIA(
            model=self.model_name,
            temperature=self.temperature,
            api_key=settings.model_api_key.get_secret_value(),
            max_tokens=self.max_output_tokens or settings.max_tokens,
            verify_ssl=False,
            # top_p= top_p or settings.top_p,
            model_kwargs = {'chat_template_kwargs':{'thinking':self.thinking}},
        )
