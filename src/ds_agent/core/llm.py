# from langchain_openai import ChatOpenAI
from langchain_nvidia import ChatNVIDIA
from ds_agent.config import settings

class LLMFactory:
    """
    Factory for creating LLM instances with structured output and tool support.
    """
    def __init__(self, model_name: str = settings.model_name, temperature: float = settings.temperature ,thinking: bool = True, max_output_tokens: int = 2048):
        self.model_name = model_name
        self.temperature = temperature
        self.thinking = thinking
        self.max_output_tokens = max_output_tokens

    def create(self):
        """
        Creates and returns a configured ChatNVIDIA instance.
        """
        return ChatNVIDIA(
            model=self.model_name,
            temperature=self.temperature,
            streaming=True,
            api_key=settings.model_api_key.get_secret_value(),
            max_tokens=self.max_output_tokens or settings.max_tokens,
            # top_p= top_p or settings.top_p,
            model_kwargs = {'chat_template_kwargs':{'thinking':self.thinking}},
        ).with_retry(stop_after_attempt=3)