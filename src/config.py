"""
Módulo de configuração do ambiente e inicialização do LLM (LangChain).

Responsável por carregar variáveis de ambiente com segurança e instanciar
o modelo de linguagem escolhido (Google Gemini ou OpenAI GPT), aplicando
tratamento adequado para ausência de credenciais ou erros de inicialização.
"""

import os
from typing import Optional
from dotenv import load_dotenv

# Carrega automaticamente variáveis definidas no arquivo .env, se existir
load_dotenv()

DEFAULT_MODELS = {
    "gemini": "gemini-3.8-flash",
    "openai": "gpt-4o-mini",
}


def get_llm(
    provider: str = "gemini",
    api_key: Optional[str] = None,
    model_name: Optional[str] = None,
    temperature: float = 0.2,
):
    """
    Instancia e retorna o modelo de linguagem (LLM) configurado para o LangChain.

    Parâmetros:
        provider (str): Provedor desejado ("gemini" ou "openai").
        api_key (Optional[str]): Chave de API informada manualmente (ex.: via Streamlit).
                                 Caso omitida, busca nas variáveis de ambiente.
        model_name (Optional[str]): Nome específico do modelo (padrão: gemini-3.8-flash / gpt-4o-mini).
        temperature (float): Grau de criatividade do modelo (baixo para maior precisão no ReAct).

    Retorna:
        BaseChatModel: Instância do LLM pronta para uso no agente.

    Lança:
        ValueError: Quando a chave de API não é encontrada ou o provedor é inválido.
        RuntimeError: Quando ocorre falha ao inicializar o cliente do LLM.
    """
    provider_clean = provider.strip().lower()

    if provider_clean == "gemini":
        key = api_key or os.getenv("GOOGLE_API_KEY")
        if not key or "sua_chave" in key:
            raise ValueError(
                "Chave GOOGLE_API_KEY não configurada. Defina no arquivo .env "
                "ou informe na barra lateral da interface."
            )
        chosen_model = (model_name or os.getenv("GEMINI_MODEL") or DEFAULT_MODELS["gemini"]).strip()
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI

            return ChatGoogleGenerativeAI(
                model=chosen_model,
                google_api_key=key,
                temperature=temperature,
            )
        except Exception as exc:
            raise RuntimeError(f"Erro ao inicializar Google Gemini ({chosen_model}): {exc}") from exc

    elif provider_clean == "openai":
        key = api_key or os.getenv("OPENAI_API_KEY")
        if not key or "sua_chave" in key:
            raise ValueError(
                "Chave OPENAI_API_KEY não configurada. Defina no arquivo .env "
                "ou informe na barra lateral da interface."
            )
        chosen_model = (model_name or os.getenv("OPENAI_MODEL") or DEFAULT_MODELS["openai"]).strip()
        try:
            from langchain_openai import ChatOpenAI

            return ChatOpenAI(
                model=chosen_model,
                api_key=key,
                temperature=temperature,
            )
        except Exception as exc:
            raise RuntimeError(f"Erro ao inicializar OpenAI GPT ({chosen_model}): {exc}") from exc

    else:
        raise ValueError(
            f"Provedor '{provider}' não suportado. Utilize 'gemini' ou 'openai'."
        )
