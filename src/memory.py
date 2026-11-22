"""
Módulo de Memória Conversacional (LangChain Core).

Utiliza `InMemoryChatMessageHistory` do `langchain_core.chat_history` com
janela deslizante (window_size), mantendo compatibilidade total com Python 3.10
a 3.14 e permitindo que o agente lembre dados de turnos anteriores
(como CEP, cidade, metragem e serviço mencionado).
"""

from langchain_core.chat_history import InMemoryChatMessageHistory


class ConversationalWindowMemory:
    """Gerencia o histórico conversacional com janela deslizante de interações."""

    def __init__(self, window_size: int = 8):
        self.window_size = window_size
        self.history = InMemoryChatMessageHistory()

    def add_exchange(self, user_message: str, ai_message: str) -> None:
        """Armazena um par pergunta/resposta e mantém o limite da janela deslizante."""
        self.history.add_user_message(user_message)
        self.history.add_ai_message(ai_message)

        max_messages = self.window_size * 2
        if len(self.history.messages) > max_messages:
            self.history.messages = self.history.messages[-max_messages:]

    def format_history(self) -> str:
        """Formata o histórico armazenado em texto legível para injeção no prompt ReAct."""
        if not self.history.messages:
            return "(Nenhuma interação anterior nesta sessão.)"

        linhas = []
        for msg in self.history.messages:
            papel = "Usuário" if msg.type == "human" else "ObraSmart"
            linhas.append(f"{papel}: {msg.content}")
        return "\n".join(linhas)

    def clear(self) -> None:
        """Limpa todo o histórico conversacional da sessão."""
        self.history.clear()


def create_conversational_memory(window_size: int = 8) -> ConversationalWindowMemory:
    """Fábrica que retorna uma instância configurada de memória conversacional."""
    return ConversationalWindowMemory(window_size=window_size)
