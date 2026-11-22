"""
Módulo do Agente ReAct (Reasoning + Acting) construído sobre LangChain Core.

Integra o LLM, as ferramentas externas (@tool) e a memória conversacional
(InMemoryChatMessageHistory) em um ciclo ReAct explícito:
    Thought (Raciocínio) -> Action (Ação) -> Action Input -> Observation (Observação)
até consolidar a Final Answer.

Nota técnica:
    Construído diretamente sobre `langchain_core` (PromptTemplate, Tools e
    ChatMessageHistory) para garantir compatibilidade total inclusive com
    Python 3.14+, evitando o módulo legado `langchain.chains.base`.
"""

import re
from typing import Any, Dict, List, Optional
from langchain_core.prompts import PromptTemplate

from src.config import get_llm
from src.memory import create_conversational_memory
from src.tools import get_all_tools

# Template ReAct padronizado em português com suporte explícito à memória conversacional
REACT_PROMPT_TEMPLATE = """Você é o ObraSmart, um engenheiro assistente especialista em planejamento de obras residenciais, dimensionamento de materiais, logística climática e custos de construção.
Responda sempre em português brasileiro de forma clara, técnica e objetiva.

Você tem acesso às seguintes ferramentas:
{tools}

Use OBRIGATORIAMENTE o seguinte formato ReAct passo a passo:

Question: a pergunta de entrada que você deve responder
Thought: reflita sobre o que precisa fazer, verificando se o histórico da conversa já contém informações úteis (como cidade, CEP ou área)
Action: o nome exato de UMA ferramenta dentre [{tool_names}]
Action Input: a entrada exata para a ferramenta escolhida (em uma única linha, sem explicações extras)
Observation: o resultado retornado pela ferramenta
... (repita Thought/Action/Action Input/Observation quantas vezes forem necessárias, SEMPRE uma ação por vez)
Thought: agora tenho todas as informações necessárias para formular a resposta final
Final Answer: a resposta final completa e organizada para o usuário

Regras críticas:
1. NUNCA escreva "Observation:" por conta própria; pare imediatamente após escrever "Action Input:" e aguarde o sistema executar a ferramenta.
2. Nunca invente endereços, previsões climáticas, cotações de moedas ou cálculos técnicos; use sempre as ferramentas disponíveis.
3. Se uma ferramenta retornar uma mensagem iniciada por "ERRO_", analise o erro no próximo Thought e explique ao usuário de forma clara como corrigir, sem inventar dados.
4. Utilize o histórico da conversa abaixo para resolver referências como "na mesma cidade", "a mesma área", "o serviço anterior", etc.

Histórico da conversa:
{chat_history}

Comece!

Question: {input}
{agent_scratchpad}"""


def _format_tools_description(tools) -> str:
    """Formata o catálogo de nomes e descrições das ferramentas para o prompt ReAct."""
    linhas = []
    for t in tools:
        desc = " ".join((t.description or "").split())
        linhas.append(f"- {t.name}: {desc}")
    return "\n".join(linhas)


def _extract_text_content(response: Any) -> str:
    """Extrai texto limpo da resposta retornada por diferentes LLMs no LangChain."""
    content = getattr(response, "content", response)
    if isinstance(content, list):
        partes = []
        for item in content:
            if isinstance(item, dict) and "text" in item:
                partes.append(str(item["text"]))
            else:
                partes.append(str(item))
        return "".join(partes).strip()
    return str(content).strip()


class ObraSmartAgent:
    """Agente ReAct modular com memória conversacional e execução iterativa de ferramentas."""

    def __init__(
        self,
        provider: str = "gemini",
        api_key: Optional[str] = None,
        model_name: Optional[str] = None,
        window_size: int = 8,
        max_iterations: int = 8,
    ):
        self.provider = provider
        self.model_name = model_name
        self.llm = get_llm(provider=provider, api_key=api_key, model_name=model_name)
        self.tools = get_all_tools()
        self.tool_map = {t.name: t for t in self.tools}
        self.memory = create_conversational_memory(window_size=window_size)
        self.prompt = PromptTemplate.from_template(REACT_PROMPT_TEMPLATE)
        self.max_iterations = max_iterations

        # Descrições pré-calculadas para injeção no template ReAct
        self._tools_text = _format_tools_description(self.tools)
        self._tool_names = ", ".join(self.tool_map.keys())

    def _parse_react_step(self, text: str) -> Dict[str, Optional[str]]:
        """
        Interpreta a saída textual do LLM no padrão ReAct:
        - Se houver `Final Answer:`, encerra o ciclo.
        - Se houver `Action:` e `Action Input:`, extrai a ferramenta e o argumento.
        """
        final_match = re.search(r"Final Answer\s*:(.*)", text, flags=re.DOTALL | re.IGNORECASE)
        if final_match:
            thought_part = text[: final_match.start()].strip()
            thought_part = re.sub(r"^Thought\s*:\s*", "", thought_part, flags=re.IGNORECASE).strip()
            return {
                "type": "final",
                "thought": thought_part,
                "final_answer": final_match.group(1).strip(),
            }

        action_match = re.search(
            r"Action\s*:\s*([^\n]+)\n\s*Action Input\s*:\s*([^\n]+)",
            text,
            flags=re.IGNORECASE,
        )
        if action_match:
            thought_part = text[: action_match.start()].strip()
            thought_part = re.sub(r"^Thought\s*:\s*", "", thought_part, flags=re.IGNORECASE).strip()
            tool_name = action_match.group(1).strip().strip("`").strip('"').strip("'")
            tool_input = action_match.group(2).strip().strip("`")
            return {
                "type": "action",
                "thought": thought_part,
                "tool": tool_name,
                "tool_input": tool_input,
            }

        return {"type": "invalid", "raw": text}

    def run(self, user_input: str) -> Dict[str, Any]:
        """
        Executa o loop ReAct (Thought -> Action -> Observation) até obter a Final Answer.

        Retorna:
            dict:
                - output (str): Resposta final consolidada.
                - steps (List[Dict]): Trilha completa de raciocínio e chamadas às ferramentas.
                - error (Optional[str]): Detalhe técnico em caso de erro.
        """
        if not user_input or not user_input.strip():
            return {
                "output": "Por favor, digite uma pergunta ou solicitação válida.",
                "steps": [],
                "error": "Entrada vazia.",
            }

        pergunta = user_input.strip()
        chat_history = self.memory.format_history()
        scratchpad = "Thought:"
        steps: List[Dict[str, str]] = []

        try:
            for _ in range(self.max_iterations):
                prompt_renderizado = self.prompt.format(
                    tools=self._tools_text,
                    tool_names=self._tool_names,
                    chat_history=chat_history,
                    input=pergunta,
                    agent_scratchpad=scratchpad,
                )

                # Stop sequence impede que o modelo alucine a Observation antes da ferramenta rodar
                try:
                    llm_response = self.llm.invoke(prompt_renderizado, stop=["Observation:"])
                except TypeError:
                    llm_response = self.llm.invoke(prompt_renderizado)

                texto_modelo = _extract_text_content(llm_response)
                # Garante corte caso o provedor ignore a stop sequence
                if "Observation:" in texto_modelo:
                    texto_modelo = texto_modelo.split("Observation:")[0].strip()

                parsed = self._parse_react_step(texto_modelo)

                if parsed["type"] == "final":
                    resposta_final = parsed["final_answer"] or texto_modelo
                    self.memory.add_exchange(pergunta, resposta_final)
                    return {
                        "output": resposta_final,
                        "steps": steps,
                        "error": None,
                    }

                if parsed["type"] == "action":
                    tool_name = parsed["tool"] or ""
                    tool_input = parsed["tool_input"] or ""
                    thought = parsed["thought"] or ""

                    if tool_name in self.tool_map:
                        try:
                            observation = str(self.tool_map[tool_name].invoke(tool_input))
                        except Exception as tool_exc:
                            observation = f"ERRO_EXECUCAO_FERRAMENTA: {tool_exc}"
                    else:
                        observation = (
                            f"ERRO_FERRAMENTA_DESCONHECIDA: '{tool_name}' não existe. "
                            f"Escolha exatamente uma dentre: [{self._tool_names}]."
                        )

                    steps.append(
                        {
                            "thought": thought,
                            "tool": tool_name,
                            "tool_input": tool_input,
                            "observation": observation,
                        }
                    )

                    # Atualiza o scratchpad com o passo completo para a próxima iteração ReAct
                    scratchpad += (
                        f" {thought}\n"
                        f"Action: {tool_name}\n"
                        f"Action Input: {tool_input}\n"
                        f"Observation: {observation}\n"
                        f"Thought:"
                    )
                else:
                    # Recuperação automática de erro de formatação (equivalente ao handle_parsing_errors)
                    if not steps:
                        # Se o modelo respondeu diretamente sem precisar de ferramentas
                        self.memory.add_exchange(pergunta, texto_modelo)
                        return {
                            "output": texto_modelo,
                            "steps": steps,
                            "error": None,
                        }

                    scratchpad += (
                        f" {texto_modelo}\n"
                        "Observation: ERRO_FORMATO_REACT: Responda seguindo exatamente "
                        "'Action: <ferramenta>\nAction Input: <entrada>' ou 'Final Answer: <resposta>'.\n"
                        "Thought:"
                    )

            # Caso atinja o limite de iterações sem Final Answer
            fallback = (
                "Atingi o limite máximo de etapas de raciocínio para esta consulta. "
                "Tente dividir a pergunta em partes menores."
            )
            self.memory.add_exchange(pergunta, fallback)
            return {
                "output": fallback,
                "steps": steps,
                "error": "Limite máximo de iterações ReAct atingido.",
            }

        except Exception as exc:
            return {
                "output": (
                    "Ocorreu um erro durante o processamento pelo agente. "
                    "Verifique sua chave de API, a conexão com a internet ou reformule a pergunta."
                ),
                "steps": steps,
                "error": str(exc),
            }

    def clear_memory(self) -> None:
        """Limpa o histórico conversacional da sessão atual."""
        self.memory.clear()
