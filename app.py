"""
Interface Web em Streamlit para o ObraSmart ReAct Agent.

Permite conversar com o agente inteligente, inspecionar cada etapa do ciclo
ReAct (Thought -> Action -> Observation), testar casos de uso pré-configurados
e gerenciar a memória conversacional da sessão.
"""

import os
import streamlit as st
from src.agent import ObraSmartAgent
from src.config import DEFAULT_MODELS
from src.tools import get_all_tools

st.set_page_config(
    page_title="ObraSmart — Agente ReAct de Obras",
    page_icon="🏗️",
    layout="wide",
)

st.title("🏗️ ObraSmart — Agente ReAct de Planejamento de Obras")
st.caption(
    "LangChain + Framework ReAct + Memória Conversacional + 4 Ferramentas Integradas "
    "(ViaCEP, Open-Meteo, Calculadora Técnica e AwesomeAPI)"
)

# --- Barra Lateral: Configurações e Casos de Uso ---
with st.sidebar:
    st.header("⚙️ Configuração do LLM")
    provider = st.selectbox(
        "Provedor de LLM",
        options=["gemini", "openai"],
        format_func=lambda x: "Google Gemini" if x == "gemini" else "OpenAI GPT",
    )

    model_name = st.text_input(
        "Modelo",
        value=DEFAULT_MODELS[provider],
        help="Padrão atualizado: gemini-3.8-flash (Gemini) ou gpt-4o-mini (OpenAI).",
    )

    env_var = "GOOGLE_API_KEY" if provider == "gemini" else "OPENAI_API_KEY"
    default_key = os.getenv(env_var, "")
    api_key = st.text_input(
        f"Chave de API ({env_var})",
        value=default_key if "sua_chave" not in default_key else "",
        type="password",
        help="Você pode definir no arquivo .env ou colar diretamente aqui.",
    )

    if st.button("🔄 Inicializar / Reiniciar Agente", use_container_width=True):
        try:
            st.session_state.agent = ObraSmartAgent(
                provider=provider,
                api_key=api_key,
                model_name=model_name,
            )
            st.session_state.messages = []
            st.success(f"Agente inicializado com {provider.upper()} ({model_name})!")
        except Exception as exc:
            st.error(f"Falha ao inicializar agente: {exc}")

    if st.button("🧹 Limpar Memória Conversacional", use_container_width=True):
        if "agent" in st.session_state:
            st.session_state.agent.clear_memory()
        st.session_state.messages = []
        st.info("Histórico e memória conversacional limpos.")

    st.divider()
    st.subheader("🛠️ Ferramentas Ativas")
    for t in get_all_tools():
        st.markdown(f"- **`{t.name}`**")

    st.divider()
    st.subheader("🧪 Casos de Uso Rápidos")
    st.caption("Clique para enviar os cenários de teste documentados:")

    caso_1 = (
        "Vou construir um muro de alvenaria de 45 m² no CEP 70040-010. "
        "Qual cidade é essa, qual a previsão de chuva para amanhã lá (dá para trabalhar?) "
        "e quais materiais vou precisar com custo estimado?"
    )
    caso_2 = (
        "E se eu fizer pintura na mesma área que mencionei antes lá naquela cidade? "
        "Além disso, quero importar um impermeabilizante especial que custa 200 dólares: "
        "quanto ele sai em reais na cotação de hoje?"
    )
    caso_3 = (
        "Consulte o CEP 00000-000 e calcule o material para 50 m² de telhado espacial."
    )

    prompt_selecionado = None
    if st.button("Caso 1: Multi-Tool ReAct (CEP + Clima + Alvenaria)", use_container_width=True):
        prompt_selecionado = caso_1
    if st.button("Caso 2: Memória + Pintura + Cotação USD", use_container_width=True):
        prompt_selecionado = caso_2
    if st.button("Caso 3: Robustez e Tratamento de Erros", use_container_width=True):
        prompt_selecionado = caso_3

# --- Inicialização automática se houver chave no ambiente ---
if "messages" not in st.session_state:
    st.session_state.messages = []

if "agent" not in st.session_state and api_key:
    try:
        st.session_state.agent = ObraSmartAgent(
            provider=provider,
            api_key=api_key,
            model_name=model_name,
        )
    except Exception:
        pass

if "agent" not in st.session_state:
    st.warning(
        "Informe sua chave de API do Google Gemini ou OpenAI na barra lateral "
        "e clique em **Inicializar / Reiniciar Agente** para começar."
    )

# --- Exibição do histórico de conversa ---
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("steps"):
            with st.expander(f"🧠 Ver Raciocínio ReAct ({len(msg['steps'])} etapa(s))"):
                for idx, step in enumerate(msg["steps"], start=1):
                    st.markdown(f"**Etapa {idx} — Ferramenta:** `{step['tool']}`")
                    if step.get("thought"):
                        st.markdown(f"- **Thought:** {step['thought']}")
                    st.markdown(f"- **Action Input:** `{step['tool_input']}`")
                    st.code(step["observation"], language="text")

# --- Entrada do usuário ---
user_input = st.chat_input("Pergunte sobre CEP, clima da obra, cálculo de materiais ou cotação...")
if prompt_selecionado:
    user_input = prompt_selecionado

if user_input:
    if "agent" not in st.session_state:
        st.error("Inicialize o agente na barra lateral antes de enviar mensagens.")
    else:
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):
            with st.spinner("Agente raciocinando (ReAct: Thought → Action → Observation)..."):
                resultado = st.session_state.agent.run(user_input)

            st.markdown(resultado["output"])
            if resultado.get("error"):
                st.error(f"Detalhes técnicos: {resultado['error']}")

            if resultado.get("steps"):
                with st.expander(f"🧠 Ver Raciocínio ReAct ({len(resultado['steps'])} etapa(s))"):
                    for idx, step in enumerate(resultado["steps"], start=1):
                        st.markdown(f"**Etapa {idx} — Ferramenta:** `{step['tool']}`")
                        if step.get("thought"):
                            st.markdown(f"- **Thought:** {step['thought']}")
                        st.markdown(f"- **Action Input:** `{step['tool_input']}`")
                        st.code(step["observation"], language="text")

            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": resultado["output"],
                    "steps": resultado.get("steps", []),
                }
            )
