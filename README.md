# 🏗️ ObraSmart — Agente Inteligente ReAct para Planejamento de Obras

**Disciplina:** Desenvolvimento de Data-Driven Apps com Python (DR3) — Teste de Performance 3 (TP3)  
**Instituição:** Instituto Infnet — Ciência de Dados e Inteligência Artificial  
**Aluno:** Wendell Barbosa Costa  

---

## 1. Definição do Problema, Objetivo e Público-Alvo

### Problema
No dia a dia de reformas e construções residenciais, a tomada de decisão rápida depende de cruzar dados heterogêneos que estão espalhados em fontes diferentes: identificar o endereço exato pelo CEP, checar a previsão climática local para evitar prejuízos em etapas sensíveis à chuva (concretagem, reboco, pintura externa), calcular o quantitativo técnico de materiais considerando perdas e converter preços de insumos ou equipamentos importados pela cotação do câmbio do dia.

### Objetivo da Aplicação
Desenvolver o **ObraSmart**, um agente inteligente baseado em **LLMs**, **LangChain (`langchain-core`)**, **framework ReAct (Reasoning + Acting)** e **memória conversacional**, capaz de decompor perguntas complexas em etapas, acionar ferramentas externas reais de forma autônoma, lembrar o contexto da conversa e entregar respostas técnicas consolidadas.

### Público-Alvo
- Pequenos construtores, mestres de obras e técnicos em edificações;
- Arquitetos e engenheiros autônomos;
- Proprietários planejando reformas residenciais.

### Funcionalidades Planejadas e Implementadas
1. **Raciocínio Interativo (ReAct):** Loop explícito `Thought → Action → Action Input → Observation → Final Answer`, com exibição transparente das etapas na interface Streamlit.
2. **Memória Conversacional:** `InMemoryChatMessageHistory` (`langchain_core.chat_history`) encapsulada com janela deslizante de 8 interações (`ConversationalWindowMemory`), permitindo perguntas de acompanhamento que referenciam dados anteriores (ex.: *"na mesma área"*, *"naquela cidade"*).
3. **4 Ferramentas Integradas (`src/tools.py`):**
   - `consultar_cep`: consulta a API pública **ViaCEP** para resolver logradouro, bairro, cidade e UF.
   - `consultar_clima_obra`: consulta a API pública **Open-Meteo** (Geocoding + Forecast) e emite parecer técnico para serviços externos.
   - `calcular_materiais_construcao`: calcula quantitativos técnicos (alvenaria, piso, pintura e concreto) com margem de perda de 10% e estimativa de custo em BRL.
   - `consultar_cotacao_moeda`: consulta a **AwesomeAPI** em tempo real (USD, EUR, GBP → BRL) para insumos importados.
4. **Duas Interfaces Funcionais:** Aplicação web interativa em **Streamlit (`app.py`)** e interface de terminal **CLI (`cli.py`)**.
5. **Tratamento Robusto de Erros:** Validação de entradas, timeouts HTTP, recuperação automática de erros de formatação no ciclo ReAct e mensagens claras quando APIs retornam códigos de erro ou dados inexistentes.

---

## 2. Arquitetura Modular do Projeto

```text
ObraSmart_ReAct_Agent/
├── app.py                        # Interface Web em Streamlit com chat e inspetor ReAct
├── cli.py                        # Interface alternativa em linha de comando (CLI)
├── requirements.txt              # Dependências compatíveis com Python 3.10 a 3.14
├── .env.example                  # Modelo de variáveis de ambiente
├── .gitignore                    # Arquivos ignorados no versionamento Git
├── README.md                     # Documentação completa do projeto
├── src/
│   ├── __init__.py
│   ├── config.py                 # Inicialização segura do LLM (Google Gemini / OpenAI)
│   ├── tools.py                  # 4 ferramentas LangChain (@tool) com tratamento de erros
│   ├── memory.py                 # Memória conversacional (InMemoryChatMessageHistory + janela)
│   └── agent.py                  # Prompt ReAct e loop iterativo Thought -> Action -> Observation
└── tests/
    └── test_tools_and_cases.py   # Script de testes práticos das ferramentas e validações
```

---

## 3. Casos de Uso Testados e Resultados Observados

### Caso de Uso 1 — Planejamento Integrado com Múltiplas Ferramentas (ReAct Multi-Step)
- **Entrada:** *"Vou construir um muro de alvenaria de 45 m² no CEP 70040-010. Qual cidade é essa, qual a previsão de chuva para amanhã lá (dá para trabalhar?) e quais materiais vou precisar com custo estimado?"*
- **Fluxo ReAct Executado pelo Agente:**
  1. `Thought`: Preciso primeiro descobrir a cidade correspondente ao CEP informado.
     `Action`: `consultar_cep` → `Action Input`: `70040-010`
     `Observation`: *Endereço localizado para o CEP 70040010: Logradouro: SBN Quadra 1 | Bairro: Asa Norte | Cidade: Brasília | UF: DF.*
  2. `Thought`: Agora que sei que a obra é em Brasília, preciso verificar a previsão do tempo para amanhã.
     `Action`: `consultar_clima_obra` → `Action Input`: `Brasília`
     `Observation`: *Dados reais de temperatura mínima/máxima, precipitação (mm), probabilidade de chuva (%) e parecer técnico da obra.*
  3. `Thought`: Agora preciso calcular os materiais para 45 m² de alvenaria.
     `Action`: `calcular_materiais_construcao` → `Action Input`: `alvenaria, 45`
     `Observation`: *49,50 m² (com 10% de perda): 1.238 blocos cerâmicos, 891,0 kg de argamassa (~4 sacos de cimento 50 kg) e custo médio estimado de R$ 2.417,30.*
  4. `Final Answer`: Resposta estruturada consolidando local, viabilidade climática e lista de compras.

### Caso de Uso 2 — Uso de Memória Conversacional + Conversão Cambial
- **Entrada (enviada logo após o Caso 1):** *"E se eu fizer pintura na mesma área que mencionei antes lá naquela cidade? Além disso, quero importar um impermeabilizante especial que custa 200 dólares: quanto ele sai em reais na cotação de hoje?"*
- **Comportamento Observado:**
  - Sem que o usuário precisasse repetir "45 m²" ou "Brasília", o agente recuperou ambos os dados da memória conversacional.
  - Acionou `calcular_materiais_construcao("pintura, 45")` → *9,9 litros de tinta (2 demãos + 10% perda), 4,1 litros de selador, custo médio de R$ 342,80*.
  - Acionou `consultar_cotacao_moeda("USD")` → obteve a cotação atual do dólar na AwesomeAPI e multiplicou por US$ 200,00 para entregar o valor exato em reais.

### Caso de Uso 3 — Robustez e Tratamento de Erros
- **Entrada:** *"Consulte o CEP 00000-000 e calcule o material para 50 m² de telhado espacial."*
- **Comportamento Observado:**
  - `consultar_cep` retornou `ERRO_NAO_ENCONTRADO` de forma controlada (sem lançar exceção não tratada).
  - `calcular_materiais_construcao` retornou `ERRO_SERVICO_NAO_SUPORTADO` listando os serviços válidos (`alvenaria`, `piso`, `pintura`, `concreto`).
  - O agente leu ambas as observações no ciclo ReAct e orientou o usuário educadamente sem alucinar números ou derrubar a aplicação.

---

## 4. Instruções Detalhadas para Execução do Código

### Pré-requisitos
- Python 3.10 ou superior (testado inclusive em Python 3.14);
- Uma chave de API do **Google Gemini** (gratuita em [aistudio.google.com](https://aistudio.google.com/)) ou da **OpenAI**.

### Passo a Passo

1. **Clone o repositório:**
   ```bash
   git clone https://github.com/wendellcosta-commits/ObraSmart-ReAct-Agent-TP3.git
   cd ObraSmart-ReAct-Agent-TP3
   ```

2. **Crie e ative um ambiente virtual:**
   ```bash
   # Windows (PowerShell)
   python -m venv .venv
   .venv\Scripts\Activate.ps1

   # Windows (Git Bash)
   python -m venv .venv
   source .venv/Scripts/activate

   # Linux / macOS
   python3 -m venv .venv
   source .venv/bin/activate
   ```

3. **Instale as dependências:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure a chave de API:**
   - Copie o arquivo `.env.example` para `.env`:
     ```bash
     cp .env.example .env
     ```
   - Abra o arquivo `.env` e preencha `GOOGLE_API_KEY` (ou `OPENAI_API_KEY`).
   *(Opcionalmente, você também pode colar a chave diretamente na barra lateral do Streamlit.)*

5. **Execute os testes unitários das ferramentas:**
   ```bash
   python tests/test_tools_and_cases.py
   ```

6. **Execute a aplicação Web (Streamlit):**
   ```bash
   streamlit run app.py
   ```
   Acesse no navegador o endereço exibido (geralmente `http://localhost:8501`).

7. **Alternativa via Terminal (CLI):**
   ```bash
   python cli.py
   ```

---

## 5. Reflexão: Agente ReAct vs. Abordagem Manual

| Dimensão | Abordagem Manual Tradicional | Com o Agente ObraSmart (ReAct + Memória) |
| :--- | :--- | :--- |
| **Tempo de Execução** | Exige abrir 4 ferramentas distintas (site dos Correios, portal do clima, planilha de cálculo de obra e site de câmbio) e cruzar manualmente os valores (10 a 15 minutos por consulta). | Uma única pergunta em linguagem natural orquestra todas as chamadas em sequência autônoma e devolve a síntese em poucos segundos. |
| **Encadeamento de Contexto** | A cada mudança de cenário (trocar alvenaria por pintura ou testar outra metragem), o usuário precisa refazer contas e lembrar premissas anteriores. | A memória conversacional retém área, CEP e cidade, permitindo simulações incrementais naturais. |
| **Precisão e Confiabilidade** | Risco de erro humano em fórmulas de margem de perda ou uso de cotações desatualizadas. | O LLM não "chuta" cálculos nem previsões: delega dados dinâmicos e contas determinísticas às ferramentas, reduzindo drasticamente alucinações. |
| **Escalabilidade** | Difícil de padronizar entre diferentes membros de uma equipe de obras. | Basta adicionar novas `@tool` (ex.: tabela SINAPI ou estoque próprio) e o agente passa a utilizá-las automaticamente. |

---

## 6. Declaração de Uso de Inteligência Artificial (Sinal Verde 🟢)

Em conformidade com as diretrizes de integridade acadêmica do Instituto Infnet e a Portaria CNPq sobre transparência no uso de IA:
- A ferramenta de Inteligência Artificial Generativa **Google Gemini** foi utilizada como assistente de desenvolvimento para estruturação modular do projeto, revisão do template de prompt ReAct, boas práticas de tratamento de erros e apoio na redação da documentação.
- A aplicação utiliza LLMs via **LangChain** (`gemini-3.8-flash` / `gpt-4o-mini`) como motor de raciocínio em tempo de execução.
- Todo o código, arquitetura, casos de uso e documentação foram revisados, validados e assumidos integralmente pelo aluno **Wendell Barbosa Costa**.
