"""
Interface de Linha de Comando (CLI) para execução rápida do ObraSmart ReAct Agent.

Útil para testes diretos no terminal sem necessidade de abrir o navegador.
Uso: python cli.py
"""

import sys
from src.agent import ObraSmartAgent


def main():
    print("=" * 65)
    print("🏗️  ObraSmart — Agente ReAct de Planejamento de Obras (CLI)")
    print("=" * 65)
    print("Comandos: digite sua pergunta, 'limpar' para zerar memória ou 'sair'.\n")

    provider = input("Escolha o provedor [gemini/openai] (padrão: gemini): ").strip().lower() or "gemini"

    try:
        agent = ObraSmartAgent(provider=provider)
        print(f"\n✅ Agente inicializado com sucesso usando {provider.upper()}!\n")
    except Exception as exc:
        print(f"\n❌ Erro ao inicializar agente: {exc}")
        print("Configure GOOGLE_API_KEY ou OPENAI_API_KEY no arquivo .env e tente novamente.")
        sys.exit(1)

    while True:
        try:
            pergunta = input("\nVocê: ").strip()
            if not pergunta:
                continue
            if pergunta.lower() in {"sair", "exit", "quit"}:
                print("Encerrando ObraSmart. Até logo!")
                break
            if pergunta.lower() == "limpar":
                agent.clear_memory()
                print("🧹 Memória conversacional limpa.")
                continue

            resultado = agent.run(pergunta)
            print(f"\n🤖 ObraSmart:\n{resultado['output']}")

            if resultado.get("error"):
                print(f"⚠️  Detalhe do erro: {resultado['error']}")

        except KeyboardInterrupt:
            print("\nInterrompido pelo usuário. Encerrando...")
            break


if __name__ == "__main__":
    main()
