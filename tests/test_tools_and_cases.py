"""
Testes práticos das ferramentas externas, validações e tratamento de erros.

Execução:
    python tests/test_tools_and_cases.py
"""

import os
import sys

# Garante importação a partir da raiz do projeto
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.tools import (
    calcular_materiais_construcao,
    consultar_cep,
    consultar_clima_obra,
    consultar_cotacao_moeda,
)


def executar_testes():
    print("=== 1. Teste de Cálculo de Materiais (Alvenaria 45 m²) ===")
    res_alv = calcular_materiais_construcao.invoke("alvenaria, 45")
    print(res_alv)
    assert "Blocos cerâmicos" in res_alv, "Falha no cálculo de alvenaria"

    print("\n=== 2. Teste de Cálculo de Materiais (Pintura 45 m²) ===")
    res_pin = calcular_materiais_construcao.invoke("pintura, 45")
    print(res_pin)
    assert "Tinta acrílica" in res_pin, "Falha no cálculo de pintura"

    print("\n=== 3. Teste de Tratamento de Erro — Serviço Inexistente ===")
    res_err_serv = calcular_materiais_construcao.invoke("telhado espacial, 50")
    print(res_err_serv)
    assert "ERRO_SERVICO_NAO_SUPORTADO" in res_err_serv, "Erro não tratado corretamente"

    print("\n=== 4. Teste de Tratamento de Erro — CEP com formato inválido ===")
    res_err_cep = consultar_cep.invoke("123")
    print(res_err_cep)
    assert "ERRO_VALIDACAO" in res_err_cep, "Validação de CEP falhou"

    print("\n=== 5. Teste de Tratamento de Erro — Moeda inválida ===")
    res_err_moeda = consultar_cotacao_moeda.invoke("MOEDA123")
    print(res_err_moeda)
    assert "ERRO_VALIDACAO" in res_err_moeda, "Validação de moeda falhou"

    print("\n=== 6. Testes de APIs Externas Reais (requer acesso à internet) ===")
    print("[ViaCEP 70040-010]:", consultar_cep.invoke("70040-010"))
    print("[Open-Meteo Brasília]:", consultar_clima_obra.invoke("Brasília"))
    print("[AwesomeAPI USD]:", consultar_cotacao_moeda.invoke("USD"))

    print("\n✅ Todos os testes estruturais e de tratamento de erros foram concluídos!")


if __name__ == "__main__":
    executar_testes()
