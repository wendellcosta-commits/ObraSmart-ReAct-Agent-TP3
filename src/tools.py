"""
Módulo de Ferramentas Externas (Tools) integradas ao LangChain.

Define quatro ferramentas com nomes, descrições claras (essenciais para o
raciocínio ReAct do LLM decidir qual ação tomar) e tratamento robusto de erros:
1. consultar_cep: API externa ViaCEP para localizar o endereço da obra.
2. consultar_clima_obra: API externa Open-Meteo para previsão meteorológica e impacto na obra.
3. calcular_materiais_construcao: Motor técnico de quantitativos com margem de perda e custo estimado.
4. consultar_cotacao_moeda: API externa AwesomeAPI para conversão de insumos importados em BRL.
"""

import json
import re
import requests
from langchain_core.tools import tool

# Timeout padrão (em segundos) para evitar travamento em requisições HTTP externas
HTTP_TIMEOUT = 10


def _limpar_texto(valor: str) -> str:
    """Remove aspas extras e espaços laterais que LLMs costumam enviar no Action Input."""
    return valor.strip().strip('"').strip("'").strip()


@tool
def consultar_cep(cep: str) -> str:
    """
    Consulta informações de endereço (logradouro, bairro, cidade e estado) de um CEP brasileiro
    usando a API pública ViaCEP.
    Use esta ferramenta sempre que o usuário informar um CEP de obra e você precisar descobrir
    a cidade ou o endereço completo.
    Entrada esperada: apenas os 8 dígitos do CEP (ex: "70040010" ou "70040-010").
    """
    try:
        cep_digitos = re.sub(r"\D", "", _limpar_texto(cep))
        if len(cep_digitos) != 8:
            return (
                f"ERRO_VALIDACAO: O CEP '{cep}' é inválido. "
                "Informe exatamente 8 dígitos numéricos (exemplo: 70040-010)."
            )

        url = f"https://viacep.com.br/ws/{cep_digitos}/json/"
        resp = requests.get(url, timeout=HTTP_TIMEOUT)
        resp.raise_for_status()
        dados = resp.json()

        if dados.get("erro"):
            return f"ERRO_NAO_ENCONTRADO: O CEP {cep_digitos} não foi localizado na base do ViaCEP."

        logradouro = dados.get("logradouro") or "Não informado"
        bairro = dados.get("bairro") or "Não informado"
        cidade = dados.get("localidade") or "Não informada"
        uf = dados.get("uf") or "--"

        return (
            f"Endereço localizado para o CEP {cep_digitos}: "
            f"Logradouro: {logradouro} | Bairro: {bairro} | Cidade: {cidade} | UF: {uf}."
        )

    except requests.exceptions.Timeout:
        return "ERRO_CONEXAO: Tempo limite excedido ao consultar a API ViaCEP. Tente novamente."
    except requests.exceptions.RequestException as exc:
        return f"ERRO_CONEXAO: Falha na comunicação com o serviço ViaCEP ({exc})."
    except Exception as exc:
        return f"ERRO_INESPERADO: Não foi possível consultar o CEP ({exc})."


@tool
def consultar_clima_obra(cidade: str) -> str:
    """
    Consulta a previsão meteorológica em tempo real (hoje e amanhã) para uma cidade usando
    a API pública Open-Meteo e avalia a viabilidade climática para serviços externos de obra
    (como concretagem, reboco e pintura externa).
    Use esta ferramenta quando precisar verificar chuva, temperatura ou vento no local da obra.
    Entrada esperada: nome da cidade (ex: "Brasília", "Rio de Janeiro", "Curitiba").
    """
    try:
        nome_cidade = _limpar_texto(cidade)
        if not nome_cidade:
            return "ERRO_VALIDACAO: Informe o nome de uma cidade válida para consultar o clima."

        # Etapa 1: Geocodificação do nome da cidade para latitude e longitude
        geo_url = "https://geocoding-api.open-meteo.com/v1/search"
        geo_resp = requests.get(
            geo_url,
            params={"name": nome_cidade, "count": 1, "language": "pt", "format": "json"},
            timeout=HTTP_TIMEOUT,
        )
        geo_resp.raise_for_status()
        geo_data = geo_resp.json()

        resultados = geo_data.get("results")
        if not resultados:
            return f"ERRO_NAO_ENCONTRADO: Não foi possível localizar as coordenadas da cidade '{nome_cidade}'."

        local = resultados[0]
        lat = local["latitude"]
        lon = local["longitude"]
        cidade_resolvida = local.get("name", nome_cidade)
        estado_resolvido = local.get("admin1", "")

        # Etapa 2: Consulta da previsão diária (hoje e amanhã)
        weather_url = "https://api.open-meteo.com/v1/forecast"
        weather_resp = requests.get(
            weather_url,
            params={
                "latitude": lat,
                "longitude": lon,
                "daily": "temperature_2m_max,temperature_2m_min,precipitation_sum,precipitation_probability_max,wind_speed_10m_max",
                "timezone": "America/Sao_Paulo",
                "forecast_days": 2,
            },
            timeout=HTTP_TIMEOUT,
        )
        weather_resp.raise_for_status()
        w_data = weather_resp.json().get("daily", {})

        datas = w_data.get("time", [])
        t_max = w_data.get("temperature_2m_max", [])
        t_min = w_data.get("temperature_2m_min", [])
        chuva_mm = w_data.get("precipitation_sum", [])
        prob_chuva = w_data.get("precipitation_probability_max", [])
        vento_max = w_data.get("wind_speed_10m_max", [])

        linhas = [f"Previsão climática para {cidade_resolvida} ({estado_resolvido}):"]
        for i in range(min(2, len(datas))):
            rotulo = "Hoje" if i == 0 else "Amanhã"
            prob = prob_chuva[i] if i < len(prob_chuva) and prob_chuva[i] is not None else 0
            mm = chuva_mm[i] if i < len(chuva_mm) and chuva_mm[i] is not None else 0.0

            if prob >= 60 or mm >= 5.0:
                parecer = "RISCO ALTO DE CHUVA — evitar pintura externa e concretagem descoberta."
            elif prob >= 30 or mm > 0.5:
                parecer = "ATENÇÃO — possibilidade de chuva leve; manter lonas de proteção disponíveis."
            else:
                parecer = "FAVORÁVEL — boas condições para serviços externos e concretagem."

            linhas.append(
                f"- {rotulo} ({datas[i]}): Min {t_min[i]}°C / Max {t_max[i]}°C | "
                f"Chuva: {mm} mm (prob. {prob}%) | Vento máx: {vento_max[i]} km/h | Parecer: {parecer}"
            )

        return "\n".join(linhas)

    except requests.exceptions.Timeout:
        return "ERRO_CONEXAO: Tempo limite excedido ao consultar a API meteorológica Open-Meteo."
    except requests.exceptions.RequestException as exc:
        return f"ERRO_CONEXAO: Falha ao acessar o serviço meteorológico ({exc})."
    except Exception as exc:
        return f"ERRO_INESPERADO: Erro ao processar dados climáticos ({exc})."


@tool
def calcular_materiais_construcao(parametros: str) -> str:
    """
    Calcula o quantitativo técnico de materiais (com margem de perda de 10%) e o custo médio
    estimado em reais (BRL) para etapas de obra residencial.
    Serviços suportados: 'alvenaria' (m²), 'piso' (m²), 'pintura' (m²), 'concreto' (m³).
    Entrada esperada: texto no formato "servico, quantidade" (ex: "alvenaria, 45" ou "pintura, 60").
    """
    try:
        entrada = _limpar_texto(parametros)

        # Permite tanto formato JSON {"servico": "...", "quantidade": ...} quanto "servico, quantidade"
        servico = ""
        quantidade = 0.0

        if entrada.startswith("{") and entrada.endswith("}"):
            obj = json.loads(entrada)
            servico = str(obj.get("servico", "")).strip().lower()
            quantidade = float(obj.get("quantidade", 0))
        else:
            partes = [p.strip() for p in re.split(r"[,;|]", entrada) if p.strip()]
            if len(partes) < 2:
                return (
                    "ERRO_FORMATO: Informe no formato 'servico, quantidade'. "
                    "Exemplo: 'alvenaria, 45'. Serviços suportados: alvenaria, piso, pintura, concreto."
                )
            servico = partes[0].lower()
            num_match = re.search(r"\d+(?:[.,]\d+)?", partes[1])
            if not num_match:
                return f"ERRO_VALIDACAO: Não foi possível identificar um número válido em '{partes[1]}'."
            quantidade = float(num_match.group(0).replace(",", "."))

        if quantidade <= 0:
            return "ERRO_VALIDACAO: A quantidade (área ou volume) deve ser maior que zero."

        fator_perda = 1.10  # Margem técnica padrão de 10% para quebras e recortes

        # Tabelas técnicas de referência (valores médios de mercado brasileiro em BRL)
        if "alvenaria" in servico or "muro" in servico or "parede" in servico:
            area_total = quantidade * fator_perda
            blocos = int(round(area_total * 25))       # ~25 blocos cerâmicos (9x19x19) por m²
            argamassa_kg = round(area_total * 18, 1)   # ~18 kg/m²
            sacos_cimento = int(round((argamassa_kg / 5) / 50)) or 1
            custo_estimado = round(blocos * 1.35 + sacos_cimento * 38.00 + area_total * 12.00, 2)
            return (
                f"Cálculo Técnico — Alvenaria ({quantidade:.2f} m² + 10% perda = {area_total:.2f} m²):\n"
                f"- Blocos cerâmicos (9x19x19 cm): {blocos} unidades\n"
                f"- Argamassa de assentamento: {argamassa_kg} kg (aprox. {sacos_cimento} sacos de cimento 50kg + areia/cal)\n"
                f"- Custo médio estimado de materiais: R$ {custo_estimado:,.2f}"
            )

        elif "piso" in servico or "porcelanato" in servico or "revestimento" in servico:
            area_total = quantidade * fator_perda
            argamassa_sacos = max(1, int(round(area_total / 4.0)))  # 1 saco 20kg rende ~4 m²
            rejunte_kg = round(area_total * 0.4, 1)
            custo_estimado = round(area_total * 75.00 + argamassa_sacos * 32.00 + rejunte_kg * 12.00, 2)
            return (
                f"Cálculo Técnico — Piso/Revestimento ({quantidade:.2f} m² + 10% perda = {area_total:.2f} m²):\n"
                f"- Piso/Porcelanato necessário: {area_total:.2f} m²\n"
                f"- Argamassa colante AC-II/AC-III: {argamassa_sacos} sacos de 20 kg\n"
                f"- Rejunte flexível: {rejunte_kg} kg\n"
                f"- Custo médio estimado de materiais: R$ {custo_estimado:,.2f}"
            )

        elif "pintura" in servico or "tinta" in servico:
            # Considera 2 demãos + 10% de perda; rendimento médio de 10 m²/litro por demão
            area_total = quantidade * 2 * fator_perda
            litros_tinta = round(area_total / 10.0, 1)
            latas_18l = int(litros_tinta // 18)
            galoes_36l = max(1, int(round((litros_tinta % 18) / 3.6))) if (litros_tinta % 18) > 0 else 0
            selador_litros = round((quantidade * fator_perda) / 12.0, 1)
            custo_estimado = round(litros_tinta * 28.00 + selador_litros * 16.00, 2)
            return (
                f"Cálculo Técnico — Pintura ({quantidade:.2f} m², 2 demãos + 10% margem):\n"
                f"- Tinta acrílica necessária: {litros_tinta} litros "
                f"(sugestão: {latas_18l} lata(s) de 18L + {galoes_36l} galão(ões) de 3,6L)\n"
                f"- Selador acrílico: {selador_litros} litros\n"
                f"- Custo médio estimado de materiais: R$ {custo_estimado:,.2f}"
            )

        elif "concreto" in servico or "laje" in servico:
            vol_total = quantidade * fator_perda
            sacos_cimento = int(round(vol_total * 7))   # ~7 sacos de 50kg por m³ (FCK 25 MPa)
            areia_m3 = round(vol_total * 0.55, 2)
            brita_m3 = round(vol_total * 0.80, 2)
            custo_estimado = round(sacos_cimento * 38.00 + areia_m3 * 130.00 + brita_m3 * 145.00, 2)
            return (
                f"Cálculo Técnico — Concreto FCK 25 MPa ({quantidade:.2f} m³ + 10% perda = {vol_total:.2f} m³):\n"
                f"- Cimento CP-II (50 kg): {sacos_cimento} sacos\n"
                f"- Areia média: {areia_m3} m³\n"
                f"- Brita 1: {brita_m3} m³\n"
                f"- Custo médio estimado de materiais: R$ {custo_estimado:,.2f}"
            )

        else:
            return (
                f"ERRO_SERVICO_NAO_SUPORTADO: O serviço '{servico}' não está cadastrado. "
                "Serviços disponíveis: alvenaria (m²), piso (m²), pintura (m²) e concreto (m³)."
            )

    except Exception as exc:
        return f"ERRO_CALCULO: Falha ao calcular materiais ({exc})."


@tool
def consultar_cotacao_moeda(moeda: str) -> str:
    """
    Consulta a cotação em tempo real de uma moeda estrangeira em relação ao Real Brasileiro (BRL)
    usando a API pública AwesomeAPI.
    Use esta ferramenta quando o usuário quiser converter preços de ferramentas, revestimentos
    ou insumos importados (em USD, EUR ou GBP) para reais.
    Entrada esperada: código ISO de 3 letras da moeda (ex: "USD", "EUR", "GBP").
    """
    try:
        codigo = _limpar_texto(moeda).upper()
        # Aceita nomes comuns convertendo para código ISO
        mapa_nomes = {"DOLAR": "USD", "DÓLAR": "USD", "EURO": "EUR", "LIBRA": "GBP"}
        codigo = mapa_nomes.get(codigo, codigo)

        if not re.fullmatch(r"[A-Z]{3}", codigo):
            return (
                f"ERRO_VALIDACAO: Código de moeda '{moeda}' inválido. "
                "Informe um código de 3 letras como USD, EUR ou GBP."
            )

        url = f"https://economia.awesomeapi.com.br/json/last/{codigo}-BRL"
        resp = requests.get(url, timeout=HTTP_TIMEOUT)
        if resp.status_code == 404:
            return f"ERRO_NAO_ENCONTRADO: Cotação para '{codigo}-BRL' não encontrada na AwesomeAPI."
        resp.raise_for_status()

        dados = resp.json()
        chave = f"{codigo}BRL"
        if chave not in dados:
            return f"ERRO_NAO_ENCONTRADO: Par de moedas {codigo}/BRL indisponível."

        info = dados[chave]
        compra = float(info["bid"])
        venda = float(info["ask"])
        variacao = info.get("pctChange", "0")
        atualizado = info.get("create_date", "N/A")

        return (
            f"Cotação atual {codigo}/BRL: Compra R$ {compra:.4f} | Venda R$ {venda:.4f} "
            f"| Variação: {variacao}% | Atualizado em: {atualizado}."
        )

    except requests.exceptions.Timeout:
        return "ERRO_CONEXAO: Tempo limite excedido ao consultar a AwesomeAPI."
    except requests.exceptions.RequestException as exc:
        return f"ERRO_CONEXAO: Erro de rede ao consultar cotação ({exc})."
    except Exception as exc:
        return f"ERRO_INESPERADO: Falha ao obter cotação da moeda ({exc})."


def get_all_tools():
    """Retorna a lista completa de ferramentas configuradas para o agente."""
    return [
        consultar_cep,
        consultar_clima_obra,
        calcular_materiais_construcao,
        consultar_cotacao_moeda,
    ]
