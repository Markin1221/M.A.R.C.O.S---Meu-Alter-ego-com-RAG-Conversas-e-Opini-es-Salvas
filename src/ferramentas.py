"""Ferramentas do MARCOS: busca na web e memória das conversas com ele."""
import json
from datetime import datetime

import ollama

from buscar import EU, PASTA

PASTA_CHAT = PASTA / "chat"
ARQUIVO_LEMBRETES = PASTA / "lembretes.txt"
NOME_CONVERSA_CHAT = "MARCOS (chat)"
AUTOR_MARCOS = "MARCOS"

MAX_RESULTADOS_WEB = 3
MAX_CARACTERES_RESULTADO = 500
MAX_LEMBRETES = 15


FORMATO_DECISAO = {
    "type": "object",
    "properties": {"buscar": {"type": "boolean"}, "consulta": {"type": "string"}},
    "required": ["buscar", "consulta"],
}
PEDIDO_DECISAO = """Hoje é {hoje}. Decida se a pergunta abaixo precisa de informação da internet.

Precisa: preços, notícias, lançamentos, clima, resultados de jogos, especificações de produtos, como fazer algo técnico, fatos do mundo que mudam. Também precisa quando é um conselho de compra (o que comprar, qual modelo pegar), porque depende de preços e opções atuais.
Não precisa: a vida da pessoa, gostos, sentimentos, lembranças, amigos, conselhos pessoais que não envolvem comprar nada, conversa casual.

Se precisar, escreva uma consulta curta de busca em português, sem nomes de pessoas nem dados pessoais. Em compras, inclua na consulta o preço máximo e, se o valor for baixo para produtos novos, a palavra "usada" ou "usado". Se não precisar, deixe a consulta vazia.
{anterior}
Pergunta: {pergunta}"""


def decidir_busca(modelo, pergunta, pergunta_anterior=""):
    """Pergunta ao modelo se vale buscar na web. Devolve a consulta ou ''."""
    anterior = f"\nPergunta anterior, para contexto: {pergunta_anterior}\n" if pergunta_anterior else ""
    try:
        resposta = ollama.chat(
            model=modelo,
            messages=[{"role": "user", "content": PEDIDO_DECISAO.format(
                hoje=f"{datetime.now():%d/%m/%Y}", anterior=anterior, pergunta=pergunta)}],
            format=FORMATO_DECISAO,
            think=False,
            # Mesmo num_ctx do chat: se mudar, o Ollama recarrega o modelo a cada pergunta.
            options={"num_ctx": 4096, "num_predict": 40, "temperature": 0},
        )
        decisao = json.loads(resposta["message"]["content"])
    except Exception:
        return ""
    consulta = str(decisao.get("consulta", "")).strip()
    return consulta if decisao.get("buscar") and consulta else ""


def buscar_na_web(pergunta):
    """Busca sem conta nem chave. Devolve (texto para o modelo, links)."""
    try:
        from ddgs import DDGS
    except ImportError:
        return "(A busca na web falhou: rode pip install ddgs)", []

    erros = []
    # Tenta algumas combinações: às vezes um buscador bloqueia ou não acha nada.
    for opcoes in ({"region": "br-pt"}, {"region": "wt-wt"}, {"region": "br-pt", "backend": "bing"}):
        try:
            resultados = DDGS().text(pergunta, max_results=MAX_RESULTADOS_WEB, **opcoes)
        except Exception as erro:
            erros.append(str(erro) or type(erro).__name__)
            continue
        if resultados:
            texto = "\n\n".join(
                f"[{r.get('title', '')}]\n{(r.get('body') or '')[:MAX_CARACTERES_RESULTADO]}" for r in resultados
            )
            return texto, [r.get("href", "") for r in resultados]

    return f"(A busca na web falhou: {' / '.join(erros) or 'nada encontrado'})", []


def registrar_conversa(pergunta, resposta):
    """Salva a troca num arquivo por dia, no mesmo formato das conversas do WhatsApp."""
    PASTA_CHAT.mkdir(exist_ok=True)
    agora = datetime.now()
    arquivo = PASTA_CHAT / f"{agora:%Y-%m-%d}.jsonl"
    with open(arquivo, "a", encoding="utf-8") as f:
        for autor, texto in ((EU, pergunta), (AUTOR_MARCOS, resposta)):
            mensagem = {
                "conversa": NOME_CONVERSA_CHAT,
                "data": f"{agora:%d/%m/%Y}",
                "hora": f"{agora:%H:%M:%S}",
                "autor": autor,
                "texto": texto.strip(),
            }
            f.write(json.dumps(mensagem, ensure_ascii=False) + "\n")


def ler_conversas_chat():
    """Todas as conversas salvas com o MARCOS, em ordem de data."""
    mensagens = []
    for arquivo in sorted(PASTA_CHAT.glob("*.jsonl")):
        with open(arquivo, encoding="utf-8") as f:
            mensagens.extend(json.loads(linha) for linha in f if linha.strip())
    return mensagens


def adicionar_lembrete(texto):
    with open(ARQUIVO_LEMBRETES, "a", encoding="utf-8") as f:
        f.write(f"{datetime.now():%d/%m/%Y}: {texto.strip()}\n")


def ler_lembretes():
    """Os lembretes mais recentes, para entrar junto do perfil."""
    if not ARQUIVO_LEMBRETES.exists():
        return ""
    linhas = [l.strip() for l in ARQUIVO_LEMBRETES.read_text(encoding="utf-8-sig").splitlines() if l.strip()]
    return "\n".join(f"- {linha}" for linha in linhas[-MAX_LEMBRETES:])
