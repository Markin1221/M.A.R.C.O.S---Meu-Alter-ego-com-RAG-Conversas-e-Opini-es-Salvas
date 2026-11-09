"""Agrupa as mensagens limpas em blocos de conversa para o RAG."""
import json
import sys
from datetime import datetime
from pathlib import Path

# Por padrão usa C:\Marcos-dados; outra pasta pode ser passada ao rodar o script.
PASTA = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(r"C:\Marcos-dados")
ENTRADA = PASTA / "mensagens_limpas.jsonl"
SAIDA = PASTA / "blocos.jsonl"

MAX_MINUTOS = 30        # pausa maior que isso começa um bloco novo
MAX_CARACTERES = 1500   # blocos curtos cabem no contexto pequeno do modelo


def ler_mensagens():
    with open(ENTRADA, encoding="utf-8") as f:
        return [json.loads(linha) for linha in f]


def data_hora(msg):
    return datetime.strptime(f"{msg['data']} {msg['hora']}", "%d/%m/%Y %H:%M:%S")


def formatar(msg):
    return f"{msg['autor']}: {msg['texto'].strip()}"


def agrupar(mensagens):
    grupos = []
    atual = []
    tamanho = 0
    for msg in mensagens:
        linha = len(formatar(msg)) + 1
        if atual:
            anterior = atual[-1]
            mudou = msg["conversa"] != anterior["conversa"]
            pausa = (data_hora(msg) - data_hora(anterior)).total_seconds() / 60
            if mudou or pausa > MAX_MINUTOS or tamanho + linha > MAX_CARACTERES:
                grupos.append(atual)
                atual = []
                tamanho = 0
        atual.append(msg)
        tamanho += linha
    if atual:
        grupos.append(atual)
    return grupos


def montar_bloco(grupo):
    return {
        "conversa": grupo[0]["conversa"],
        "inicio": f"{grupo[0]['data']} {grupo[0]['hora']}",
        "fim": f"{grupo[-1]['data']} {grupo[-1]['hora']}",
        "mensagens": len(grupo),
        "texto": "\n".join(formatar(m) for m in grupo),
    }


def main():
    mensagens = ler_mensagens()
    blocos = [montar_bloco(g) for g in agrupar(mensagens)]

    with open(SAIDA, "w", encoding="utf-8") as f:
        for bloco in blocos:
            f.write(json.dumps(bloco, ensure_ascii=False) + "\n")

    print(f"{len(mensagens)} mensagens viraram {len(blocos)} blocos")


if __name__ == "__main__":
    main()
