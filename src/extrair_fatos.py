"""Lê as conversas com calma e anota fatos sobre o Markin. Pode rodar por várias noites."""
import json
import time

import ollama

from buscar import PASTA_INDICE, dias_atras, tem_contexto
from indexar import chave, mostrar_progresso, minhas_falas
from marcos import MODELO, marcar_autores

ARQUIVO_FATOS = PASTA_INDICE / "fatos.jsonl"
MIN_CARACTERES_FALAS = 150   # conversa com pouca fala dele quase nunca tem fato útil
OPCOES = {"num_ctx": 2048, "num_predict": 150, "temperature": 0}
FORMATO = {
    "type": "object",
    "properties": {"fatos": {"type": "array", "items": {"type": "string"}}},
    "required": ["fatos"],
}
PEDIDO = """Na conversa abaixo, as linhas "EU:" são da pessoa analisada e as linhas "OUTRO" são de outras pessoas.

Liste até 3 fatos duradouros sobre a pessoa, usando só as linhas "EU:": gostos, coisas que ela possui, trabalho, estudos, família, objetivos e planos. Escreva frases curtas, como "Tem uma Ecosport". Ignore piadas, ironias e assuntos passageiros. Se não houver fatos, devolva a lista vazia.

Conversa:
{conversa}"""


def ja_processados():
    if not ARQUIVO_FATOS.exists():
        return set()
    with open(ARQUIVO_FATOS, encoding="utf-8") as f:
        return {json.loads(linha)["chave"] for linha in f}


def escolher_trechos():
    """Só conversas com falas suficientes, das mais novas para as mais antigas."""
    with open(PASTA_INDICE / "trechos.jsonl", encoding="utf-8") as f:
        trechos = [json.loads(linha) for linha in f]
    escolhidos = [t for t in trechos if tem_contexto(t) and len(minhas_falas(t)) >= MIN_CARACTERES_FALAS]
    return sorted(escolhidos, key=dias_atras)


def extrair(trecho):
    resposta = ollama.chat(
        model=MODELO,
        messages=[{"role": "user", "content": PEDIDO.format(conversa=marcar_autores(trecho["texto"]))}],
        format=FORMATO,
        think=False,
        options=OPCOES,
    )
    try:
        return [str(fato).strip() for fato in json.loads(resposta["message"]["content"])["fatos"]][:3]
    except (json.JSONDecodeError, KeyError, TypeError):
        return []


def main():
    feitos = ja_processados()
    pendentes = [t for t in escolher_trechos() if chave(t["texto"]) not in feitos]
    print(f"{len(feitos)} conversas já lidas, {len(pendentes)} para ler")

    inicio = time.perf_counter()
    with open(ARQUIVO_FATOS, "a", encoding="utf-8") as f:
        for numero, trecho in enumerate(pendentes, start=1):
            registro = {
                "chave": chave(trecho["texto"]),
                "conversa": trecho["conversa"],
                "data": trecho["inicio"],
                "fatos": extrair(trecho),
            }
            f.write(json.dumps(registro, ensure_ascii=False) + "\n")
            f.flush()   # garante que nada se perde se o PC desligar
            mostrar_progresso(numero, len(pendentes), numero, inicio)
    print("\nPronto: fatos extraídos")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nParado. Rode de novo para continuar de onde salvou.")
