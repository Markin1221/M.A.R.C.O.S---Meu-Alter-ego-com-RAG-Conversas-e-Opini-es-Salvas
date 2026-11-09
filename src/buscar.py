"""Busca nas conversas os trechos com significado parecido com a pergunta."""
import json
import os
from pathlib import Path

import numpy as np
import ollama

PASTA = Path(os.environ.get("MARCOS_DADOS", r"C:\Marcos-dados"))
PASTA_INDICE = PASTA / "indice"
MODELO = "bge-m3"
QUANTIDADE = 5
EU = "Markin🔱"
MIN_MENSAGENS = 3     # trechos curtos demais batem por uma palavra solta
MIN_CARACTERES = 100


def tem_contexto(trecho):
    """Fica só com trechos que têm conversa de verdade e a sua participação."""
    return (
        trecho["mensagens"] >= MIN_MENSAGENS
        and len(trecho["texto"]) >= MIN_CARACTERES
        and f"{EU}:" in trecho["texto"]
    )


def carregar_indice():
    with open(PASTA_INDICE / "trechos.jsonl", encoding="utf-8") as f:
        trechos = [json.loads(linha) for linha in f]
    vetores = np.load(PASTA_INDICE / "vetores.npy")

    # Filtra aqui para não precisar gerar os vetores de novo.
    manter = [i for i, trecho in enumerate(trechos) if tem_contexto(trecho)]
    trechos = [trechos[i] for i in manter]
    vetores = vetores[manter]

    # Com todos os vetores de tamanho 1, a similaridade vira uma multiplicação.
    vetores = vetores / np.linalg.norm(vetores, axis=1, keepdims=True)
    return trechos, vetores


def buscar(pergunta, trechos, vetores, quantidade=QUANTIDADE):
    """Devolve os trechos mais parecidos, do melhor para o pior."""
    consulta = np.array(ollama.embed(
        model=MODELO,
        input=pergunta,
        options={"num_gpu": 0},  # na CPU, para deixar os 4 GB da GPU para o modelo de conversa
    )["embeddings"][0])
    consulta = consulta / np.linalg.norm(consulta)
    notas = vetores @ consulta
    melhores = np.argsort(notas)[::-1][:quantidade]
    return [(float(notas[i]), trechos[i]) for i in melhores]


def main():
    trechos, vetores = carregar_indice()
    print(f"{len(trechos)} trechos carregados. Enter vazio para sair.\n")

    while True:
        pergunta = input("Pergunta: ").strip()
        if not pergunta:
            break
        for nota, trecho in buscar(pergunta, trechos, vetores):
            print(f"\n[{nota:.3f}] {trecho['conversa']} | {trecho['inicio']}")
            print(trecho["texto"][:400])
        print()


if __name__ == "__main__":
    main()
