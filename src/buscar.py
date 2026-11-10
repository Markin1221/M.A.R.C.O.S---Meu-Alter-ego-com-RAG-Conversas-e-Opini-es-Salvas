"""Busca nas conversas os trechos com significado parecido com a pergunta."""
import json
import os
import re
from datetime import datetime
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
PESO_EU = 0.7         # quanto as suas falas pesam na busca (o resto é o bloco inteiro)
PESO_RECENTE = 0.03   # pequeno bônus para mensagens novas; cai pela metade a cada ano

AUTOR = re.compile(r"^([^:\n]{1,40}): ")


def separar_linhas(texto):
    """Devolve (autor, conteúdo) de cada linha; continuações herdam o autor anterior."""
    autor = None
    linhas = []
    for linha in texto.split("\n"):
        achou = AUTOR.match(linha)
        if achou:
            autor, linha = achou[1], linha[achou.end():]
        linhas.append((autor, linha))
    return linhas


def dias_atras(trecho):
    data = datetime.strptime(trecho["inicio"], "%d/%m/%Y %H:%M:%S")
    return (datetime.now() - data).days


def normalizar(vetores):
    return vetores / np.linalg.norm(vetores, axis=1, keepdims=True)


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
    vetores = normalizar(vetores[manter])

    # Mistura o bloco inteiro com só as suas falas, se esse índice já existir.
    arquivo_eu = PASTA_INDICE / "vetores_eu.npy"
    if arquivo_eu.exists():
        vetores_eu = np.load(arquivo_eu)
        if len(vetores_eu) > max(manter):
            vetores = PESO_EU * normalizar(vetores_eu[manter]) + (1 - PESO_EU) * vetores
        else:
            print("Aviso: vetores_eu.npy está incompleto. Rode src\\indexar_eu.py até o fim.\n")
    return trechos, vetores


def vetor_pergunta(pergunta):
    consulta = np.array(ollama.embed(
        model=MODELO,
        input=pergunta,
        options={"num_gpu": 0},  # na CPU, para deixar os 4 GB da GPU para o modelo de conversa
    )["embeddings"][0])
    return consulta / np.linalg.norm(consulta)


def carregar_fatos():
    """Devolve (fatos, vetores), ou listas vazias se os fatos ainda não foram indexados."""
    arquivo_lista = PASTA_INDICE / "fatos_lista.jsonl"
    arquivo_vetores = PASTA_INDICE / "vetores_fatos.npy"
    if not (arquivo_lista.exists() and arquivo_vetores.exists()):
        return [], None
    with open(arquivo_lista, encoding="utf-8") as f:
        fatos = [json.loads(linha) for linha in f]
    return fatos, normalizar(np.load(arquivo_vetores))


def buscar_fatos(consulta, fatos, vetores, quantidade=8, nota_minima=0.45):
    if not fatos:
        return []
    notas = vetores @ consulta
    melhores = np.argsort(notas)[::-1][:quantidade]
    return [fatos[i] for i in melhores if notas[i] >= nota_minima]


def buscar(pergunta, trechos, vetores, quantidade=QUANTIDADE, consulta=None):
    """Devolve os trechos mais parecidos, do melhor para o pior."""
    if consulta is None:
        consulta = vetor_pergunta(pergunta)
    idades = np.array([dias_atras(t) for t in trechos])
    notas = vetores @ consulta + PESO_RECENTE * 0.5 ** (idades / 365)
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
    try:
        main()
    except KeyboardInterrupt:
        print("\nParado. Rode de novo para continuar de onde salvou.")
