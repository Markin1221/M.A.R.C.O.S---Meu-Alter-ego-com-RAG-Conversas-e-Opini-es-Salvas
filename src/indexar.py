"""Gera os vetores do RAG. Só calcula o que ainda não está no cache."""
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import ollama

from buscar import EU, PASTA, separar_linhas

ENTRADA = PASTA / "blocos.jsonl"
PASTA_INDICE = PASTA / "indice"
ARQUIVO_TRECHOS = PASTA_INDICE / "trechos.jsonl"
ARQUIVO_VETORES = PASTA_INDICE / "vetores.npy"
ARQUIVO_VETORES_EU = PASTA_INDICE / "vetores_eu.npy"

MODELO = "bge-m3"
MAX_CARACTERES = 1500   # divide as poucas mensagens gigantes
TAMANHO_LOTE = 16       # quantos textos vão de uma vez para o modelo
SALVAR_A_CADA = 20      # lotes entre um salvamento e outro


def criar_trechos():
    """Lê os blocos e divide os que passam do limite."""
    trechos = []
    with open(ENTRADA, encoding="utf-8") as f:
        for linha in f:
            bloco = json.loads(linha)
            texto = bloco["texto"]
            for inicio in range(0, len(texto), MAX_CARACTERES):
                trechos.append({**bloco, "texto": texto[inicio:inicio + MAX_CARACTERES]})
    return trechos


def texto_bloco(trecho):
    return trecho["texto"]


def minhas_falas(trecho):
    falas = [conteudo for autor, conteudo in separar_linhas(trecho["texto"]) if autor == EU]
    # Sem falas dele, usa o texto inteiro para manter um vetor por trecho.
    return "\n".join(falas).strip() or trecho["texto"]


def chave(texto):
    """Identifica um texto: mesmo texto, mesma chave, mesmo vetor."""
    return hashlib.sha1(texto.encode("utf-8")).hexdigest()


def arquivo_cache(nome):
    return PASTA_INDICE / f"cache_{nome}.npz"


def carregar_cache(nome):
    if not arquivo_cache(nome).exists():
        return {}
    dados = np.load(arquivo_cache(nome))
    return dict(zip(dados["chaves"].tolist(), dados["vetores"]))


def salvar_cache(nome, cache):
    """Grava num arquivo temporário antes, para não corromper o cache."""
    temporario = PASTA_INDICE / f"cache_{nome}_tmp.npz"
    np.savez(temporario, chaves=np.array(list(cache)), vetores=np.array(list(cache.values()), dtype=np.float32))
    os.replace(temporario, arquivo_cache(nome))


def migrar(nome, arquivo_vetores, texto_de):
    """Aproveita os vetores já calculados antes de o cache existir."""
    if arquivo_cache(nome).exists() or not arquivo_vetores.exists() or not ARQUIVO_TRECHOS.exists():
        return
    with open(ARQUIVO_TRECHOS, encoding="utf-8") as f:
        trechos = [json.loads(linha) for linha in f]
    salvar_cache(nome, {chave(texto_de(t)): v for t, v in zip(trechos, np.load(arquivo_vetores))})


def formatar_tempo(segundos):
    horas, resto = divmod(int(segundos), 3600)
    minutos, segundos = divmod(resto, 60)
    return f"{horas}h{minutos:02d}m{segundos:02d}s"


def mostrar_progresso(feitos, total, feitos_agora, inicio):
    """Estima o tempo restante pela velocidade desta execução."""
    velocidade = feitos_agora / (time.perf_counter() - inicio)
    falta = formatar_tempo((total - feitos) / velocidade)
    porcentagem = 100 * feitos / total
    print(
        f"\r{feitos}/{total} ({porcentagem:.1f}%) | "
        f"{velocidade:.1f} trechos/s | faltam ~{falta}   ",
        end="",
        flush=True,
    )


def gerar_vetores(nome, arquivo_vetores, texto_de, trechos):
    """Calcula o que falta e grava um vetor por trecho, na ordem dos trechos."""
    cache = carregar_cache(nome)
    novos = {}
    for trecho in trechos:
        texto = texto_de(trecho)
        if chave(texto) not in cache:
            novos[chave(texto)] = texto
    print(f"{nome}: {len(trechos) - len(novos)} prontos, {len(novos)} para calcular")

    chaves = list(novos)
    inicio = time.perf_counter()
    for lote, i in enumerate(range(0, len(chaves), TAMANHO_LOTE), start=1):
        grupo = chaves[i:i + TAMANHO_LOTE]
        resposta = ollama.embed(model=MODELO, input=[novos[c] for c in grupo])
        cache.update(zip(grupo, resposta["embeddings"]))
        mostrar_progresso(min(i + TAMANHO_LOTE, len(chaves)), len(chaves), min(i + TAMANHO_LOTE, len(chaves)), inicio)
        if lote % SALVAR_A_CADA == 0:
            salvar_cache(nome, cache)
    if chaves:
        print()
        salvar_cache(nome, cache)

    temporario = PASTA_INDICE / f"{arquivo_vetores.stem}_tmp.npy"
    np.save(temporario, np.array([cache[chave(texto_de(t))] for t in trechos], dtype=np.float32))
    os.replace(temporario, arquivo_vetores)


def indexar_falas():
    PASTA_INDICE.mkdir(exist_ok=True)
    migrar("falas", ARQUIVO_VETORES_EU, minhas_falas)
    gerar_vetores("falas", ARQUIVO_VETORES_EU, minhas_falas, criar_trechos())
    print("Pronto: falas indexadas")


def main():
    PASTA_INDICE.mkdir(exist_ok=True)
    # Migra as duas antes de regravar trechos.jsonl, que pode mudar de ordem.
    migrar("blocos", ARQUIVO_VETORES, texto_bloco)
    migrar("falas", ARQUIVO_VETORES_EU, minhas_falas)

    trechos = criar_trechos()
    gerar_vetores("blocos", ARQUIVO_VETORES, texto_bloco, trechos)

    temporario = PASTA_INDICE / "trechos_tmp.jsonl"
    with open(temporario, "w", encoding="utf-8") as f:
        for trecho in trechos:
            f.write(json.dumps(trecho, ensure_ascii=False) + "\n")
    os.replace(temporario, ARQUIVO_TRECHOS)

    gerar_vetores("falas", ARQUIVO_VETORES_EU, minhas_falas, trechos)
    print(f"Pronto: {len(trechos)} trechos indexados")


if __name__ == "__main__":
    main()
