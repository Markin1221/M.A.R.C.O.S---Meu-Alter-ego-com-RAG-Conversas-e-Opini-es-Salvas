"""Transforma os blocos em vetores e salva o índice do RAG."""
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import ollama

# Por padrão usa C:\Marcos-dados; outra pasta pode ser passada ao rodar o script.
PASTA = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(r"C:\Marcos-dados")
ENTRADA = PASTA / "blocos.jsonl"
PASTA_INDICE = PASTA / "indice"
ARQUIVO_TRECHOS = PASTA_INDICE / "trechos.jsonl"
ARQUIVO_VETORES = PASTA_INDICE / "vetores.npy"

MODELO = "bge-m3"
MAX_CARACTERES = 1500   # divide as poucas mensagens gigantes
TAMANHO_LOTE = 16       # quantos trechos vão de uma vez para o modelo
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


def carregar_progresso(total):
    """Continua de onde parou se o script tiver sido interrompido."""
    if not ARQUIVO_VETORES.exists():
        return []
    vetores = np.load(ARQUIVO_VETORES)
    return list(vetores) if len(vetores) <= total else []


def salvar(vetores):
    """Grava num arquivo temporário antes, para não corromper o índice."""
    temporario = PASTA_INDICE / "vetores_tmp.npy"
    np.save(temporario, np.array(vetores, dtype=np.float32))
    os.replace(temporario, ARQUIVO_VETORES)


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


def main():
    PASTA_INDICE.mkdir(exist_ok=True)
    trechos = criar_trechos()

    with open(ARQUIVO_TRECHOS, "w", encoding="utf-8") as f:
        for trecho in trechos:
            f.write(json.dumps(trecho, ensure_ascii=False) + "\n")

    vetores = carregar_progresso(len(trechos))
    if vetores:
        print(f"Continuando do trecho {len(vetores)}")

    lotes_feitos = 0
    ja_tinha = len(vetores)
    inicio = time.perf_counter()
    for i in range(len(vetores), len(trechos), TAMANHO_LOTE):
        textos = [t["texto"] for t in trechos[i:i + TAMANHO_LOTE]]
        vetores.extend(ollama.embed(model=MODELO, input=textos)["embeddings"])
        lotes_feitos += 1
        mostrar_progresso(len(vetores), len(trechos), len(vetores) - ja_tinha, inicio)
        if lotes_feitos % SALVAR_A_CADA == 0:
            salvar(vetores)

    salvar(vetores)
    print(f"\nPronto: {len(vetores)} trechos indexados")


if __name__ == "__main__":
    main()
