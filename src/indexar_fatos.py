"""Junta os fatos extraídos, remove repetidos e gera os vetores para a busca."""
import json
from datetime import datetime

from buscar import PASTA, PASTA_INDICE
from indexar import gerar_vetores

ARQUIVO_FATOS = PASTA_INDICE / "fatos.jsonl"
ARQUIVO_LISTA = PASTA_INDICE / "fatos_lista.jsonl"
ARQUIVO_VETORES = PASTA_INDICE / "vetores_fatos.npy"
ARQUIVO_LEITURA = PASTA / "fatos.txt"


def data_de(registro):
    return datetime.strptime(registro["data"], "%d/%m/%Y %H:%M:%S")


def juntar_fatos():
    """Um item por fato. Se repetir, fica a vez mais recente em que foi dito."""
    with open(ARQUIVO_FATOS, encoding="utf-8") as f:
        registros = sorted((json.loads(linha) for linha in f), key=data_de)
    fatos = {}
    for registro in registros:
        for fato in registro["fatos"]:
            texto = fato.strip().rstrip(".")
            if len(texto) >= 8:
                fatos[texto.lower()] = {"fato": texto, "data": registro["data"], "conversa": registro["conversa"]}
    return list(fatos.values())


def main():
    fatos = juntar_fatos()
    with open(ARQUIVO_LISTA, "w", encoding="utf-8") as f:
        for fato in fatos:
            f.write(json.dumps(fato, ensure_ascii=False) + "\n")

    # Versão para você ler e conferir, da mais nova para a mais antiga.
    with open(ARQUIVO_LEITURA, "w", encoding="utf-8") as f:
        for fato in sorted(fatos, key=data_de, reverse=True):
            f.write(f"{fato['data'][:10]}  {fato['fato']}\n")

    gerar_vetores("fatos", ARQUIVO_VETORES, lambda fato: fato["fato"], fatos)
    print(f"Pronto: {len(fatos)} fatos indexados")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nParado. Rode de novo para continuar de onde salvou.")
