"""Gera só os vetores das falas do Markin."""
from indexar import indexar_falas

if __name__ == "__main__":
    try:
        indexar_falas()
    except KeyboardInterrupt:
        print("\nParado. Rode de novo para continuar de onde salvou.")
