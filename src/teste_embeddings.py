"""Mostra como o modelo de embeddings mede a semelhança de significado."""
import math

import ollama

MODELO = "bge-m3"

pergunta = "estou cansado do meu trabalho"
frases = [
    "meu emprego está me esgotando",
    "o gato subiu no telhado",
    "comprei pão na padaria hoje",
    "não aguento mais meu chefe",
]


def vetor(texto):
    """Transforma um texto numa lista de números."""
    return ollama.embed(model=MODELO, input=texto)["embeddings"][0]


def similaridade(a, b):
    """Quanto mais perto de 1, mais parecido é o significado."""
    produto = sum(x * y for x, y in zip(a, b))
    return produto / (math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b)))


v_pergunta = vetor(pergunta)
print(f"Cada texto virou {len(v_pergunta)} números\n")

notas = [(similaridade(v_pergunta, vetor(frase)), frase) for frase in frases]
for nota, frase in sorted(notas, reverse=True):
    print(f"{nota:.3f}  {frase}")
