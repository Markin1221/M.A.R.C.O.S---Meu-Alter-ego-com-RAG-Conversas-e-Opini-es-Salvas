import re
import zipfile
from collections import Counter
from pathlib import Path
import json






def ler_conversas(zip_path):
    PADRAO = re.compile(r"^\[(\d{2}/\d{2}/\d{4}), (\d{2}:\d{2}:\d{2})\] (.+?): (.*)$")
    proibidos = {'figurinha omitida', 'imagem ocultada', 'áudio ocultado','vídeo omitido'}

    with zipfile.ZipFile(zip_path) as z:
        nome = next(n for n in z.namelist() if n.endswith(".txt"))
        texto = z.read(nome).decode("utf-8-sig")

    mensagens = []
    for linha in texto.splitlines():
        linha = linha.replace("\u200e", "")  # remove um caractere invisível do WhatsApp
        m = PADRAO.match(linha)
        if m:
            mensagens.append({"conversa": zip_path.stem,"data": m[1], "hora": m[2], "autor": m[3], "texto": m[4]})
        elif mensagens:
            mensagens[-1]["texto"] += "\n" + linha

    resultado = []
    for msg in mensagens:
        if msg['texto'].strip() not in proibidos:
            resultado.append(msg)
    return resultado

def processar_conversas(path_pasta):
    todas = []
    with open(r"C:\Marcos-dados\mensagens_limpas.jsonl", "w", encoding="utf-8") as f:
        for zip_path in Path(path_pasta).glob("*.zip"):
            todas.extend(ler_conversas(zip_path))
        for msg in todas:
            f.write(json.dumps(msg,ensure_ascii=False) + "\n")


processar_conversas(r"C:\Marcos-dados")
