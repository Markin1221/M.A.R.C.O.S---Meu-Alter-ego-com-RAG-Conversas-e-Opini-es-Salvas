"""Painel local: envia os zips e dispara o processamento. Abre só neste PC."""
import os
import re
import subprocess
import sys
import threading
import zipfile
from pathlib import Path

from flask import Flask, abort, jsonify, render_template, request

PASTA = Path(os.environ.get("MARCOS_DADOS", r"C:\Marcos-dados"))
PASTA_SRC = Path(__file__).parent
PORTA = 5000
ENDERECOS = {f"127.0.0.1:{PORTA}", f"localhost:{PORTA}"}

TAREFAS = {
    "indexar": [
        ("Limpando as conversas", ["limpar_whatsapp.py"]),
        ("Agrupando em blocos", ["criar_blocos.py", str(PASTA)]),
        ("Gerando os vetores", ["indexar.py"]),
    ],
    "fatos": [
        ("Extraindo fatos", ["extrair_fatos.py"]),
        ("Indexando os fatos", ["indexar_fatos.py"]),
    ],
}

TAREFAS["noite"] = TAREFAS["indexar"] + TAREFAS["fatos"]

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 4 * 1024 * 1024 * 1024

trava = threading.Lock()
processo = None
parar_pedido = False
estado = {"rodando": False, "tarefa": None, "etapa": "", "linha": "", "porcentagem": None, "erro": None, "fim": False}


@app.before_request
def so_deste_pc():
    """Impede que outro site aberto no navegador use o painel."""
    origem = request.headers.get("Origin")
    if request.host not in ENDERECOS or (origem and origem.split("//")[-1] not in ENDERECOS):
        abort(403)


def validar_zip(arquivo):
    """Devolve o nome seguro do arquivo ou um erro."""
    nome = Path((arquivo.filename or "").replace("\\", "/")).name
    if not nome.lower().endswith(".zip") or re.search(r'[<>:"/\\|?*]', nome):
        return None, f"{nome or 'arquivo'}: precisa ser um .zip"
    try:
        with zipfile.ZipFile(arquivo.stream) as z:
            if not any(n.endswith(".txt") for n in z.namelist()):
                return None, f"{nome}: não tem conversa exportada dentro"
    except zipfile.BadZipFile:
        return None, f"{nome}: zip inválido"
    arquivo.stream.seek(0)
    return nome, None


def executar(nome_tarefa):
    """Roda as etapas em sequência, lendo o progresso de cada script."""
    global processo, parar_pedido
    ambiente = {**os.environ, "PYTHONIOENCODING": "utf-8", "MARCOS_DADOS": str(PASTA)}
    for etapa, comando in TAREFAS[nome_tarefa]:
        estado.update(etapa=etapa, linha="Iniciando...", porcentagem=None)
        processo = subprocess.Popen(
            [sys.executable, "-u", *comando],
            cwd=PASTA_SRC, env=ambiente, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        )
        ultimas = ler_saida(processo)
        if parar_pedido:
            estado.update(rodando=False, linha="Parado. Dá para continuar de onde estava.", porcentagem=None)
            return
        if processo.wait() != 0:
            estado.update(rodando=False, erro="\n".join(ultimas[-8:]))
            return
    estado.update(rodando=False, fim=True, etapa="Concluído", linha="Tudo pronto.", porcentagem=100)


def ler_saida(proc):
    """Lê a saída aos poucos: o progresso usa \\r para reescrever a mesma linha."""
    guardadas = []
    buffer = b""
    while True:
        byte = proc.stdout.read(1)
        if not byte:
            break
        if byte not in (b"\r", b"\n"):
            buffer += byte
            continue
        if buffer:
            texto = buffer.decode("utf-8", "replace").strip()
            buffer = b""
            guardadas = (guardadas + [texto])[-20:]
            achou = re.search(r"\((\d+\.\d)%\)", texto)
            estado.update(linha=texto, porcentagem=float(achou[1]) if achou else None)
    return guardadas


@app.get("/")
def inicio():
    return render_template("painel.html")


@app.get("/api/estado")
def ler_estado():
    PASTA.mkdir(parents=True, exist_ok=True)
    zips = [{"nome": z.name, "kb": round(z.stat().st_size / 1024)} for z in sorted(PASTA.glob("*.zip"))]
    return jsonify(zips=zips, tarefa=estado)


@app.post("/api/enviar")
def enviar():
    if estado["rodando"]:
        return jsonify(erro="Espere o processamento terminar."), 409
    PASTA.mkdir(parents=True, exist_ok=True)
    salvos, erros = [], []
    for arquivo in request.files.getlist("arquivos"):
        nome, erro = validar_zip(arquivo)
        if erro:
            erros.append(erro)
            continue
        temporario = PASTA / f"{nome}.parcial"
        arquivo.save(temporario)
        os.replace(temporario, PASTA / nome)
        salvos.append(nome)
    return jsonify(salvos=salvos, erros=erros)


@app.post("/api/iniciar/<nome>")
def iniciar(nome):
    global parar_pedido
    if nome not in TAREFAS:
        abort(404)
    with trava:
        if estado["rodando"]:
            return jsonify(erro="Já tem um processamento rodando."), 409
        parar_pedido = False
        estado.update(rodando=True, tarefa=nome, etapa="Iniciando", linha="", porcentagem=None, erro=None, fim=False)
        threading.Thread(target=executar, args=(nome,), daemon=True).start()
    return jsonify(ok=True)


@app.post("/api/parar")
def parar():
    global parar_pedido
    if estado["rodando"] and processo:
        parar_pedido = True
        processo.terminate()
    return jsonify(ok=True)


if __name__ == "__main__":
    print(f"Abra http://127.0.0.1:{PORTA} no navegador")
    app.run(host="127.0.0.1", port=PORTA, threaded=True)
