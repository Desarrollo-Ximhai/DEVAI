# -*- coding: utf-8 -*-
import time
import os
from funciones import debug

from qdrant_client import QdrantClient
from qdrant_client.http.models import Filter, FieldCondition, MatchValue, PointStruct
import uuid
from datetime import datetime
from fastapi import FastAPI, Request, UploadFile
from pydantic import BaseModel
import uvicorn
from typing import Optional
import json

from accionesQdrant import Qdrant, conectarQdrant
#from accionesQdrant import conectarQdrant, borrar_por_chat_id, borrar_por_point_id, search_in_qdrant, save_to_qdrant, getProjectMemory, embebirBaseDatos
from accionesGemini import conectarGemini, generate_response, embed_with_gemini

#para la autenticacion de la API
from fastapi import Depends
from fastapi import HTTPException, Header

#Para el tokenizador
from typing import Any, Optional
import tiktoken

ADMIN_KEY = os.environ.get("ADMIN_API_KEY")
def verificar_clave(api_key: str = Header(...)):
    if api_key != ADMIN_KEY:
        raise HTTPException(status_code=403, detail="No autorizado: Clave inválida")
QDRANT_URL = os.environ["QDRANT_URL"]
QDRANT_API_KEY = os.environ.get("QDRANT_API_KEY") 
KEY_FREE2 = os.environ.get("GOOGLE_API_KEY2") 
GOOGLE_API_KEY= os.environ.get('KEY-FREE') 
tokens_entrada_acumulados =0
tokens_salida_acumulados =0
conectarGemini(GOOGLE_API_KEY)

client = conectarQdrant(QDRANT_URL, QDRANT_API_KEY)


def optimizar_y_aplanar_historial(historial: Any, max_tokens: int):
    """
    Parsea, limpia y aplana el historial sin importar si viene como 
    String JSON o como lista de diccionarios desde PHP.
    """
    if isinstance(historial, str):
        try:
            historial = json.loads(historial)
        except Exception:
            return []
            
    if not isinstance(historial, list):
        return []

    encoding = tiktoken.get_encoding("cl100k_base")
    historial_plano_final = []
    tokens_acumulados = 0
    
    for turno in reversed(historial):
        
        if isinstance(turno, dict):
            u_text = turno.get("user", "")
            a_text = turno.get("assistant", "")
        else:
            continue
            
        tokens_user = len(encoding.encode(str(u_text))) + 4
        tokens_assistant = len(encoding.encode(str(a_text))) + 4
        tokens_turno = tokens_user + tokens_assistant
        
        if tokens_acumulados + tokens_turno > max_tokens:
            break
            
        componentes_turno = [
            {"role": "user", "content": u_text},
            {"role": "assistant", "content": a_text}
        ]
        
        historial_plano_final = componentes_turno + historial_plano_final
        tokens_acumulados += tokens_turno
    print(tokens_acumulados)
    return historial_plano_final




def build_prompt_from_chunks(chunksCodigo, chunksBD, chunksArchivo, user_query, memory=None, historial = ''):

    def extraer_texto(chunk):
        if isinstance(chunk, dict):
            return chunk.get("text", "")  
        elif hasattr(chunk, "payload") and chunk.payload:
            return chunk.payload.get("text", "")  
        return ""

    contextCodigo = "\n\n---\n\n".join([
        extraer_texto(chunk) for chunk in chunksCodigo if extraer_texto(chunk)
    ])

    contextBD = "\n\n---\n\n".join([
        extraer_texto(chunk) for chunk in chunksBD if extraer_texto(chunk)
    ])

    contextoArchivo = "\n\n---\n\n".join([
        extraer_texto(chunk) for chunk in chunksArchivo if extraer_texto(chunk)
    ])

    memoria = ""
    if memory and len(memory) > 0:
        memoria = "\n\n".join([
            f"[{i+1}] {extraer_texto(chunk)}"
            for i, chunk in enumerate(memory) if extraer_texto(chunk)
        ])

    memoria_block = ""
    if memoria:
        memoria_block = (
            "[MEMORIA] MEMORIA DE CHATS RELACIONADOS A LA PREGUNTA:\n"
            + memoria +
            "\n\n---\n"
        )

    codigo_block = ""
    if contextCodigo:
        codigo_block = (
            "[CODIGO] CONTEXTO DE CÓDIGO :\n"
            + contextCodigo +
            "\n\n---\n"
        )
    bd_block = ""
    if contextBD:
        bd_block = (
            "[BD] CONTEXTO DE BASE DE DATOS :\n"
            + contextBD +
            "\n\n---\n"
        )

    archivo_block = ""
    if contextoArchivo:
        archivo_block = (
            "[ANALISIS] CONTEXTO DE ANÁLISIS:\n"
            + contextoArchivo +
            "\n\n---\n"
        )

    prompt = f"""
Eres un asistente de desarrollo extremadamente preciso y especializado en interpretar código PHP, HTML y SQL dentro de un framework personalizado.

A continuación tienes fragmentos REALES de código fuente del framework ([CODIGO]). No inventes ni completes nada que no esté explícitamente en el texto. No menciones de dónde salió el fragmento. No hagas suposiciones. Si no hay suficiente información para responder con certeza, responde claramente que no es posible responder y da la razón.

INSTRUCCIONES:
- Usa solo lo que se encuentra en el contexto ([CODIGO], [BD], [ANALISIS]) y en la memoria([MEMORIA], [HISTORIAL]).
- Si el usuario adjuntó imágenes, diagramas o archivos directamente en la petición actual, analízalos rigurosamente junto con el contexto de código provisto.
- Responde de forma concreta y profesional.
- No repitas el prompt ni resumas el contexto.
- En caso de las vistas no inventes inputs ni etiquetas HTML, utiliza siempre la clase Ximhai o los ejemplos de código([CODIGO]) para extraer datos.
- No generes estructuras incompletas.
- El codigo debe ir encasillado dentro de (```)

{memoria_block}
---

{codigo_block}

---
{bd_block}

---
{archivo_block}

---

---
[HISTORIAL] HISTORIAL DE CONVERSACIÓN:
{historial}

---


PREGUNTA:
{user_query}

---

RESPUESTA:
"""
    return prompt.strip()




def decontextualize_query(historial_plano, nueva_pregunta, model_name="models/gemini-3.1-flash-lite"):
    global tokens_entrada_acumulados
    global tokens_salida_acumulados
    """
    Toma el historial y la pregunta actual, y devuelve una query optimizada para búsqueda vectorial.
    """
    # Convertimos el historial plano a un string legible para el modelo de reformulación
    historial_texto = ""
    for turno in historial_plano:
        historial_texto += f"{turno['role'].upper()}: {turno['content']}\n"
        
    prompt_reformador = f"""
        A continuación se muestra una conversación entre un USUARIO y un ASISTENTE, seguida de una NUEVA PREGUNTA del usuario.
        Tu única tarea es analizar la conversación y reescribir la NUEVA PREGUNTA para que sea una consulta independiente, clara y rica en contexto, ideal para buscar en una base de datos vectorial.

        REGLAS ESTRICTAS:
        1. Reemplaza pronombres o referencias ambiguas ("eso", "aquello", "la clase", "el error anterior") por los nombres de los conceptos reales mencionados en el historial.
        2. Si la NUEVA PREGUNTA ya es independiente y no depende del historial, devuélvela EXACTAMENTE igual, sin añadir nada.
        3. NO respondas la pregunta. NO agregues saludos ni explicaciones. Devuelve SOLO la pregunta reformulada.

        [HISTORIAL DE CONVERSACIÓN]
        {historial_texto}

        [NUEVA PREGUNTA]
        {nueva_pregunta}

        QUERY REFORMULADA OPTIMIZADA:
        """
    response = generate_response(prompt_reformador, model_name)
    tokens_entrada_acumulados += response["tokens_entrada"]
    tokens_salida_acumulados += response["tokens_salida"]
    texto = response["texto"].strip()
    debug(f"Query de descontextualizacion, TokIn+: {tokens_entrada_acumulados}, TokOut+: {tokens_salida_acumulados} Entro: {nueva_pregunta}, salió: {texto} ")
    return response["texto"].strip()

def query_rag(user_query: str, memoria, chat_id:int, codigo, bd, archivo, proyecto: str = "default", model_name= "models/gemini-3-flash-preview", historial = '', max_tokens = 6000, archivos = None  ):
    print(model_name)
    global tokens_entrada_acumulados
    global tokens_salida_acumulados
    global client
    try:

        if not user_query:
            return {'error': 'No se recibió un prompt válido'}, 400
        if not chat_id:
            return {'error': 'No se recibió un id de chat válido'}, 400

        historialModificado = optimizar_y_aplanar_historial(historial, max_tokens)
        query_para_busqueda = decontextualize_query(historialModificado, user_query)
        user_queryAux = user_query
        user_query = query_para_busqueda

        query_embedding = embed_with_gemini(user_query, tipo= "retrieval_query" )
        if query_embedding is None:
            return {'error': 'Failed to generate embedding for query'}, 500

        query_embedding768 = embed_with_gemini(user_query,768, tipo= "retrieval_query")
        if query_embedding768 is None:
            return {'error': 'Failed to generate embedding 768 for query'}, 500

        collection_memory = memoria

        objBD = Qdrant(
            client=client,
            collection=bd,
            proyecto=proyecto
        )
        objCodigo = Qdrant(
            client=client,
            collection=codigo,
            proyecto=proyecto
        )
        objArchivo = Qdrant(
            client=client,
            collection=archivo,
            proyecto=proyecto
        )

        objMemoria = Qdrant(
            client=client,
            collection=collection_memory,
            proyecto=proyecto
        )

        chunksCodigo = objCodigo.search_in_qdrant( user_query, query_embedding, None, k=25 )
        chunksBD = objBD.search_in_qdrant(user_query, query_embedding768, proyecto , k=40)
        chunksArchivo = objArchivo.search_in_qdrant(user_query, query_embedding768, None, k=10)

        memory = objMemoria.getProjectMemory(
            embed_fn=embed_with_gemini,
            user_query=user_query,
            collection_memory=collection_memory,
            chat_id=chat_id,
            proyecto=proyecto,
            limit=4
        )
        
        user_query = user_queryAux
        prompt = build_prompt_from_chunks(chunksCodigo, chunksBD, chunksArchivo, user_query, memory, historialModificado)
        debug('_____________________________________________________')
        debug(prompt)
        debug('_____________________________________________________')
        response = generate_response(prompt, model_name, archivos)        

        tokens_entrada_acumulados += response["tokens_entrada"]
        tokens_salida_acumulados += response["tokens_salida"]
        response_text = response["texto"].strip()
        debug(f"Query de rag, TokIn+: {tokens_entrada_acumulados}, TokOut+: {tokens_salida_acumulados}")

        uuids = objMemoria.save_to_qdrant(
            embed_fn=embed_with_gemini,
            user_query=user_query,
            collection_memory=collection_memory,
            respuesta=response_text,
            chat_id=chat_id,
            proyecto=proyecto
        )
        return {'response': response_text, 'uuids' : uuids, 'tokens_entrada' : tokens_entrada_acumulados, 'tokens_salida': tokens_salida_acumulados}, 200

    except Exception as e:
        return {'error': str(e)}, 500

def enrutar_consulta(user_query: str, historial: str = "", modelo = 'models/gemini-3.1-flash-lite') -> str:
    global tokens_entrada_acumulados
    global tokens_salida_acumulados
    """
    Analiza la consulta del usuario y decide si requiere el contexto del framework (RAG)
    o si puede ser respondida directamente por el LLM (FREE).
    """
    prompt_router = f"""
    Actúas como un clasificador de consultas de alta precisión para un sistema de desarrollo de software.
    Tu única tarea es analizar la NUEVA PREGUNTA del usuario (y el historial si es necesario) y determinar si para responderla se requiere buscar información específica dentro del código fuente, la estructura de la base de datos o los análisis del framework personalizado del usuario.

    RESPONDE ÚNICAMENTE CON UNA DE ESTAS DOS PALABRAS:
    - 'RAG': Si la pregunta menciona componentes, vistas, clases, tablas específicas, lógica del framework personalizado, o frases como "cómo se arma la consulta en X tabla".
    - 'FREE': Si es una pregunta de conocimiento general de programación, dudas sobre APIs externas (ej. precios o modelos de Gemini), saludos, o charlas generales que el modelo puede responder con su propio conocimiento sin ver el framework.

    [HISTORIAL RECIENTE]
    {historial}

    [NUEVA PREGUNTA]
    {user_query}

    DECISIÓN (Escribe solo RAG o FREE):"""

    try:
        # Usamos el modelo más rápido disponible para no penalizar la latencia
        response = generate_response(prompt_router, modelo)

        tokens_entrada_acumulados += response["tokens_entrada"]
        tokens_salida_acumulados += response["tokens_salida"]
        response = response["texto"].strip()

        decision = response.upper()
        debug(f"Query de enrutamiento, TokIn+: {tokens_entrada_acumulados}, TokOut+: {tokens_salida_acumulados}. Ruta: {decision}")
        # Sanitizamos la respuesta por si el LLM añade puntos o espacios
        if "FREE" in decision:
            return "FREE"
        return "RAG"
    except Exception as e:
        return "RAG" # Por seguridad, si falla el router, usamos el RAG


app = FastAPI()

@app.get("/health")
def health():
    return {
        "status": "ok"
    }

@app.post("/devai", dependencies=[Depends(verificar_clave)])
async def devai_endpoint(request: Request):
    form_data = await request.form()
    
    query = form_data.get("query", "")
    memoria = form_data.get("memoria", "DevAI-Memory")
    chat_id = int(form_data.get("chat_id", 0))
    codigo = form_data.get("codigo", "DEVAI-embeddings")
    bd = form_data.get("basedatos", form_data.get("bd", "DevAI-DB"))
    archivo = form_data.get("analisis", form_data.get("archivo", "DevAI-Analisis"))
    proyecto = form_data.get("proyecto", "default")
    model_name = form_data.get("model_name", "models/gemini-3.1-flash-lite")
    historial = form_data.get("historial", "")
    max_tokens = int(form_data.get("max_tokens", 6000))

    archivos_procesados = []
    for key, value in form_data.items():
        if key.startswith("files[") and hasattr(value, "filename"):
            contenido_bytes = await value.read()
            archivos_procesados.append({
                "mime_type": value.content_type,   
                "data": contenido_bytes          
            })

    

    respuesta = query_rag(
		user_query=query,
		memoria=memoria,
		chat_id=chat_id,
		codigo=codigo,
		bd=bd,
		archivo=archivo,
		proyecto=proyecto,
        model_name=model_name,
        historial=historial,
        max_tokens=max_tokens,
        archivos=archivos_procesados
        )
	#print('respuesta')
	#print(respuesta)
    return {"response": respuesta}


@app.post("/devaiAgent", dependencies=[Depends(verificar_clave)])
async def devai_endpoint(request: Request):
    global tokens_entrada_acumulados
    global tokens_salida_acumulados
    # 1. Extraemos todo el contenido del formulario multipart
    form_data = await request.form()
    
    # 2. Extraemos los campos de texto con los mismos valores por defecto que tenías
    query = form_data.get("query", "")
    memoria = form_data.get("memoria", "DevAI-Memory")
    chat_id = int(form_data.get("chat_id", 0))
    codigo = form_data.get("codigo", "DEVAI-embeddings")
    bd = form_data.get("basedatos", form_data.get("bd", "DevAI-DB"))
    archivo = form_data.get("analisis", form_data.get("archivo", "DevAI-Analisis"))
    proyecto = form_data.get("proyecto", "default")
    model_name = form_data.get("model_name", "models/gemini-3.1-flash-lite")
    historial = form_data.get("historial", "")
    max_tokens = int(form_data.get("max_tokens", 6000))

    archivos_procesados = []
    for key, value in form_data.items():
        if key.startswith("files[") and hasattr(value, "filename"):
            contenido_bytes = await value.read()
            archivos_procesados.append({
                "mime_type": value.content_type,   
                "data": contenido_bytes          
            })


    ruta = enrutar_consulta(query, historial)
    print(f"🤖 [ROUTER SEMÁNTICO] Consulta: '{query}' -> Clasificada como: {ruta}")

    if ruta == "FREE":
        print('Entrando en respuesta FREE')
        response = generate_response(query, model_name)
        tokens_entrada_acumulados += response["tokens_entrada"]
        tokens_salida_acumulados += response["tokens_salida"]
        response = response["texto"].strip()
        return {'response': response}, 200

    print('Entrando en respuesta RAG')
    respuesta = query_rag(
		user_query=query,
		memoria=memoria,
		chat_id=chat_id,
		codigo=codigo,
		bd=bd,
		archivo=archivo,
		proyecto=proyecto,
        model_name=model_name,
        historial=historial,
        max_tokens=max_tokens,
        archivos=archivos_procesados
        )
	#print('respuesta')
	#print(respuesta)
    return {"response": respuesta}

#
# =================================================================
# NUEVO APARTADO: ENDPOINT PARA PROMPTS LIBRES (SIN RAG / QDRANT)
# =================================================================

class FreePromptRequest(BaseModel):
    prompt: str
    model_name: str 

@app.post("/prompt", dependencies=[Depends(verificar_clave)])
def free_prompt_endpoint(request: FreePromptRequest):
    global tokens_entrada_acumulados
    global tokens_salida_acumulados
    conectarGemini(KEY_FREE2)
    try:
        if not request.prompt:
            return {"error": "No se recibió un prompt válido"}, 400
        if not request.model_name:
            return {"error": "No se recibió un modelo válido"}, 400        

        response = generate_response(request.prompt, request.model_name)
        
        tokens_entrada_acumulados += response["tokens_entrada"]
        tokens_salida_acumulados += response["tokens_salida"]
        response = response["texto"].strip()
        
        return {"response": response}
        
    except Exception as e:
        return {"error": str(e)}, 500


# =================================================================
# NUEVO APARTADO: Acciones de QDRANT para borrar puntos 
# =================================================================
class BorrarChatRequest(BaseModel):
    collection_name: str
    chat_id: int

class BorrarPuntoRequest(BaseModel):
    collection_name: str
    point_id: str

@app.post("/borrar_chat", dependencies=[Depends(verificar_clave)])
def endpoint_borrar_chat(request: BorrarChatRequest):
    global client
    """Endpoint para borrar todo el historial de un chat por ID."""
    try:
        res = borrar_por_chat_id(client, request.collection_name, request.chat_id)
        return {"status": "success", "result": res.status}
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.post("/borrar_punto", dependencies=[Depends(verificar_clave)])
def endpoint_borrar_punto(request: BorrarPuntoRequest):
    """Endpoint para borrar un único punto por su ID."""
    try:
        res = borrar_por_point_id(client, request.collection_name, request.point_id)
        return {"status": "success", "result": res.status}
    except Exception as e:
        return {"status": "error", "message": str(e)}



#
# =================================================================
# NUEVO APARTADO: Para poner una nueva base de datos
# =================================================================

@app.post("/nueva-bd", dependencies=[Depends(verificar_clave)])
async def devai_endpoint(request: Request):
    form_data = await request.form()
    
    descripcion = form_data.get("descripcion", "")
    proyecto = form_data.get("proyecto", "")
    model_name = form_data.get("model_name", "models/gemini-3.1-flash-lite")

    archivos_procesados = []
    for key, value in form_data.items():
        if key.startswith("files[") and hasattr(value, "filename"):
            contenido_bytes = await value.read()
            archivos_procesados.append({
                "mime_type": value.content_type,   
                "data": contenido_bytes,
                "filename": value.filename
            })
    archivo = archivos_procesados[0]
    respuesta =  embebirBaseDatos(client, descripcion, archivo, proyecto)
	#print('respuesta')
	#print(respuesta)
    return {"response": respuesta}
