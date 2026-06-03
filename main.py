# -*- coding: utf-8 -*-
import time
import os
import google.generativeai as genai
from qdrant_client import QdrantClient
from qdrant_client.http.models import Filter, FieldCondition, MatchValue, PointStruct
import uuid
from datetime import datetime
from fastapi import FastAPI, Request, UploadFile
from pydantic import BaseModel
import uvicorn
from typing import Optional
import json

#para el borrado de puntos
from accionesQdrant import borrar_por_chat_id, borrar_por_point_id

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
KEY_FREE = os.environ.get("KEY_FREE") 
KEY_FREE2 = os.environ.get("GOOGLE_API_KEY2") 
GOOGLE_API_KEY= os.environ.get('KEY-FREE') 


# Configure Gemini globally with GOOGLE_API_KEY for embedding operations
genai.configure(api_key=GOOGLE_API_KEY)


# Retrieve Qdrant API key from Colab secrets
try:
    qdrant_api_key = QDRANT_API_KEY
    print("Qdrant API Key obtenida de los secretos de Colab.")
except:
    qdrant_api_key = None
    print("No se encontró 'QDRANT_API_KEY' en los secretos de Colab. Si tu instancia de Qdrant requiere una API Key, asegúrate de haberla guardado correctamente.")

# Conecta con tu Qdrant (local o remoto)
client = QdrantClient(
    url= QDRANT_URL,  # o tu URL remota
    api_key=qdrant_api_key # Pasa la API key durante la inicialización
)


top_k = 5


class ChatTurn(BaseModel):
    user: str
    assistant: str


def optimizar_y_aplanar_historial(historial: Any, max_tokens: int):
    """
    Parsea, limpia y aplana el historial sin importar si viene como 
    String JSON o como lista de diccionarios desde PHP.
    """
    # Si viene como un String de texto debido a un json_encode en PHP, lo convertimos a lista
    if isinstance(historial, str):
        try:
            historial = json.loads(historial)
        except Exception:
            return [], 0  # Si el JSON está mal formado, devolvemos historial vacío
            
    # Si después de intentar parsearlo no es una lista, abortamos pacíficamente
    if not isinstance(historial, list):
        return [], 0

    encoding = tiktoken.get_encoding("cl100k_base")
    historial_plano_final = []
    tokens_acumulados = 0
    
    # Iteramos los turnos de atrás hacia adelante (del más nuevo al más viejo)
    for turno in reversed(historial):
        
        # Nos aseguramos de que el elemento sea un diccionario/objeto válido
        if isinstance(turno, dict):
            u_text = turno.get("user", "")
            a_text = turno.get("assistant", "")
        else:
            # Si hay basura dentro de la lista, la saltamos
            continue
            
        # Calculamos tokens
        tokens_user = len(encoding.encode(str(u_text))) + 4
        tokens_assistant = len(encoding.encode(str(a_text))) + 4
        tokens_turno = tokens_user + tokens_assistant
        
        # Si supera el límite de tokens pasados desde PHP, cortamos el pasado
        if tokens_acumulados + tokens_turno > max_tokens:
            break
            
        # Estructuramos al formato plano clásico que le gusta a Gemini/OpenAI
        componentes_turno = [
            {"role": "user", "content": u_text},
            {"role": "assistant", "content": a_text}
        ]
        
        # Los inyectamos al principio para mantener el orden cronológico
        historial_plano_final = componentes_turno + historial_plano_final
        tokens_acumulados += tokens_turno
        
    return historial_plano_final



def embed_with_gemini(text, dimension=3072):
    """Devuelve un embedding del texto usando Gemini."""
    # genai.embed_content usará la clave globalmente configurada (GOOGLE_API_KEY)
    res = genai.embed_content(
        model='models/gemini-embedding-001',
        content=text,
        task_type="retrieval_document",
        output_dimensionality=dimension
    )
    return res["embedding"] if "embedding" in res else None

def search_in_qdrant(client, collection_name, query_embedding, k=top_k):
    """Busca los k chunks más relevantes en Qdrant para el embedding dado."""
    print("Buscando en ")
    print(collection_name)
    results = client.query_points(
        collection_name=collection_name,
        query=query_embedding,
        limit=k,
        )

    return results.points # This line was modified to access the 'points' attribute





def guardar_memoria_en_qdrant(client, embed_fn, user_query, collection_memory, respuesta, chat_id, proyecto="default"):
    """
    Guarda en Qdrant un turno de conversación (usuario + asistente) como memoria semántica.
    embed_fn: función que recibe texto y regresa embedding (por ejemplo, embed_with_gemini)
    """
    textos = [
        {"role": "user", "text": user_query.strip()},
        {"role": "assistant", "text": respuesta.strip()},
    ]

    points = []
    uuids = []
    for item in textos:
        emb = embed_fn(item["text"],768)
        if emb is None:
            continue
        unUUUID = uuid.uuid4()
        uuids.append(unUUUID)
        points.append(
            PointStruct(
                id=str(unUUUID),
                vector=emb,
                payload={
                    "text": item["text"],
                    "chat_id": chat_id,
                    "role": item["role"],
                    "project": proyecto,
                    "timestamp": datetime.utcnow().isoformat()
                }
            )
        )

    if not points:
        print("⚠️ No se generaron embeddings para guardar memoria.")
        return

    client.upsert(
        collection_name=collection_memory,
        points=points,
        wait=True
    )
    print(f"✅ Memoria guardada ({len(points)} puntos) para proyecto '{proyecto}'.")
    return uuids


def recuperar_memoria_proyecto(client, embed_fn, user_query, collection_memory, chat_id, proyecto="default", limit=5):
    """
    Recupera memoria relevante para un proyecto dado, usando query_points.
    Regresa una lista de puntos (ScoredPoint-like) que luego pasas a build_prompt_from_chunks como `memory`.
    """
    filtros = [
        FieldCondition(
            key="role",
            match=MatchValue(value="assistant")
        ),
        
    ]
    if proyecto:
        filtros.append(
            FieldCondition(
                key="project",
                match=MatchValue(value=proyecto)
            )
        )

    if chat_id:
        filtros.append(
            FieldCondition(
                key="chat_id",
                match=MatchValue(value=chat_id)
            )
        )
    query_emb = embed_fn(user_query,768)
    res = client.query_points(
        collection_name=collection_memory,
        query=query_emb,
        limit=limit,
        with_payload=True,
        with_vectors=False,
        query_filter=Filter(
            must=filtros
        )
    )
    puntos = res.points

    # ordenar cronológicamente
    puntos.sort(
        key=lambda x: x.payload.get("timestamp", 0)
    )
    print("puntos que trajo la BD")
    print(puntos)
    return puntos


def build_prompt_from_chunks(chunksCodigo, chunksBD, chunksArchivo, user_query, memory=None, historial = ''):
    contextCodigo = "\n\n---\n\n".join([
        chunk.payload["text"] for chunk in chunksCodigo
        if chunk.payload and "text" in chunk.payload
    ])

    contextBD = "\n\n---\n\n".join([
        chunk.payload["text"] for chunk in chunksBD
        if chunk.payload and "text" in chunk.payload
    ])

    contextoArchivo = "\n\n---\n\n".join([
        chunk.payload["text"] for chunk in chunksArchivo
        if chunk.payload and "text" in chunk.payload
    ])

    memoria = ""
    if memory and len(memory) > 0:
        memoria = "\n\n".join([
            f"[{i+1}] {chunk.payload['text']}"
            for i, chunk in enumerate(memory)
            if chunk.payload and "text" in chunk.payload
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
        codigo_archivo = (
            "[ANALISIS] CONTEXTO DE ANÁLISIS:\n"
            + contextoArchivo +
            "\n\n---\n"
        )


    #print(memoria)
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



#def generate_response(prompt, model_name="gemini-2.5-flash"):
def generate_response(prompt, model_name="models/gemini-3-flash-preview", archivos: list = None):

    chat_model = genai.GenerativeModel(model_name)
    contenidos_payload = [prompt]
    if archivos:
        for arc in archivos:
            contenidos_payload.append({
                "mime_type": arc["mime_type"],
                "data": arc["data"]
            })
    convo = chat_model.start_chat()
    #response = convo.send_message(prompt)
    response = convo.send_message(contenidos_payload)
    tokens = chat_model.count_tokens((contenidos_payload + response.text))
    return response.text



def query_rag(user_query: str, memoria, chat_id:int, codigo, bd, archivo, proyecto: str = "default", model_name= "models/gemini-3-flash-preview", historial = '', max_tokens = 6000, archivos = None  ):
    t0 = time.time()

    try:

        #basedatos = data.get('basedatos', 'default')
        #codigo = data.get('codigo', false)

        if not user_query:
            return {'error': 'No se recibió un prompt válido'}, 400
        if not chat_id:
            return {'error': 'No se recibió un id de chat válido'}, 400


        # Step 1: embedding the user query
        
        query_embedding = embed_with_gemini(user_query)
        if query_embedding is None:
            return {'error': 'Failed to generate embedding for query'}, 500
        print("Embedding:", time.time() - t0)

        t1 = time.time()
        query_embedding768 = embed_with_gemini(user_query,768)
        if query_embedding is None:
            return {'error': 'Failed to generate embedding for query'}, 500
        print("Embedding2:", time.time() - t1)

        #DEVAI-embeddings

        #DevAI-Memory
        collection_memory = memoria
        # Step 2: retrieval from Qdrant
        t2 = time.time()
        chunksCodigo = search_in_qdrant(client, codigo, query_embedding, k=10)
        print("En codigo:", time.time() - t2)
        t3 = time.time()
        chunksBD = search_in_qdrant(client, bd, query_embedding768, k=10)
        print("En bd:", time.time() - t3)
        t4 = time.time()
        chunksArchivo = search_in_qdrant(client, archivo, query_embedding768, k=10)
        print("En Archivo:", time.time() - t4)

        print("Despues de hacer buscar en qdrant")
        t5 = time.time()
        # Step 2.5: retrieval of memory
        memory = recuperar_memoria_proyecto(
            client=client,
            embed_fn=embed_with_gemini,
            user_query=user_query,
            collection_memory=collection_memory,
            chat_id=chat_id,
            proyecto=proyecto,
            limit=8
        )
        print("Despues de hacer buscar en memoria:", time.time() - t5)
        # Step 3: build prompt

        historialModificado = optimizar_y_aplanar_historial(historial, max_tokens)

        prompt = build_prompt_from_chunks(chunksCodigo, chunksBD, chunksArchivo, user_query, memory, historialModificado)
        print('prompt:')
        print(prompt)
        # Configure Gemini for response generation (using KEY_FREE2)
        genai.configure(api_key=KEY_FREE2)
        t6 = time.time()
        # Step 4: generate response
        response_text = generate_response(prompt, model_name, archivos)
        print(response_text)
        # Configure Gemini back for embedding (using GOOGLE_API_KEY)
        genai.configure(api_key=GOOGLE_API_KEY)
        print("Despues de respuesta:", time.time() - t6)
        # Step 5: save conversation memory
        t7 = time.time()
        uuids = guardar_memoria_en_qdrant(
            client=client,
            embed_fn=embed_with_gemini,
            user_query=user_query,
            collection_memory=collection_memory,
            respuesta=response_text,
            chat_id=chat_id,
            proyecto=proyecto
        )
        print("Acaba:", time.time() - t7)
        return {'response': response_text, 'uuids' : uuids}, 200

    except Exception as e:
        return {'error': str(e)}, 500
	

app = FastAPI()

#Esta clase ya no se usa, agarramos los datos directo del request.
class QueryRequest(BaseModel):
    query: str
    memoria:str ="DevAI-Memory"
    chat_id:int
    codigo:str = "DEVAI-embeddings"
    bd:str = "DevAI-DB"
    archivo:str = "DevAI-Analisis" 
    proyecto: str = "default"
    model_name: str = "models/gemini-3-flash-preview"
    historial: str = ""
    max_tokens: int = 6000

@app.get("/health")
def health():
    return {
        "status": "ok"
    }

@app.post("/devai", dependencies=[Depends(verificar_clave)])
async def devai_endpoint(request: Request):
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

    # CAMBIO AQUÍ: Procesamos los archivos a un formato compatible con Gemini
    archivos_procesados = []
    for key, value in form_data.items():
        if key.startswith("files[") and isinstance(value, UploadFile):
            # Leemos los bytes de forma asíncrona
            contenido_bytes = await value.read()
            archivos_procesados.append({
                "mime_type": value.content_type,   # Ej: "image/png" o "application/pdf"
                "data": contenido_bytes           # Los bytes puros del archivo
            })

    # En este punto, 'archivos_recibidos' es una lista limpia de objetos UploadFile de FastAPI
    # [UploadFile(filename="archivo1.pdf", ...), UploadFile(filename="imagen.png", ...)]


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


# =================================================================
# NUEVO APARTADO: ENDPOINT PARA PROMPTS LIBRES (SIN RAG / QDRANT)
# =================================================================

class FreePromptRequest(BaseModel):
    prompt: str
    model_name: str 

def generate_free_response(prompt_text: str, model_name: str):
    genai.configure(api_key=KEY_FREE2)
    chat_model = genai.GenerativeModel(model_name)
    response = chat_model.generate_content(prompt_text)
    
    return response.text

@app.post("/prompt", dependencies=[Depends(verificar_clave)])
def free_prompt_endpoint(request: FreePromptRequest):
    """
    Endpoint para enviar cualquier prompt directo a Gemini.
    """
    try:
        if not request.prompt:
            return {"error": "No se recibió un prompt válido"}, 400
        if not request.model_name:
            return {"error": "No se recibió un modelo válido"}, 400
        
        print(f"Recibiendo prompt libre: {request.prompt}")
        respuesta_texto = generate_free_response(request.prompt, request.model_name)
        
        return {"response": respuesta_texto}
        
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


