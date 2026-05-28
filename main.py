# -*- coding: utf-8 -*-
import time
import os
import google.generativeai as genai
from qdrant_client import QdrantClient
from qdrant_client.http.models import Filter, FieldCondition, MatchValue, PointStruct
import uuid
from datetime import datetime
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn

#para el borrado de puntos
from accionesQdrant import borrar_por_chat_id, borrar_por_point_id

#para la autenticacion de la API
from fastapi import HTTPException, Header

def verificar_clave(x_api_key: str = Header(...)):
    if x_api_key != ADMIN_KEY:
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

    return puntos


def build_prompt_from_chunks(chunksCodigo, chunksBD, chunksArchivo, user_query, memory=None):
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
            "MEMORIA DE LA CONVERSACIÓN ANTERIOR:\n"
            + memoria +
            "\n\n---\n"
        )

    codigo_block = ""
    if contextCodigo:
        codigo_block = (
            "CONTEXTO DE CÓDIGO :\n"
            + contextCodigo +
            "\n\n---\n"
        )
    bd_block = ""
    if contextBD:
        bd_block = (
            "CONTEXTO DE BASE DE DATOS :\n"
            + contextBD +
            "\n\n---\n"
        )

    archivo_block = ""
    if contextoArchivo:
        codigo_archivo = (
            "CONTEXTO DE ANÁLISIS:\n"
            + contextoArchivo +
            "\n\n---\n"
        )


    #print(memoria)
    prompt = f"""
Eres un asistente de desarrollo extremadamente preciso y especializado en interpretar código PHP, HTML y SQL dentro de un framework personalizado.

A continuación tienes fragmentos REALES de código fuente del framework. No inventes ni completes nada que no esté explícitamente en el texto. No menciones de dónde salió el fragmento. No hagas suposiciones. Si no hay suficiente información para responder con certeza, responde claramente que no es posible responder.

INSTRUCCIONES:
- Usa solo lo que se encuentra en el contexto y en la memoria.
- Responde de forma concreta y profesional.
- No repitas el prompt ni resumas el contexto.
- En caso de las vistas no inventes inputs ni etiquetas HTML, utiliza siempre la clase Ximhai o los ejemplos de código para extraer datos.
- No generes estructuras incompletas.
- No menciones el nombre de los archivos ni rutas.

{memoria_block}
---

{codigo_block}

---
{bd_block}

---
{archivo_block}

---


PREGUNTA:
{user_query}

---

RESPUESTA:
"""
    return prompt.strip()



#def generate_response(prompt, model_name="gemini-2.5-flash"):
def generate_response(prompt, model_name="models/gemini-3-flash-preview"):

    chat_model = genai.GenerativeModel(model_name)
    convo = chat_model.start_chat()
    response = convo.send_message(prompt)
    tokens = chat_model.count_tokens((prompt + response.text))
    return response.text



def query_rag(user_query: str, memoria, chat_id:int, codigo, bd, archivo, proyecto: str = "default", model_name= "models/gemini-3-flash-preview"  ):
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
        prompt = build_prompt_from_chunks(chunksCodigo, chunksBD, chunksArchivo, user_query, memory)
        #print(prompt)
        # Configure Gemini for response generation (using KEY_FREE2)
        genai.configure(api_key=KEY_FREE2)
        t6 = time.time()
        # Step 4: generate response
        response_text = generate_response(prompt, model_name)
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
class QueryRequest(BaseModel):
    query: str
    memoria:str ="DevAI-Memory"
    chat_id:int
    codigo:str = "DEVAI-embeddings"
    bd:str = "DevAI-DB"
    archivo:str = "DevAI-Analisis" 
    proyecto: str = "default"
    model_name: str = "models/gemini-3-flash-preview"

@app.get("/health")
def health():
    return {
        "status": "ok"
    }

@app.post("/devai", dependencies=[Depends(verificar_clave)])
def devai_endpoint(request: QueryRequest):
	respuesta = query_rag(
		user_query=request.query,
		memoria=request.memoria,
		chat_id=request.chat_id,
		codigo=request.codigo,
		bd=request.bd,
		archivo=request.archivo,
		proyecto=request.proyecto,
        model_name=request.model_name
	)
	print('respuesta')
	print(respuesta)
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


