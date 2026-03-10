# -*- coding: utf-8 -*-

import os
import google.generativeai as genai
from qdrant_client import QdrantClient
from qdrant_client.http.models import Filter, FieldCondition, MatchValue, PointStruct
import uuid
from datetime import datetime
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn

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
    for item in textos:
        emb = embed_fn(item["text"],768)
        if emb is None:
            continue

        points.append(
            PointStruct(
                id=str(uuid.uuid4()),
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

    return res.points


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



def query_rag(user_query: str, memoria, chat_id:int, codigo, bd, archivo, proyecto: str = "default"  ):
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

        query_embedding768 = embed_with_gemini(user_query,768)
        if query_embedding is None:
            return {'error': 'Failed to generate embedding for query'}, 500


        print("Despues de hacer embedding")

        #DEVAI-embeddings

        #DevAI-Memory
        collection_memory = memoria
        # Step 2: retrieval from Qdrant

        chunksCodigo = search_in_qdrant(client, codigo, query_embedding, k=10)
        print("en codigo");
        chunksBD = search_in_qdrant(client, bd, query_embedding768, k=10)
        print("en bd");
        chunksArchivo = search_in_qdrant(client, archivo, query_embedding768, k=10)
        print("en archivo");

        print("Despues de hacer buscar en qdrant")

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
        
        print("Despues de hacer buscar en memoria")
        # Step 3: build prompt
        prompt = build_prompt_from_chunks(chunksCodigo, chunksBD, chunksArchivo, user_query, memory)
        #print(prompt)
        # Configure Gemini for response generation (using KEY_FREE2)
        genai.configure(api_key=KEY_FREE2)

        # Step 4: generate response
        response_text = generate_response(prompt)
        print(response_text)
        # Configure Gemini back for embedding (using GOOGLE_API_KEY)
        genai.configure(api_key=GOOGLE_API_KEY)

        # Step 5: save conversation memory

        guardar_memoria_en_qdrant(
            client=client,
            embed_fn=embed_with_gemini,
            user_query=user_query,
            collection_memory=collection_memory,
            respuesta=response_text,
            chat_id=chat_id,
            proyecto=proyecto
        )
        print('acabo')
        return {'response': response_text}, 200

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

@app.post("/devai")
def devai_endpoint(request: QueryRequest):
	respuesta = query_rag(
		user_query=request.query,
		memoria=request.memoria,
		chat_id=request.chat_id,
		codigo=request.codigo,
		bd=request.bd,
		archivo=request.archivo,
		proyecto=request.proyecto
	)
	print('respuesta')
	print(respuesta)
	return {"response": respuesta}







