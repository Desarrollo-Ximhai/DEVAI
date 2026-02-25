# -*- coding: utf-8 -*-

import os
import google.generativeai as genai
from qdrant_client import QdrantClient
from qdrant_client.http.models import Filter, FieldCondition, MatchValue, PointStruct
import uuid
from datetime import datetime
from fastapi import FastAPI
from pydantic import BaseModel
import nest_asyncio
import uvicorn

QDRANT_URL = os.environ["QDRANT_URL"]
QDRANT_API_KEY = os.environ.get("QDRANT_API_KEY") 
KEY_FREE = os.environ.get("KEY_FREE") 
KEY_FREE2 = os.environ.get("GOOGLE_API_KEY2") 
GOOGLE_API_KEY= os.environ.get('KEY-FREE') 

# Get both API keys from Colab secrets
GOOGLE_API_KEY= userdata.get('KEY-FREE') # Asumiendo que esta es para embeddings
KEY_FREE2= userdata.get('GOOGLE_API_KEY2') # Asumiendo que esta es para embeddings
KEY_FREE= userdata.get('KEY-FREE') # Asumiendo que esta es para el LLM

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
        model='gemini-embedding-001',
        content=text,
        task_type="retrieval_document",
        output_dimensionality=dimension
    )
    return res["embedding"] if "embedding" in res else None

def search_in_qdrant(client, collction_name, query_embedding, k=top_k):
    """Busca los k chunks más relevantes en Qdrant para el embedding dado."""
    results = client.query_points(
        collection_name=collection_name,
        query=query_embedding,
        limit=k,
        )

    return results.points # This line was modified to access the 'points' attribute





def guardar_memoria_en_qdrant(client, embed_fn, user_query, respuesta, proyecto="default"):
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


def recuperar_memoria_proyecto(client, embed_fn, user_query, collection_memory, proyecto="default", limit=5):
    """
    Recupera memoria relevante para un proyecto dado, usando query_points.
    Regresa una lista de puntos (ScoredPoint-like) que luego pasas a build_prompt_from_chunks como `memory`.
    """
    query_emb = embed_fn(user_query,768)

    res = client.query_points(
        collection_name=collection_memory,
        query=query_emb,
        limit=limit,
        with_payload=True,
        with_vectors=False
    )

    # res.points es lo que tú ya usas
    return res.points


def build_prompt_from_chunks(chunks, user_query, memory=None):
    context = "\n\n---\n\n".join([
        chunk.payload["text"] for chunk in chunks
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
CONTEXTO DEL CÓDIGO:
{context}

---

PREGUNTA:
{user_query}

---

RESPUESTA:
"""
    return prompt.strip()



#def generate_response(prompt, model_name="gemini-2.5-flash"):
def generate_response(prompt, model_name="gemini-3-flash-preview"):

    chat_model = genai.GenerativeModel(model_name)
    convo = chat_model.start_chat()
    response = convo.send_message(prompt)
    tokens = chat_model.count_tokens((prompt + response.text))
    return response.text




nest_asyncio.apply()

app = FastAPI()

class QueryRequest(BaseModel):
    query: str
    proyecto: str = "default"

@app.post("/devai")
def devai_endpoint(request: QueryRequest):
    respuesta = query_rag(
        user_query=request.query,
        proyecto=request.proyecto
    )
    return {"response": respuesta}



def query_rag(user_query: str, proyecto: str = "default"):
    try:
       
        #basedatos = data.get('basedatos', 'default')
        #codigo = data.get('codigo', false)

        if not user_query:
            return {'error': 'Missing user_query in request body'}, 400

        # Step 1: embedding the user query
        query_embedding = embed_with_gemini(user_query)
        if query_embedding is None:
            return {'error': 'Failed to generate embedding for query'}, 500

        collection_name = "DEVAI-embeddings"

        collection_memory = "DevAI-Memory-CAPUFE"
        # Step 2: retrieval from Qdrant
        chunks = search_in_qdrant(client, collection_name, query_embedding, k=10)

        # Step 2.5: retrieval of memory
        memory = recuperar_memoria_proyecto(
            client=client,
            embed_fn=embed_with_gemini,
            user_query=user_query,
            collection_memory=collection_memory,
            proyecto=proyecto,
            limit=8
        )

        # Step 3: build prompt
        prompt = build_prompt_from_chunks(chunks, user_query, memory)

        # Configure Gemini for response generation (using KEY_FREE2)
        genai.configure(api_key=KEY_FREE2)

        # Step 4: generate response
        response_text = generate_response(prompt)

        # Configure Gemini back for embedding (using GOOGLE_API_KEY)
        genai.configure(api_key=GOOGLE_API_KEY)

        # Step 5: save conversation memory
        guardar_memoria_en_qdrant(
            client=client,
            embed_fn=embed_with_gemini,
            user_query=user_query,
            respuesta=response_text,
            proyecto=proyecto
        )

        return {'response': response_text}, 200

    except Exception as e:
        return {'error': str(e)}, 500

