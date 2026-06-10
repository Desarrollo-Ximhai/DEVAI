# accionesQdrant.py
import uuid
import os
from qdrant_client import QdrantClient
from qdrant_client.http.models import Filter, FieldCondition, MatchValue, PointStruct, SparseVector, Prefetch, Fusion, FusionQuery
from qdrant_client.http import models 
from datetime import datetime
from fastembed.sparse import SparseTextEmbedding
from funciones import debug
sparse_model = SparseTextEmbedding(model_name="Qdrant/bm25")
import requests



def conectarQdrant(qdrant_url, qdrant_api_key):
    try:
        qdrant_api_key = qdrant_api_key
        print("Qdrant API Key obtenida de los secretos de Colab.")
    except:
        qdrant_api_key = None
        print("No se encontró 'QDRANT_API_KEY' en los secretos de Colab. Si tu instancia de Qdrant requiere una API Key, asegúrate de haberla guardado correctamente.")

    # Conecta con tu Qdrant (local o remoto)
    client = QdrantClient(
        url= qdrant_url,  # o tu URL remota
        api_key=qdrant_api_key # Pasa la API key durante la inicialización
    )
    return client

def borrar_por_chat_id(client: QdrantClient, collection_name: str, chat_id: int):
    """
    Borra todos los puntos en una colección que pertenezcan a un chat_id específico.
    """
    # Filtro para encontrar puntos donde el chat_id coincida
    filtro = Filter(
        must=[
            FieldCondition(
                key="chat_id",
                match=MatchValue(value=chat_id)
            )
        ]
    )
    
    resultado = client.delete(
        collection_name=collection_name,
        points_selector=filtro
    )
    return resultado

def borrar_por_point_id(client: QdrantClient, collection_name: str, point_id: str):
    """
    Borra un punto específico de la colección dado su ID único (UUID).
    """
    resultado = client.delete(
        collection_name=collection_name,
        points_selector=[point_id]
    )
    return resultado


def rerank_con_langsearch(query_usuario, candidatos, top_n=4):
    """
    Usa la API de LangSearch para reordenar los chunks de Qdrant.
    Consumo de RAM local = 0 MB.
    """
    RERANK_KEY= os.environ.get('RERANK_KEY') 
    RERANK_URL= os.environ.get('RERANK_URL') 

    if not candidatos:
        return []

    url = f"{RERANK_URL}"
    headers = {
        "Authorization": f"Bearer {RERANK_KEY}", 
        "Content-Type": "application/json"
    }

    # Extraemos solo las cadenas de texto limpias de los candidatos de Qdrant
    documentos = [
        c.payload.get("text", "") if hasattr(c, "payload") else c.get("text", "")
        for c in candidatos
    ]

    payload = {
        "model": "langsearch-reranker-v1",
        "query": query_usuario,
        "top_n": top_n,
        "return_documents": False, # No necesitamos que nos devuelva el texto, solo los índices
        "documents": documentos
    }

    try:
        response = requests.post(url, json=payload, headers=headers, timeout=5)
        if response.status_code == 200:
            res_data = response.json()
            
            # Re-mapeamos los índices ganadores a tus objetos originales de Qdrant
            chunks_finales = []
            for hit in res_data.get("results", []):
                idx = hit.get("index")
                if idx is not None and idx < len(candidatos):
                    chunks_finales.append(candidatos[idx])
            
            return chunks_finales
        else:
            print(f"⚠️ LangSearch respondió con error {response.status_code}, usando fallback.")
            return candidatos[:top_n]
            
    except Exception as e:
        print(f"⚠️ Falló la conexión con LangSearch: {e}")
        # Tu RAG no se muere si se cae la API, solo usa los primeros por defecto
        return candidatos[:top_n]


#Busqueda anterior, solo era sobre los vectores, ahora lo hacemos tambien de manera dispersa(palabras clave)
# def search_in_qdrant(client, collection_name, query_embedding, k=5):
#     results = client.query_points(
#         collection_name=collection_name,
#         query=query_embedding,
#         limit=k,
#         )

#     return results.points 



def search_in_qdrant(client, collection_name, user_query, query_embedding, proyecto, k):
    global sparse_model
    filtros = []
    if proyecto:
        filtros.append(
            FieldCondition(
                key="project",
                match=MatchValue(value=proyecto)
            )
        )
    
    sparse_emb = list(sparse_model.embed(user_query))[0]
    qdrant_sparse_vector = SparseVector(
        indices=sparse_emb.indices.tolist(),
        values=sparse_emb.values.tolist()
    )
    results = client.query_points(
        collection_name=collection_name,
        prefetch=[
            # Sub-petición 1: Búsqueda Semántica (Densa)
            Prefetch(query=query_embedding, limit=k),
            # Sub-petición 2: Búsqueda por Palabras Clave (Dispersa)
            Prefetch(query=qdrant_sparse_vector, using="text-sparse", limit=k)
        ],
        # El motor fusiona ambos rankings automáticamente usando RRF
        query=FusionQuery(fusion=Fusion.RRF),
        limit=k,
        query_filter=Filter(
            must=filtros
        )
    )
    debug(f"Busqueda en qdrant con k:{k}" )

    #Reranking
    return rerank_con_langsearch(user_query, results.points, 5) 

def save_to_qdrant(client, embed_fn, user_query, collection_memory, respuesta, chat_id, proyecto="default"):
    
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


def getProjectMemory(client, embed_fn, user_query, collection_memory, chat_id, proyecto="default", limit=5):
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
    # puntos.sort(
    #     key=lambda x: x.payload.get("timestamp", 0)
    # )
    
    return puntos