# accionesQdrant.py
import uuid
from qdrant_client import QdrantClient
from qdrant_client.http.models import Filter, FieldCondition, MatchValue, PointStruct, SparseVector, Prefetch, Fusion, FusionQuery
from qdrant_client.http import models 
from datetime import datetime
from fastembed.sparse import SparseTextEmbedding
from flashrank import Ranker, RerankRequest
from main import debug

sparse_model = SparseTextEmbedding(model_name="Qdrant/bm25")
ranker = Ranker(model_name="ms-marco-MiniLM-L-6-v2")

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
    global ranker
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
    
    passages = [
        {
            "id": i,
            "text": chunk.payload["text"],
            "meta": chunk.payload  
        }
        for i, chunk in enumerate(results.points) if chunk.payload
    ]

    rerank_request = RerankRequest(query=user_query, passages=passages)
    results = ranker.rerank(rerank_request)
    nuevosPuntos = results[:5]
    
    return nuevosPuntos

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