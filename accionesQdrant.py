# accionesQdrant.py
import uuid
from qdrant_client import QdrantClient
from qdrant_client.http.models import Filter, FieldCondition, MatchValue

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

def search_in_qdrant(client, collection_name, query_embedding, k=5):
    results = client.query_points(
        collection_name=collection_name,
        query=query_embedding,
        limit=k,
        )

    return results.points 

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