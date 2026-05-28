# accionesQdrant.py
from qdrant_client import QdrantClient
from qdrant_client.http.models import Filter, FieldCondition, MatchValue

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