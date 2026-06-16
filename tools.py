
from accionesGemini import conectarGemini, generate_response, embed_with_gemini
from accionesQdrant import Qdrant, conectarQdrant
import os
QDRANT_URL = os.environ["QDRANT_URL"]
QDRANT_API_KEY = os.environ.get("QDRANT_API_KEY") 
client = conectarQdrant(QDRANT_URL, QDRANT_API_KEY)

# =================================================================
# Busqueda en QDRANT
# =================================================================

class AgenteTools:
    def __init__(self, objQdrant: Qdrant):
        """
        Recibe el motor de Qdrant ya configurado con el proyecto 
        y la colección de la petición actual.
        """
        self.ObjQdrant = objQdrant

    def buscar_conocimiento_base_datos(self, query: str) -> str:
        print(f"Buscando en base de datos con tool. {query}")
        """
        Busca esquemas de tablas, descripciones lógicas, relaciones de llaves foráneas 
        y lógica de negocio en la base de datos del proyecto actual.
        
        Args:
            query: Términos o conceptos de negocio a buscar (ej: "mantenimientos", "pagos").
        Returns:
            Un string en formato Markdown con las tablas y descripciones más relevantes encontradas.
        """
        query_embedding768 = embed_with_gemini(query,768, tipo= "retrieval_query")
        if query_embedding768 is None:
            return {'error': 'Failed to generate embedding 768 for query'}, 500


        puntos_ganadores = self.ObjQdrant.search_in_qdrant(user_query=query, k=40)
        
        # Mapeamos a texto limpio para la IA
        contexto_para_el_agente = []
        for p in puntos_ganadores:
            texto_chunk = p.payload.get("text", "")
            contexto_para_el_agente.append(texto_chunk)
            
        return "\n\n---\n\n".join(contexto_para_el_agente)