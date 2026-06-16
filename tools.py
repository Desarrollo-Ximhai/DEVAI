
from accionesGemini import conectarGemini, generate_response, embed_with_gemini
from accionesQdrant import Qdrant, conectarQdrant
# import os
# QDRANT_URL = os.environ["QDRANT_URL"]
# QDRANT_API_KEY = os.environ.get("QDRANT_API_KEY") 
# client = conectarQdrant(QDRANT_URL, QDRANT_API_KEY)

# =================================================================
# Busqueda en QDRANT
# =================================================================

class AgenteTools:
    def __init__(self, objQdrant: Qdrant):
        """
        Recibe el motor de Qdrant ya configurado con el proyecto 
        y la colección de la petición actual.
        """
        self.objQdrant = objQdrant

    def buscar_conocimiento_base_datos(self, conceptos_a_buscar: str) -> str:
        """
        Busca esquemas de tablas, descripciones lógicas, relaciones de llaves foráneas 
        y lógica de negocio en la base de datos del proyecto actual.
        
        Args:
            conceptos_a_buscar: Términos o conceptos de negocio a buscar (ej: "mantenimientos", "pagos").
        Returns:
            Un string en formato Markdown con las tablas y descripciones más relevantes encontradas.
        """
        # Invocamos al motor aislado
        puntos_ganadores = self.objQdrant.search_in_qdrant(user_query=conceptos_a_buscar, k=40)
        
        # Mapeamos a texto limpio para la IA
        contexto_para_el_agente = []
        for p in puntos_ganadores:
            texto_chunk = p.payload.get("text", "")
            contexto_para_el_agente.append(texto_chunk)
            
        return "\n\n---\n\n".join(contexto_para_el_agente)