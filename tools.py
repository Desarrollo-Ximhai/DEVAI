
from accionesGemini import conectarGemini, generate_response, embed_with_gemini
from accionesQdrant import Qdrant, conectarQdrant

import requests

# =================================================================
# Clase para tools de gemini, para poder instanciar desde main
# =================================================================
class AgenteTools:
    def __init__(self, objQdrant: Qdrant):
        """
        Recibe el motor de Qdrant ya configurado con el proyecto 
        y la colección de la petición actual.
        """
        self.ObjQdrant = objQdrant

# =================================================================
# Busqueda en QDRANT
# =================================================================
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


        puntos_ganadores = self.ObjQdrant.search_in_qdrant(user_query=query, query_embedding=query_embedding768, k=20)
        
        # Mapeamos a texto limpio para la IA
        contexto_para_el_agente = []
        for p in puntos_ganadores:
            texto_chunk = p.payload.get("text", "")
            contexto_para_el_agente.append(texto_chunk)
            
        return "\n\n---\n\n".join(contexto_para_el_agente)

# =================================================================
# Hacer request a PHP
# =================================================================
    def ejecutar_consulta_php(self, sql: str) -> str:
        """
        Ejecuta una consulta SQL puramente de tipo SELECT en el servidor de producción 
        para recuperar registros y datos reales y actuales del sistema.
        
        CRÍTICO: Usa esta herramienta SOLO cuando necesites conocer datos específicos de filas, 
        totales, conteos o registros que el usuario solicitó explícitamente.
        
        Args:
            sql: Sentencia SQL SELECT completa, limpia y válida (ej: 'SELECT nombre, saldo FROM clientes WHERE saldo > 10000 LIMIT 20;').
            
        Returns:
            Un string en formato JSON con las filas encontradas o un mensaje detallado si ocurrió un error.
        """
        print(f"🚀 [Tool PHP] Ejecutando consulta solicitada por el Agente:\n👉 {sql}\n")
        
        headers = {
            # "Authorization": f"Bearer {self.PHP_API_KEY}",
            "Content-Type": "application/json"
        }
        payload = {"query": sql}
        
        try:
            # Añadimos un timeout estricto para que el agente no se quede colgado si PHP tarda
            #response = requests.post(self.PHP_API_URL, json=payload, headers=headers, timeout=15)
            response = requests.post('https://devai.ximhai.com/pruebaApi.php', json=payload, headers=headers, timeout=15)
            
            # Si PHP responde con códigos HTTP de error (500, 400, etc.)
            if response.status_code != 200:
                return f"ERROR_SISTEMA: El servidor PHP respondió con código de estado HTTP {response.status_code}. Detalle: {response.text}"
            
            data_php = response.json()
            
            # Si tu script de PHP atrapó un error de SQL y mandó {"status": "error", "message": "..."}
            if data_php.get("status") == "error":
                return f"ERROR_SQL_PHP: {data_php.get('message')}"
                
            # --- EL GUARDRAIL MÁS IMPORTANTE: Control de volumen ---
            resultados = data_php.get("resultado", [])
            if len(resultados) > 100:
                # Truncamos para no saturar la ventana de contexto de Gemini
                resultados_truncados = resultados[:100]
                return json.dumps({
                    "aviso": f"Se encontraron {len(resultados)} registros. Mostrando solo los primeros 100 por seguridad de memoria.",
                    "data": resultados_truncados
                }, ensure_ascii=False)
                
            return json.dumps(resultados, ensure_ascii=False)
            
        except requests.exceptions.Timeout:
            return "ERROR_TIMEOUT: La consulta tardó demasiado en ejecutarse en el servidor PHP. Intenta optimizar los filtros o el LIMIT."
        except Exception as e:
            return f"ERROR_CONEXION: No se pudo comunicar con el endpoint de PHP. Detalle: {str(e)}"