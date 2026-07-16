import asyncio
import httpx
import json
from langsmith import traceable
import os

from accionesGemini import embed_with_gemini
from accionesQdrant import Qdrant, conectarQdrant

from funciones import debug

XIMHAI_KEY = os.environ.get("XIMHAI_KEY")


# =================================================================
# Clase para tools de base de datos, para poder instanciar desde main
# =================================================================
class sqlTools:
    def __init__(self, objQdrant: Qdrant, url:str):
        """
        Recibe el motor de Qdrant ya configurado con el proyecto 
        y la colección de la petición actual.
        """
        self.ObjQdrant = objQdrant
        self.url = url

# =================================================================
# Busqueda en QDRANT
# =================================================================
    @traceable(run_type="tool", name="Ejecutar_Herramienta_QDSQL")
    async def buscar_conocimiento_base_datos(self, query: str) -> str:
        debug(f"Buscando en base de datos con tool. {query}")
        """
        Busca esquemas de tablas, descripciones lógicas, relaciones de llaves foráneas 
        y lógica de negocio en la base de datos del proyecto actual.
        
        Args:
            query: Términos o conceptos de negocio a buscar (ej: "mantenimientos", "pagos").
        Returns:
            Un string en formato Markdown con las tablas y descripciones más relevantes encontradas.
        """
        query_embedding768 = await embed_with_gemini(query,768, tipo= "retrieval_query")
        if query_embedding768 is None:
            return f"Error al generar embedding del query"


        puntos_ganadores = await self.ObjQdrant.search_in_qdrant(user_query=query, query_embedding=query_embedding768, k=40, top_n=10)
        
        # Mapeamos a texto limpio para la IA
        contexto_para_el_agente = []
        for p in puntos_ganadores:
            texto_chunk = p.payload.get("text", "")
            contexto_para_el_agente.append(texto_chunk)
            
        return "\n\n---\n\n".join(contexto_para_el_agente)

# =================================================================
# Busqueda en Few Shots
# =================================================================
    @traceable(run_type="tool", name="Buscar_Ejemplos_Few_Shots")
    async def buscar_ejemplos_few_shots(self, query: str) -> str:
        debug(f"Buscando ejemplos few-shots con tool. {query}")
        """
        Busca ejemplos históricos (few-shots) de cómo el sistema ha resuelto exitosamente 
        peticiones similares en el pasado. Útil para entender qué herramientas usar, cómo 
        encadenarlas y cómo corregir errores SQL o lógicos.
        
        Args:
            query: La intención o pregunta actual del usuario (ej: "lista de lotes y dueños").
        Returns:
            Un string en formato Markdown detallando los flujos de resolución previos (Chain of Thought).
        """
        query_embedding768 = await embed_with_gemini(query, 768, tipo="retrieval_query")
        if query_embedding768 is None:
            return "Error al generar embedding del query para few-shots."

        # Se reduce top_n a 2 o 3 para no reventar los tokens del LLM con JSONs largos
        puntos_ganadores = await self.ObjQdrant.search_in_qdrant(
            user_query=query, 
            query_embedding=query_embedding768, 
            k=15, 
            top_n=2 
        )
        
        if not puntos_ganadores:
            return "No se encontraron ejemplos de resolución previos para esta consulta."

        ejemplos_para_el_agente = []
        for i, p in enumerate(puntos_ganadores, 1):
            payload = p.payload
            
            # Extraemos los datos basados en la estructura del payload
            user_query = payload.get("user_query", "Sin query registrado")
            final_response = payload.get("final_response", "Sin respuesta final registrada")
            cot = payload.get("chain_of_thought", [])
            
            # Convertimos el arreglo/diccionario a un string JSON formateado para que el LLM lo lea bien
            cot_str = json.dumps(cot, indent=2, ensure_ascii=False) if isinstance(cot, (list, dict)) else str(cot)
            
            ejemplo_texto = (
                f"### EJEMPLO DE REFERENCIA {i} ###\n"
                f"**Pregunta del usuario:** {user_query}\n\n"
                f"**Proceso lógico y uso de herramientas (Chain of Thought):**\n"
                f"```json\n{cot_str}\n```\n\n"
                f"**Respuesta final generada:** {final_response}\n"
            )
            ejemplos_para_el_agente.append(ejemplo_texto)
            
        return "\n\n---\n\n".join(ejemplos_para_el_agente)

# =================================================================
# Hacer request a PHP
# =================================================================}
    @traceable(run_type="tool", name="Ejecutar_Herramienta_PHPSQL")
    async def ejecutar_consulta_php(self, sql: str) -> str:
        """
        Ejecuta exclusivamente sentencias SQL de tipo SELECT para recuperar datos reales de las filas.
        
        PROHIBIDO: No intentes usar comandos de inspección de esquemas como DESCRIBE, SHOW TABLES, 
        SHOW COLUMNS, EXPLAIN o similares. Si no conoces el nombre de una tabla o columna, es obligatorio 
        que uses primero la herramienta 'buscar_conocimiento_base_datos' para conocer el esquema.
        
        Args:
            sql: Sentencia SQL que inicie estrictamente con 'SELECT'.
        Returns:
            JSON con los registros o string de error.
        """
        debug(f"🚀 [Tool PHP] Ejecutando consulta solicitada por el Agente:\n👉 {sql}\n")
        
        headers = {
            "Authorization": f"Bearer {XIMHAI_KEY}",
            "Content-Type": "application/json"
        }
        payload = {"query": sql}
        
        try:
            # Añadimos un timeout estricto para que el agente no se quede colgado si PHP tarda
            async with httpx.AsyncClient() as client:

                #Este servicio solo puede ejecutar consultas que empiecen con SELECT, de lo contrario retorna una advertencia
                response = await client.post(f"{self.url}/pruebaApi.php", json=payload, headers=headers, timeout=30.0)
            
            # Si PHP responde con códigos HTTP de error (500, 400, etc.)
            if response.status_code != 200:
                return f"ERROR_SISTEMA: El servidor PHP respondió con código de estado HTTP {response.status_code}. Detalle: {response.text}"
            
            data_php = response.json()
            
            # Si tu script de PHP atrapó un error de SQL y mandó {"status": "error", "message": "..."}
            if data_php.get("status") == "error":
                return f"ERROR_SQL_PHP: {data_php.get('message')}"
                
            resultados = data_php.get("resultado", [])
            # if len(resultados) > 100:
            #     # Truncamos para no saturar la ventana de contexto de Gemini
            #     resultados_truncados = resultados[:100]
            #     return json.dumps({
            #         "aviso": f"Se encontraron {len(resultados)} registros. Mostrando solo los primeros 100 por seguridad de memoria.",
            #         "data": resultados_truncados
            #     }, ensure_ascii=False)
            
            return json.dumps(resultados, ensure_ascii=False)
            
        except httpx.TimeoutException:
            return "ERROR_TIMEOUT: La consulta tardó demasiado en ejecutarse en el servidor PHP. Intenta optimizar los filtros o el LIMIT."
        except Exception as e:
            return f"ERROR_CONEXION: No se pudo comunicar con el endpoint de PHP. Detalle: {str(e)}"

# =================================================================
# Clase para tools de codigo, para poder instanciar desde main
# =================================================================

class codigoTools:
    def __init__(self, objQdrant: Qdrant):
        """
        Recibe el motor de Qdrant ya configurado con el proyecto 
        y la colección de la petición actual.
        """
        self.ObjQdrant = objQdrant

    # =================================================================
    # Busqueda en QDRANT
    # =================================================================
    @traceable(run_type="tool", name="Ejecutar_Herramienta_QDCODIGO")
    async def buscar_conocimiento_fragmentos_codigo(self, query: str) -> str:
        debug(f"🔍 [TOOL] Buscando en el Framework de Código: '{query}'")
        """
        Busca fragmentos de código, clases, métodos, controladores y funciones del 
        framework del proyecto actual para entender cómo programar o interactuar con el sistema.
        
        Args:
            query: Concepto técnico, nombre de clase o funcionalidad a buscar (ej: "cómo hacer un select", "clase ObjAjuste", "conectar BD").
        Returns:
            Un string en formato Markdown con las funciones y bloques de código más relevantes del framework.
        """
        query_embedding = await embed_with_gemini(query, tipo= "retrieval_query")
        if query_embedding is None:
            return f"Error al generar embedding del query"

        puntos_ganadores = await self.ObjQdrant.search_in_qdrant(user_query=query, query_embedding=query_embedding, k=40, top_n=15)
        
        # Mapeamos a texto limpio para la IA
        contexto_para_el_agente = []
        for p in puntos_ganadores:
            texto_chunk = p.payload.get("text", "")
            contexto_para_el_agente.append(texto_chunk)
            
        return "\n\n---\n\n".join(contexto_para_el_agente)

class systemTools:
    def __init__(self, url:str):
        """
        Recibe el dominio del servidor de php del sistema
        """
        self.url = url

    @traceable(run_type="tool", name="Buscar_Herramientas_Personalizadas_PHP")
    async def buscar_herramientas_personalizadas_php(self) -> str:
        """
        Busca en el servidor backend qué funciones o herramientas personalizadas de negocio están disponibles 
        para ser ejecutadas, devolviendo sus nombres, descripciones y los parámetros exactos que requieren.
        
        OBLIGATORIO: Siempre debes usar esta herramienta antes de intentar ejecutar una acción 
        en el sistema si no conoces el nombre exacto de la función y su esquema de parámetros.
        
        Args:
            None
        Returns:
            JSON con el catálogo de funciones disponibles y el esquema (contrato) de sus parámetros.
        """
        debug(f"🔍 [Tool PHP] Buscando herramientas disponibles para:\n👉 {self.url}\n")
        
        headers = {
            "Authorization": f"Bearer {XIMHAI_KEY}",
            "Content-Type": "application/json"
        }
        
        # Indicamos a PHP que solo queremos consultar el catálogo con payload vacio
        payload = {
            
        }
        
        try:
            async with httpx.AsyncClient() as client:
                # Reemplaza 'gateway_ia.php' con el nombre de tu endpoint ruteador
                response = await client.post(f"{self.url}/funciones-devAI.php", json=payload, headers=headers, timeout=15.0)
            
            if response.status_code != 200:
                return f"ERROR_SISTEMA: El servidor PHP respondió con código HTTP {response.status_code}. Detalle: {response.text}"
            
            data_php = response.json()
            
            if data_php.get("status") == "error":
                return f"ERROR_CATALOGO_PHP: {data_php.get('message')}"
                
            resultados = data_php.get("resultado", [])
            
            if not resultados:
                return "Este sistema no cuenta con herramientas PHP personalizadas. Intenta con otras tools."
                
            return json.dumps(resultados, ensure_ascii=False)
            
        except httpx.TimeoutException:
            return "ERROR_TIMEOUT: El servidor PHP tardó demasiado en devolver el catálogo de herramientas."
        except Exception as e:
            return f"ERROR_CONEXION: No se pudo comunicar con el endpoint de PHP. Detalle: {str(e)}"

    @traceable(run_type="tool", name="Ejecutar_Herramienta_Personalizada_PHP")
    async def ejecutar_herramienta_personalizada_php(self, nombre_funcion: str, argumentos: dict) -> str:
        """
        Ejecuta una función específica en el backend de PHP. 
        
        PROHIBIDO: No inventes parámetros. Los argumentos enviados en el diccionario 'argumentos' 
        deben coincidir estrictamente con los tipos de datos y nombres requeridos por el contrato 
        obtenido previamente mediante la herramienta 'buscar_herramientas_personalizadas_php'.
        
        Args:
            nombre_funcion: El nombre exacto de la función a ejecutar (ej: 'aplicar_descuento_lote').
            argumentos: Un diccionario (JSON) con los parámetros requeridos por la función.
        Returns:
            JSON con el resultado de la ejecución exitosa o un string de error.
        """
        debug(f"🚀 [Tool PHP] Ejecutando la función '{nombre_funcion}' con argumentos:\n👉 {argumentos}\n")
        
        headers = {
            "Authorization": f"Bearer {XIMHAI_KEY}",
            "Content-Type": "application/json"
        }
        
        # Indicamos a PHP que queremos ejecutar y pasamos la función y sus argumentos
        payload = {
            "accion": "ejecutar",
            "funcion": nombre_funcion,
            "argumentos": argumentos
        }
        
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(f"{self.url}/funciones-devAI.php", json=payload, headers=headers, timeout=15.0)
            
            if response.status_code != 200:
                return f"ERROR_SISTEMA: El servidor PHP respondió con código HTTP {response.status_code}. Detalle: {response.text}"
            
            data_php = response.json()
            
            if data_php.get("status") == "error":
                return f"ERROR_EJECUCION_PHP: {data_php.get('message')}"
                
            # Asumiendo que tu PHP devuelve la respuesta final dentro de la llave "resultado"
            resultados = data_php.get("resultado", {})
            
            return json.dumps(resultados, ensure_ascii=False)
            
        except httpx.TimeoutException:
            return f"ERROR_TIMEOUT: La ejecución de la función '{nombre_funcion}' excedió el tiempo límite. Es posible que la operación se haya realizado, pero la confirmación falló."
        except Exception as e:
            return f"ERROR_CONEXION: No se pudo comunicar con el endpoint de PHP. Detalle: {str(e)}"