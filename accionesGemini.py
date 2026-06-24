import asyncio
import json 
from google.api_core import exceptions as googleExceptions
import google.generativeai as genai

from funciones import debug

def conectarGemini(key):
    debug("--Conectando Gemini")
    genai.configure(api_key=key)


async def embed_with_gemini(text, dimension=3072, tipo="retrieval_document"):
    res = await genai.embed_content_async(
        model='models/gemini-embedding-001',
        content=text,
        task_type=tipo,
        #task_type="retrieval_query",
        #task_type="retrieval_document",
        output_dimensionality=dimension
    )
    return res["embedding"] if "embedding" in res else None

async def generate_response(prompt, model_name, archivos: list = None, configuracion = None, tools: list = None, system_instruction=None, history:list = None):
    debug(f"modelo en generate: {model_name}")
    
    gen_config = {}

    if configuracion:
        if 'tipo' in configuracion:
            gen_config['response_mime_type'] = configuracion['tipo']
        
        gen_config['temperature'] = configuracion.get('temperature', 0.2)

    chat_model = genai.GenerativeModel(model_name=model_name, tools=tools)
    contenidos_payload = [prompt]
    
    if archivos:
        for arc in archivos:
            contenidos_payload.append({
                "mime_type": arc["mime_type"],
                "data": arc["data"]
            })

    # 🤖 MODO AGENTE: Orquestación manual (Function Calling Loop)
    if tools:
        debug("🤖 [INFO] Modo Agente activado. Orquestando llamadas manuales a Gemini...")

        # Iniciamos el chat SIN enable_automatic_function_calling
        chat_model = genai.GenerativeModel(model_name=model_name, tools=tools, system_instruction=system_instruction)
        chat = chat_model.start_chat(history=history)

        # Variables de control para el loop y métricas
        payload_actual = contenidos_payload
        max_iterations = 6
        interaciones_actuales = 0
        
        tokens_entrada_total = 0
        tokens_salida_total = 0
        final_text = ""

        while True:
            interaciones_actuales += 1
            if interaciones_actuales > max_iterations:
                debug(f"🛑 [AGENTE WARN] Se alcanzó el límite de protección de {max_iterations} iteraciones. Forzando cierre.")
                final_text = "Se ha alcanzado el límite de iteraciones en el razonamiento del agente."
                break

            try:
                # Enviamos el payload actual (puede ser el prompt inicial o las respuestas de las tools)
                response = await chat.send_message_async(
                    payload_actual, 
                    generation_config=gen_config if gen_config else None
                )
            except googleExceptions.GoogleAPIError as e:
                statusCode = e.code if hasattr(e, "code") else 500
                errorMessage = e.message if hasattr(e, "message") else str(e)
                
                debug(f"❌ [GEMINI ERROR {statusCode}]: {errorMessage}")
                return {
                    "texto": f"Error en el proveedor Gemini (HTTP {statusCode})",
                    "tokens_entrada": tokens_entrada_total,
                    "tokens_salida": tokens_salida_total,
                    "status": "error"
                }

            # Acumular tokens por cada ida y vuelta
            if hasattr(response, "usage_metadata") and response.usage_metadata:
                tokens_entrada_total += response.usage_metadata.prompt_token_count
                tokens_salida_total += response.usage_metadata.candidates_token_count

            # Verificamos si Gemini nos pide ejecutar herramientas en esta iteración
            function_calls = [part.function_call for part in response.parts if hasattr(part, 'function_call') and part.function_call]

            if not function_calls:
                # La IA no llamó a más herramientas, tenemos la respuesta final del agente
                final_text = response.text
                debug(f"💬 [GEMINI Agente (Respuesta Final)]: {final_text.strip()}\n")
                break

            # Limpiamos el payload para mandar SOLAMENTE las respuestas de las herramientas en la siguiente iteración
            payload_actual = []
            
            debug("\n🗺️  [TRAZA DE PASOS Y RAZONAMIENTO DEL AGENTE - GEMINI]")
            debug("──────────────────────────────────────────────────")
            
            for fc in function_calls:
                func_name = fc.name
                func_args = dict(fc.args)
                
                debug(f"🧠 [LLM PENSÓ]: Requiero extraer datos del sistema.")
                debug(f"   ↳ 🛠️  Llamando a: '{func_name}'")
                debug(f"   ↳ 📋 Argumentos calculados: {func_args}\n")
                
                # Gemini usa la lista original de funciones de Python. La buscamos por su atributo __name__
                function_to_call = next((f for f in tools if f.__name__ == func_name), None)
                
                if function_to_call:
                    # Ejecutamos la lógica local
                    if asyncio.iscoroutinefunction(function_to_call):
                        function_response = await function_to_call(**func_args)
                    else:
                        function_response = function_to_call(**func_args)
                        
                    debug(f"⚙️  [PYTHON EJECUTÓ]: '{func_name}'")
                    debug(f"   ↳ 📥 Datos devueltos al LLM con éxito.\n")
                    
                    # Gemini requiere que el campo 'response' sea un Diccionario (Struct de JSON)
                    if isinstance(function_response, str):
                        try:
                            res_dict = json.loads(function_response)
                        except:
                            res_dict = {"resultado": function_response}
                    elif isinstance(function_response, dict):
                        res_dict = function_response
                    else:
                        res_dict = {"resultado": str(function_response)}
                    
                    # Inyectamos la respuesta de la función en el payload que devolveremos
                    payload_actual.append({
                        "function_response": {
                            "name": func_name,
                            "response": res_dict
                        }
                    })
                else:
                    debug(f"⚠️ [ERROR]: La función '{func_name}' no se encuentra en el registro.")
                    payload_actual.append({
                        "function_response": {
                            "name": func_name,
                            "response": {"error": "Función no registrada en el agente."}
                        }
                    })
                    
            debug("──────────────────────────────────────────────────\n")
            # Continuamos el bucle "while" para que la IA procese 'payload_actual' (las respuestas de Python)

        debug(f"--- Info de la petición Gemini (Agente) ---")
        debug(f"Tokens Entrada Acumulados: {tokens_entrada_total} | Tokens Salida Acumulados: {tokens_salida_total}")
        debug(f"───────────────────────────")

        return {
            "texto": final_text,
            "tokens_entrada": tokens_entrada_total,
            "tokens_salida": tokens_salida_total,
            "status": "success"
        }

    # 📝 MODO NORMAL: Si no hay herramientas, se ejecuta el 'generate_content' clásico sin loop
    else:
        try:
            response = await chat_model.generate_content_async(
                contenidos_payload,
                generation_config=gen_config if gen_config else None
            )
        except googleExceptions.GoogleAPIError as e:
            statusCode = e.code if hasattr(e, "code") else 500
            errorMessage = e.message if hasattr(e, "message") else str(e)
            
            debug(f"❌ [GEMINI ERROR {statusCode}]: {errorMessage}")
            return {
                "texto": f"Error en el proveedor Gemini (HTTP {statusCode})",
                "tokens_entrada": 0,
                "tokens_salida": 0,
                "status": "error"
            }
    
        uso_tokens = response.usage_metadata
        tokens_entrada = uso_tokens.prompt_token_count
        tokens_salida = uso_tokens.candidates_token_count
        
        debug(f"--- Info de la petición (Modo Normal) ---")
        debug(f"Tokens Entrada: {tokens_entrada} | Tokens Salida: {tokens_salida}")
        debug(f"───────────────────────────")

        return {
            "texto": response.text,
            "tokens_entrada": tokens_entrada,
            "tokens_salida": tokens_salida,
            "status" : "success"
        }