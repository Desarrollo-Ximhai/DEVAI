import asyncio
import json 
from google.api_core import exceptions as googleExceptions
import google.generativeai as genai
from langsmith import traceable


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

@traceable(run_type="chain", name="Gemini_Agent_Stream")
async def generate_response_streaming(prompt, model_name, archivos: list = None, configuracion = None, tools: list = None, system_instruction=None, history:list = None):
    debug(f"modelo en generate: {model_name}")
    debug('history en gemini:')
    #debug(history)
    debug(archivos)
    
    gen_config = {}

    if configuracion:
        if 'tipo' in configuracion:
            gen_config['response_mime_type'] = configuracion['tipo']
        
    gen_config['temperature'] = 0.05
    chat_model = genai.GenerativeModel(model_name=model_name, tools=tools)
    contenidos_payload = [prompt]
    
    if archivos:
        for arc in archivos:
            mime_tipo = arc["mime_type"]
            
            # 💡 BLINDAJE: Si es cualquier variante de texto/código (php, py, js, etc.), 
            # lo homologamos a 'text/plain' para que Gemini lo acepte sin chistar.
            if mime_tipo.startswith("text/") or "php" in mime_tipo:
                mime_tipo = "text/plain"

            contenidos_payload.append({
                "mime_type": mime_tipo,
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
        chainOfThought_history = []
        while True:
            interaciones_actuales += 1
            if interaciones_actuales > max_iterations:
                debug(f"🛑 [AGENTE WARN] Se alcanzó el límite de protección de {max_iterations} iteraciones. Forzando cierre.")
                yield {
                    "type": "error",
                    "content": "Se ha alcanzado el límite de iteraciones en el razonamiento del agente."
                }
                break

            try:
                # Enviamos el payload actual (puede ser el prompt inicial o las respuestas de las tools)
                response = await chat.send_message_async(
                    payload_actual, 
                    generation_config=gen_config if gen_config else None,
                    stream=True
                )
            except googleExceptions.GoogleAPIError as e:
                statusCode = e.code if hasattr(e, "code") else 500
                errorMessage = e.message if hasattr(e, "message") else str(e)
                
                debug(f"❌ [GEMINI ERROR {statusCode}]: {errorMessage}")
                yield {
                    "type": "error",
                    "content": f"Error en el proveedor Gemini (HTTP {statusCode}): {errorMessage}"
                }
                break

            has_tool_calls = False
            function_calls = []

            async for chunk in response:
                # 📊 CLAVE 3: El conteo de tokens. Gemini expone las métricas en el último chunk de cada stream.
                # Al usar += las acumulamos correctamente a lo largo de todo el ciclo de ejecución (bucle while).

                debug(chunk)

                if hasattr(chunk, "usage_metadata") and chunk.usage_metadata:
                    tokens_entrada_total += chunk.usage_metadata.prompt_token_count
                    tokens_salida_total += chunk.usage_metadata.candidates_token_count

                # Evaluamos si el chunk actual trae intenciones de ejecución de herramientas
                function_calls_chunk = []
                if hasattr(chunk, "parts") and chunk.parts:
                    function_calls_chunk = [part.function_call for part in chunk.parts if hasattr(part, 'function_call') and part.function_call]

                if function_calls_chunk:
                    has_tool_calls = True
                    for fc in function_calls_chunk:
                        function_calls.append(fc)
                        # Notificamos de inmediato al frontend el paso de razonamiento (Chain of Thought)
                        yield {
                            "type": "thought", 
                            "content": f"🧠 Usando la técnica: `{fc.name}`."
                        }
            
                # Si el chunk contiene texto y NO se han activado herramientas en esta llamada, es la respuesta definitiva
                elif hasattr(chunk, "text") and chunk.text and not has_tool_calls:
                    yield {
                        "type": "token", 
                        "content": chunk.text
                    }
            
            if not has_tool_calls:
                break


            # Limpiamos el payload para mandar SOLAMENTE las respuestas de las herramientas en la siguiente iteración
            payload_actual = []
            
            debug("\n🗺️  [TRAZA DE PASOS Y RAZONAMIENTO DEL AGENTE - GEMINI]")
            debug("──────────────────────────────────────────────────")
            
            for fc in function_calls:
                func_name = fc.name
                func_args = dict(fc.args)
                
                paso_cot = {
                    "tool": func_name,
                    "arguments": func_args,
                    "iteration": interaciones_actuales,
                    "response": None # Lo llenaremos tras ejecutar
                }

                debug(f"🧠 [LLM PENSÓ]: Requiero extraer datos del sistema.")
                debug(f"   ↳ 🛠️  Llamando a: '{func_name}'")
                debug(f"   ↳ 📋 Argumentos calculados: {func_args}\n")

                yield {
                    "type": "thought", 
                    "content": f"🧠 Usando la herramienta `{func_name}`."
                }

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

                    yield {
                        "type": "thought", 
                        "content": f"⚙️ Ejecuté: `{func_name}` con éxito. Viendo datos..."
                    }
                    
                    # 1. Blindaje: Asegurar que el resultado SIEMPRE sea un diccionario (Struct de Protobuf lo requiere)
                    if isinstance(function_response, dict):
                        res_dict = function_response
                    elif isinstance(function_response, str):
                        try:
                            parsed = json.loads(function_response)
                            # json.loads puede devolver una lista u otros tipos, validamos:
                            if isinstance(parsed, dict):
                                res_dict = parsed
                            else:
                                res_dict = {"resultado": parsed}
                        except Exception:
                            res_dict = {"resultado": function_response}
                    else:
                        res_dict = {"resultado": str(function_response)}
                    
                    paso_cot["response"] = res_dict
                    chainOfThought_history.append(paso_cot)

                    parte_respuesta = genai.protos.Part(
                        function_response=genai.protos.FunctionResponse(
                            name=func_name,
                            response=res_dict
                        )
                    )
                    payload_actual.append(parte_respuesta)
                else:
                    debug(f"⚠️ [ERROR]: La función '{func_name}' no se encuentra en el registro.")
                    yield {
                        "type": "thought",
                        "content": f"⚠️ [ERROR]: La función '{func_name}' no se encuentra registrada. Buscando otra opcion..."
                    }
                    parte_error = genai.protos.Part(
                        function_response=genai.protos.FunctionResponse(
                            name=func_name,
                            response={"error": "Función no registrada en el agente."}
                        )
                    )
                    payload_actual.append(parte_error)
                    
            debug("──────────────────────────────────────────────────\n")
            # Continuamos el bucle "while" para que la IA procese 'payload_actual' (las respuestas de Python)

        debug(f"--- Info de la petición Gemini (Agente) ---")
        debug(f"Tokens Entrada Acumulados: {tokens_entrada_total} | Tokens Salida Acumulados: {tokens_salida_total}")
        debug(f"───────────────────────────")
        debug(chainOfThought_history)
        yield {
            "type": "metrics",
            "tokens_entrada": tokens_entrada_total,
            "tokens_salida": tokens_salida_total,
            "chain_of_thought": chainOfThought_history
        }
    # 📝 MODO NORMAL: Si no hay herramientas, se ejecuta el 'generate_content' clásico sin loop
    else:
        try:
            response = await chat_model.generate_content_async(
                contenidos_payload,
                generation_config=gen_config if gen_config else None,
                stream=True
            )
        except googleExceptions.GoogleAPIError as e:
            statusCode = e.code if hasattr(e, "code") else 500
            errorMessage = e.message if hasattr(e, "message") else str(e)
            
            debug(f"❌ [GEMINI ERROR {statusCode}]: {errorMessage}")
            yield {
                "type": "error",
                "content": f"Error en el proveedor Gemini (HTTP {statusCode}): {errorMessage}"
            }
            return

        tokens_entrada = 0
        tokens_salida = 0
        async for chunk in response:
            if hasattr(chunk, "usage_metadata") and chunk.usage_metadata:
                tokens_entrada += chunk.usage_metadata.prompt_token_count
                tokens_salida += chunk.usage_metadata.candidates_token_count
                
            if hasattr(chunk, "text") and chunk.text:
                yield {
                    "type": "token", 
                    "content": chunk.text
                }

        yield {
            "type": "metrics",
            "tokens_entrada": tokens_entrada,
            "tokens_salida": tokens_salida
        }
        
        debug(f"--- Info de la petición (Modo Normal) ---")
        debug(f"Tokens Entrada: {tokens_entrada} | Tokens Salida: {tokens_salida}")
        debug(f"───────────────────────────")

        # return {
        #     "texto": response.text,
        #     "tokens_entrada": tokens_entrada,
        #     "tokens_salida": tokens_salida,
        #     "status" : "success"
        # }

@traceable
async def generate_response(prompt, model_name):
    debug(f"modelo en generate: {model_name}")

    chat_model = genai.GenerativeModel(model_name=model_name)
    contenidos_payload = [prompt]
    try:
        response = await chat_model.generate_content_async(
            contenidos_payload,
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