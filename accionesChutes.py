import asyncio
import base64
import httpx
import json
import os
from funciones import debug

async def generate_response_chutes_streaming(prompt: str, model_name: str, api_key: str, archivos: list = None, configuracion: dict = None, tools_schemas: list = None, tool_functions: dict = None, system_instruction: str = None, history: list = None):
    debug(f"🤖 [CHUTES] Ejecutando modelo: {model_name} " )
    
    url = "https://llm.chutes.ai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    # 1. Configuración base del Payload
    payload_config = {
        "stream": True,
        "stream_options": {"include_usage": True}
    }
    
    if configuracion:
        if 'tipo' in configuracion and configuracion['tipo'] == 'application/json':
            payload_config['response_format'] = { "type": "json_object" }
        payload_config['temperature'] = configuracion.get('temperature', 0.2)

    # 2. Reconstrucción del Historial de Mensajes
    messages = []
    
    if system_instruction:
        messages.append({"role": "system", "content": system_instruction})
        
    if history:
        messages.extend(list(history))

    # 3. Formateo del Mensaje Actual (Soporte Multimodal Base64)
    user_content = []
    if prompt:
        user_content.append({"type": "text", "text": prompt})


    debug('messages')        
    debug(messages)        
        

    if archivos:
        #debug('archivos')
        #debug(archivos)
        for arc in archivos:
            mime_tipo = arc.get("mime_type", "")
            data = arc.get("data", "")
            
            # 1. Manejo de texto y código (PHP, JS, Python, etc.)
            if mime_tipo.startswith("text/") or "php" in mime_tipo or "javascript" in mime_tipo or "json" in mime_tipo:
                try:
                    # Asumiendo que 'data' viene en base64 desde tu frontend/cliente
                    try:
                        file_text = base64.b64decode(data).decode('utf-8')
                    except Exception:
                        # Fallback por si la data ya venía como string de texto plano
                        file_text = str(data)
                    
                    user_content.append({
                        "type": "text",
                        "text": f"\n\n--- Contenido de archivo adjunto ---\n{file_text}"
                    })
                except Exception as e:
                    debug(f"⚠️ Error procesando archivo de texto: {e}")
                    
            # 2. Manejo de imágenes (Formato OpenAI Vision)
            elif mime_tipo.startswith("image/"):
                user_content.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:{mime_tipo};base64,{data}"
                    }
                })

    if user_content:
        if len(user_content) == 1 and user_content[0]["type"] == "text":
            messages.append({"role": "user", "content": prompt})
            debug('messages en chutes')
            debug(messages)
        else:
            messages.append({"role": "user", "content": user_content})

    # Variables de control para las métricas
    tokens_entrada_total = 0
    tokens_salida_total = 0
    final_text = ""
    chainOfThought_history = []
    # 🛡️ CONTADOR DE ITERACIONES PARA EVITAR LOCURAS N+1
    interaciones_actuales = 0
    max_iterations = 6 

    debug("\n🗺️  Historial y messages")
    debug("──────────────────────────────────────────────────")
    #debug(messages)
    debug("──────────────────────────────────────────────────")

    modo_agente = bool(tools_schemas and tool_functions)
    async with httpx.AsyncClient() as client:
        while True:
            interaciones_actuales += 1
            if interaciones_actuales > max_iterations:
                debug(f"🛑 [AGENTE WARN] Se alcanzó el límite de protección de {max_iterations} iteraciones. Forzando cierre.")
                # Le quitamos las herramientas en la última llamada para obligarlo a responder con lo que ya tiene
                modo_agente = False 
                
            body = {
                "model": model_name,
                "messages": messages,
                **payload_config
            }
            if modo_agente:
                body["tools"] = tools_schemas
                body["tool_choice"] = "auto"


            try:
                # Disparo HTTP a Chutes
                #response = requests.post(url, json=body, headers=headers, timeout=60) Response vieja sincrona
                #response = await client.post(url, json=body, headers=headers, timeout=60.0)
                async with client.stream("POST", url, json=body, headers=headers, timeout=60.0) as response:
                    if response.status_code != 200:
                        errorText = await response.aread()
                        debug(f"❌ [CHUTES ERROR {response.status_code}]: {errorText}")
                        yield {
                            "content": f"Error en el proveedor Chutes (HTTP {response.status_code})",
                            "type": "error"
                        }
                    print('response')
                    print(response)
                    textoTurnoActual = ""
                    toolCallsStream = {}

                    async for line in response.aiter_lines():
                        lineLimpia = line.strip()
                        if not lineLimpia or not lineLimpia.startswith("data: "):
                            continue
                        
                        dataStr = lineLimpia[6:].strip()
                        if dataStr == "[DONE]":
                            continue
                            
                        try:
                            resData = json.loads(dataStr)
                        except Exception:
                            continue
                        #print('resData')
                        #print(resData)
                        # A) Extracción de métricas de uso de tokens
                        usage = resData.get("usage")
                        if usage:
                            tokens_entrada_total = usage.get("prompt_tokens", 0)
                            tokens_salida_total = usage.get("completion_tokens", 0)

                        choices = resData.get("choices", [])
                        if not choices:
                            continue
                            
                        choice = choices[0]
                        delta = choice.get("delta", {})

                        # B) Flujo normal: El LLM está respondiendo con texto ordinario
                        content = delta.get("content")
                        if content:
                            textoTurnoActual += content
                            yield {
                                "type": "token",
                                "content": content
                            }

                        # C) Flujo Agente: El LLM está transmitiendo una intención de ejecución de herramienta
                        streamToolCalls = delta.get("tool_calls") or []
                        for tc in streamToolCalls:
                            idx = tc.get("index", 0)
                            # Si es el primer chunk de esta herramienta, inicializamos su estructura
                            if idx not in toolCallsStream:
                                toolCallsStream[idx] = {
                                    "id": tc.get("id"),
                                    "type": tc.get("type", "function"),
                                    "function": {
                                        "name": tc.get("function", {}).get("name", ""),
                                        "arguments": ""
                                    }
                                }
                            else:
                                # Los chunks posteriores rellenan paulatinamente el ID o el nombre si vienen vacíos
                                if tc.get("id"):
                                    toolCallsStream[idx]["id"] = tc.get("id")
                                if tc.get("function", {}).get("name"):
                                    toolCallsStream[idx]["function"]["name"] = tc.get("function", {}).get("name")
                            
                            # Concatenamos de forma lineal el string del JSON de argumentos
                            if tc.get("function", {}).get("arguments"):
                                toolCallsStream[idx]["function"]["arguments"] += tc["function"]["arguments"]

            except Exception as e:
                debug(f"❌ [HTTP/STREAM ERROR]: {str(e)}")
                yield {
                    "type": "error",
                    "content": f"Excepción durante la conexión con Chutes: {str(e)}"
                }

            # Verificamos si se invocaron herramientas
            tool_calls = list(toolCallsStream.values()) if toolCallsStream else None
            aiMessageToHistory = {
                "role": "assistant",
                "content": textoTurnoActual if textoTurnoActual else None
            }
            if tool_calls:
                aiMessageToHistory["tool_calls"] = tool_calls
            messages.append(aiMessageToHistory)

            if tool_calls:
                debug("\n🗺️  [TRAZA DE PASOS Y RAZONAMIENTO DEL AGENTE - CHUTES]")
                debug("──────────────────────────────────────────────────")
                
                for tool_call in tool_calls:
                    func_id = tool_call.get("id")
                    func_meta = tool_call.get("function", {})
                    func_name = func_meta.get("name")
                    try:
                        func_args = json.loads(func_meta.get("arguments", "{}"))
                    except Exception:
                        func_args = {}
                    
                    paso_cot = {
                        "tool": func_name,
                        "arguments": func_args,
                        "iteration": interaciones_actuales,
                    }

                    debug(f"🧠 [LLM PENSÓ]: Requiero extraer datos contextuales.")
                    debug(f"   ↳ 🛠️  Llamando a: '{func_name}'")
                    debug(f"   ↳ 📋 Argumentos calculados: {func_args}\n")

                    yield {
                        "type": "thought",
                        "content": f"🧠 Usando la herramienta `{func_name}`."
                    }

                    if func_name in tool_functions:
                        function_to_call = tool_functions[func_name]
                        # Ejecutamos la lógica local (ej: buscar en Qdrant)
                        #function_response = function_to_call(**func_args)

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

                        content_str = function_response if isinstance(function_response, str) else json.dumps(function_response, ensure_ascii=False)                    
                        #paso_cot["response"] = content_str
                        chainOfThought_history.append(paso_cot)
                        debug(f"   ↳ 📄 [CONTENIDO ENVIADO]: {content_str}\n")

                        messages.append({
                            "role": "tool",
                            "tool_call_id": func_id,
                            "name": func_name,
                            "content": content_str
                        })
                    else:
                        debug(f"⚠️ [ERROR]: La función '{func_name}' no se encuentra en el registro.")
                        messages.append({
                            "role": "tool",
                            "tool_call_id": func_id,
                            "name": func_name,
                            "content": '{"error": "Función no registrada en el agente."}'
                        })
                        yield {
                            "type": "thought",
                            "content": f"⚠️ La función '{func_name}' no está registrada en el sistema."
                        }
                debug("──────────────────────────────────────────────────\n")
                # Continuamos el bucle "while" para enviar las respuestas de las herramientas a la IA
                continue
            else:
                debug(f"💬 [CHUTES AGENTE]: Flujo de conversación completado con éxito.\n")
                break

        debug(f"--- Info de la petición Chutes ---")
        debug(f"Tokens Entrada Acumulados: {tokens_entrada_total} | Tokens Salida Acumulados: {tokens_salida_total}")
        #debug(chainOfThought_history)
        debug(f"───────────────────────────")

        yield {
            "type": "metrics",
            "tokens_entrada": tokens_entrada_total,
            "tokens_salida": tokens_salida_total,
            "chain_of_thought": chainOfThought_history
        }

