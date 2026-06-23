import json
import os
import requests
from funciones import debug

def generate_response_chutes(prompt: str, model_name: str, api_key: str, archivos: list = None, configuracion: dict = None, tools_schemas: list = None, tool_functions: dict = None, system_instruction: str = None, history: list = None):
    """
    Genera una respuesta utilizando la API serverless de Chutes.ai mediante peticiones HTTP directas.
    Soporta modo Agente con orquestación manual (Function Calling Loop) y formato multimodal.
    """
    debug(f"🤖 [CHUTES] Ejecutando modelo: {model_name} " )
    
    url = "https://llm.chutes.ai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    # 1. Configuración base del Payload
    payload_config = {
        "stream": False # Mantener en False para el procesamiento del backend/agente
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
        
    if archivos:
        for arc in archivos:
            mime = arc.get("mime_type", "image/jpeg")
            data = arc.get("data")
            # Si los bytes vienen crudos del Form, los codificamos a string base64 si no lo están
            if isinstance(data, bytes):
                import base64
                data = base64.b64encode(data).decode('utf-8')
                
            user_content.append({
                "type": "image_url",
                "image_url": {"url": f"data:{mime};base64,{data}"}
            })

    if user_content:
        if len(user_content) == 1 and user_content[0]["type"] == "text":
            messages.append({"role": "user", "content": prompt})
        else:
            messages.append({"role": "user", "content": user_content})

    # Variables de control para las métricas
    tokens_entrada_total = 0
    tokens_salida_total = 0
    final_text = ""
    
    # 🛡️ CONTADOR DE ITERACIONES PARA EVITAR LOCURAS N+1
    interaciones_actuales = 0
    max_iterations = 6 

    debug("\n🗺️  Historial y messages")
    debug("──────────────────────────────────────────────────")
    debug(messages)
    debug("──────────────────────────────────────────────────")

    modo_agente = bool(tools_schemas and tool_functions)
    
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

        # Disparo HTTP a Chutes
        response = requests.post(url, json=body, headers=headers, timeout=60)
        
        if response.status_code != 200:
            debug(f"❌ [CHUTES ERROR {response.status_code}]: {response.text}")
            return {
                "texto": f"Error en el proveedor Chutes (HTTP {response.status_code})",
                "tokens_entrada": 0,
                "tokens_salida": 0,
                "status": "error"
            }

        res_data = response.json()
        choice = res_data["choices"][0]
        message_obj = choice["message"]
        
        # Acumulación de métricas de consumo provistas por Chutes
        usage = res_data.get("usage", {})
        tokens_entrada_total = usage.get("prompt_tokens", tokens_entrada_total)
        tokens_salida_total = usage.get("completion_tokens", tokens_salida_total)

        # Si el modelo no usó herramientas o no estamos en modo agente, terminamos de inmediato
        if not modo_agente:
            final_text = message_obj.get("content", "")
            break

        # Para el flujo del agente, guardamos el mensaje de la IA tal como nos llegó
        # Convertimos la estructura de vuelta a formato plano para el payload consecutivo
        ai_message_to_history = {
            "role": "assistant",
            "content": message_obj.get("content")
        }
        if "tool_calls" in message_obj:
            ai_message_to_history["tool_calls"] = message_obj["tool_calls"]
            
        messages.append(ai_message_to_history)

        # Verificamos si se invocaron herramientas
        tool_calls = message_obj.get("tool_calls")
        if tool_calls:
            debug("\n🗺️  [TRAZA DE PASOS Y RAZONAMIENTO DEL AGENTE - CHUTES]")
            debug("──────────────────────────────────────────────────")
            
            for tool_call in tool_calls:
                func_id = tool_call.get("id")
                func_meta = tool_call.get("function", {})
                func_name = func_meta.get("name")
                func_args = json.loads(func_meta.get("arguments", "{}"))
                
                debug(f"🧠 [LLM PENSÓ]: Requiero extraer datos contextuales.")
                debug(f"   ↳ 🛠️  Llamando a: '{func_name}'")
                debug(f"   ↳ 📋 Argumentos calculados: {func_args}\n")
                
                if func_name in tool_functions:
                    function_to_call = tool_functions[func_name]
                    # Ejecutamos la lógica local (ej: buscar en Qdrant)
                    function_response = function_to_call(**func_args)
                    
                    debug(f"⚙️  [PYTHON EJECUTÓ]: '{func_name}'")
                    debug(f"   ↳ 📥 Datos devueltos al LLM con éxito.\n")
                    
                    content_str = function_response if isinstance(function_response, str) else json.dumps(function_response, ensure_ascii=False)                    
    
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
            debug("──────────────────────────────────────────────────\n")
            # Continuamos el bucle "while" para enviar las respuestas de las herramientas a la IA
            continue
        else:
            # La IA no llamó a más herramientas, tenemos la respuesta final del agente
            final_text = message_obj.get("content", "")
            debug(f"💬 [CHUTES Agente (Respuesta Final)]: {final_text.strip()}\n")
            break

    debug(f"--- Info de la petición Chutes ---")
    debug(f"Tokens Entrada Acumulados: {tokens_entrada_total} | Tokens Salida Acumulados: {tokens_salida_total}")
    debug(f"───────────────────────────")

    return {
        "texto": final_text,
        "tokens_entrada": tokens_entrada_total,
        "tokens_salida": tokens_salida_total,
        "status": "success"
    }