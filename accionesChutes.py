import json
from openai import OpenAI
from funciones import debug 

# Cliente global de OpenAI apuntando a Chutes
client = None

def conectarChutes(key):
    global client
    client = OpenAI(
        api_key=key,
        base_url="https://llm.chutes.ai/v1"
    )

def generate_response_chutes(prompt, model_name="nombre-de-tu-modelo-en-chutes", archivos: list = None, configuracion = None, tools_schemas: list = None, tool_functions: dict = None, system_instruction=None, history: list = None):
    print('modelo en generate Chutes:', model_name)
    #hardcodeando 
    model_name = "zai-org/GLM-5.1-TEE"
    # Adaptar la configuración al estándar de OpenAI
    gen_config = {}
    if configuracion:
        if 'tipo' in configuracion and configuracion['tipo'] == 'application/json':
            gen_config['response_format'] = { "type": "json_object" }
        gen_config['temperature'] = configuracion.get('temperature', 0.2)

    messages = []
    
    if system_instruction:
        messages.append({"role": "system", "content": system_instruction})
        
    if history:
        messages.extend(list(history))

    user_content = []
    if prompt:
        user_content.append({"type": "text", "text": prompt})
        
    if archivos:
        for arc in archivos:
            mime = arc.get("mime_type", "image/jpeg")
            data = arc.get("data")
            user_content.append({
                "type": "image_url",
                "image_url": {"url": f"data:{mime};base64,{data}"}
            })

    if user_content:
        if len(user_content) == 1 and user_content[0]["type"] == "text":
            messages.append({"role": "user", "content": prompt})
        else:
            messages.append({"role": "user", "content": user_content})

    last_response = None  # Guardará la última respuesta de la API para las métricas

    # 🤖 MODO AGENTE: Orquestación manual para function calling
    if tools_schemas and tool_functions:
        print("🤖 [INFO] Modo Agente activado en Chutes. Iniciando loop de orquestación...")
        
        while True:
            response = client.chat.completions.create(
                model=model_name,
                messages=messages,
                tools=tools_schemas,
                tool_choice="auto",
                **gen_config
            )
            
            last_response = response  # Guardamos la referencia para los tokens al final
            message = response.choices[0].message
            
            # ¡CRUCIAL!: Para que OpenAI no se rompa en la siguiente iteración, 
            # debemos añadir el mensaje completo (incluyendo el campo tool_calls nativo) al historial
            messages.append(message) 
            
            # Si el modelo decidió llamar a una herramienta
            if message.tool_calls:
                print("\n🗺️  [TRAZA DE PASOS Y RAZONAMIENTO DEL AGENTE]")
                print("──────────────────────────────────────────────────")
                
                for tool_call in message.tool_calls:
                    func_name = tool_call.function.name
                    func_args = json.loads(tool_call.function.arguments)
                    
                    print(f"🧠 [LLM PENSÓ]: Necesito extraer datos del sistema.")
                    print(f"   ↳ 🛠️  Llamando a: '{func_name}'")
                    print(f"   ↳ 📋 Argumentos calculados: {func_args}\n")
                    
                    if func_name in tool_functions:
                        function_to_call = tool_functions[func_name]
                        function_response = function_to_call(**func_args)
                        
                        print(f"⚙️  [PYTHON EJECUTÓ]: '{func_name}'")
                        print(f"   ↳ 📥 Datos devueltos al LLM con éxito.\n")
                        
                        # Si tu función ya retorna un string (o JSON stringificado), lo dejamos pasar directo.
                        # Si retorna un dict/list, lo convertimos a string.
                        content_str = function_response if isinstance(function_response, str) else json.dumps(function_response, ensure_ascii=False)
                        
                        # CORRECCIÓN DE FORMATO: Así se inyecta la respuesta al historial en OpenAI
                        messages.append({
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "name": func_name,
                            "content": content_str
                        })
                    else:
                        print(f"⚠️ [ERROR]: La función '{func_name}' no existe en el diccionario.")
                        messages.append({
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "name": func_name,
                            "content": '{"error": "Función no encontrada en el Agente de Herramientas."}'
                        })
                print("──────────────────────────────────────────────────\n")
                
                # Continuamos el ciclo "while" para que el modelo analice las respuestas que le inyectamos
                continue 
            
            else:
                # El modelo terminó su razonamiento y dio texto plano (Respuesta Final)
                print(f"💬 [CHUTES (Respuesta Final)]: {message.content.strip()}\n")
                break
                
    # 📝 MODO NORMAL: Sin herramientas
    else:
        response = client.chat.completions.create(
            model=model_name,
            messages=messages,
            **gen_config
        )
        last_response = response
        message = response.choices[0].message

    # Extracción segura de métricas
    try:
        tokens_entrada = last_response.usage.prompt_tokens
        tokens_salida = last_response.usage.completion_tokens
    except Exception:
        tokens_entrada = 0
        tokens_salida = 0
    
    print(f"--- Info de la petición ---")
    print(f"Tokens Entrada: {tokens_entrada} | Tokens Salida: {tokens_salida}")
    print(f"───────────────────────────")
    
    return {
        "texto": message.content,
        "tokens_entrada": tokens_entrada,
        "tokens_salida": tokens_salida,
        "status": "success"
    }