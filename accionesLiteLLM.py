import asyncio
import base64
import json
import os
from litellm import acompletion, RateLimitError, APIError
from langsmith import traceable

from funciones import debug

@traceable(run_type="chain", name="Lite_LLM_Agent_Stream")
async def generate_response_litellm_streaming(prompt: str, model_name: str, proxy_key: str, proxy_url:str, archivos: list = None, configuracion: dict = None, tools_schemas: list = None, tool_functions: dict = None, system_instruction: str = None, history: list = None):
    debug(f"🤖 [LITELLM] Ejecutando modelo: {model_name} " )

    extra_kwargs = {
        "stream_options": {"include_usage": True}
    }
    
    if configuracion:
        if 'tipo' in configuracion and configuracion['tipo'] == 'application/json':
            extra_kwargs['response_format'] = { "type": "json_object" }
        extra_kwargs['temperature'] = configuracion.get('temperature', 0.05)

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
            mime_tipo = arc.get("mime_type", "")
            data = arc.get("data", "")
            
            if mime_tipo.startswith("text/") or "php" in mime_tipo or "javascript" in mime_tipo or "json" in mime_tipo:
                try:
                    file_text = base64.b64decode(data).decode('utf-8')
                except Exception:
                    file_text = str(data)
                user_content.append({
                    "type": "text",
                    "text": f"\n\n--- Contenido de archivo adjunto ---\n{file_text}"
                })
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
        else:
            messages.append({"role": "user", "content": user_content})

    tokens_entrada_total = 0
    tokens_salida_total = 0
    chainOfThought_history = []
    interaciones_actuales = 0
    max_iterations = 25
    modo_agente = bool(tools_schemas and tool_functions)

    while True:
        interaciones_actuales += 1
        if interaciones_actuales > max_iterations:
            debug(f"🛑 [AGENTE WARN] Límite de iteraciones alcanzado.")
            modo_agente = False 
            
        kwargs_llamada = {
            "model": model_name,
            "messages": messages,
            "api_key": proxy_key, 
            "api_base": proxy_url,
            "custom_llm_provider": "openai",
            "stream": True,
            **extra_kwargs
        }
        
        if modo_agente:
            kwargs_llamada["tools"] = tools_schemas
            kwargs_llamada["tool_choice"] = "auto"

        try:
            response_stream = await acompletion(**kwargs_llamada)
            
            textoTurnoActual = ""
            tool_calls_dict = {} 

            async for chunk in response_stream:
                debug('UNchunk')
                debug(chunk)
                usage = getattr(chunk, "usage", None)
                if usage:
                    tokens_entrada_total = chunk.usage.prompt_tokens
                    tokens_salida_total = chunk.usage.completion_tokens
                
                choices = getattr(chunk, "choices", [])
                if len(choices) > 0:
                    delta = chunk.choices[0].delta
                    
                    # Texto normal
                    if delta.content:
                        textoTurnoActual += delta.content
                        yield {
                            "type": "token",
                            "content": delta.content
                        }
                    
                    delta_tool_calls = getattr(delta, "tool_calls", None)
                    if delta_tool_calls:
                        for tc in delta.tool_calls:
                            idx = tc.index
                            if idx not in tool_calls_dict:
                                tool_calls_dict[idx] = {
                                    "id": tc.id,
                                    "type": "function",
                                    "function": {
                                        "name": tc.function.name if tc.function.name else "",
                                        "arguments": tc.function.arguments if tc.function.arguments else ""
                                    }
                                }
                            else:
                                if tc.function.name:
                                    tool_calls_dict[idx]["function"]["name"] += tc.function.name
                                if tc.function.arguments:
                                    tool_calls_dict[idx]["function"]["arguments"] += tc.function.arguments

        except (RateLimitError, APIError) as e:
            debug(f"❌ [LITELLM ERROR]: {str(e)}")
            yield {
                "type": "error",
                "content": f"Error en la API: {str(e)}"
            }
            break
        except Exception as e:
            debug(f"❌ [ERROR GENERAL]: {str(e)}")
            yield {
                "type": "error",
                "content": f"Excepción durante la conexión: {str(e)}"
            }
            break

        tool_calls = list(tool_calls_dict.values()) if tool_calls_dict else None
        
        # Guardamos la respuesta del LLM en el historial
        aiMessageToHistory = {
            "role": "assistant",
            "content": textoTurnoActual if textoTurnoActual else None
        }
        if tool_calls:
            aiMessageToHistory["tool_calls"] = tool_calls
        messages.append(aiMessageToHistory)

        if tool_calls:
            debug("\n🗺️  [TRAZA DE PASOS Y RAZONAMIENTO DEL AGENTE]")
            
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

                yield {
                    "type": "thought",
                    "content": f"🧠 Usando la herramienta `{func_name}`."
                }

                if func_name in tool_functions:
                    function_to_call = tool_functions[func_name]
                    
                    if asyncio.iscoroutinefunction(function_to_call):
                        function_response = await function_to_call(**func_args)
                    else:
                        function_response = function_to_call(**func_args)
                    
                    yield {
                        "type": "thought",
                        "content": f"⚙️ Ejecuté: `{func_name}` con éxito."
                    }

                    content_str = function_response if isinstance(function_response, str) else json.dumps(function_response, ensure_ascii=False)                    
                    chainOfThought_history.append(paso_cot)

                    # Anexamos el resultado de la herramienta al historial
                    messages.append({
                        "role": "tool",
                        "tool_call_id": func_id,
                        "name": func_name,
                        "content": content_str
                    })
                else:
                    messages.append({
                        "role": "tool",
                        "tool_call_id": func_id,
                        "name": func_name,
                        "content": '{"error": "Función no registrada."}'
                    })
            
            # Continuamos el ciclo para mandar los resultados al LLM
            continue
        else:
            # Si no hubo tool calls, el flujo terminó con una respuesta de texto final.
            break

    yield {
        "type": "metrics",
        "tokens_entrada": tokens_entrada_total,
        "tokens_salida": tokens_salida_total,
        "chain_of_thought": chainOfThought_history
    }