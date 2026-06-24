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
    debug(f"modelo en generate: {model_name}" )
    
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

    # 🤖 MODO AGENTE: Si hay herramientas, usamos 'start_chat' para ejecución automática
    if tools:
        debug("🤖 [INFO] Modo Agente activado. Orquestando llamadas automáticas...")

        chat_model = genai.GenerativeModel(model_name=model_name, tools=tools, system_instruction=system_instruction)
        chat = chat_model.start_chat(history=history, enable_automatic_function_calling=True)

        try:
            response = await chat.send_message_async(contenidos_payload, generation_config=gen_config if gen_config else None)

        except googleExceptions.GoogleAPIError as e:
            # Captura errores oficiales de la API de Google (400, 429, 403, 500, etc.)
            statusCode = e.code if hasattr(e, "code") else 500
            errorMessage = e.message if hasattr(e, "message") else str(e)
            
            debug(f"❌ [GEMINI ERROR {statusCode}]: {errorMessage}")
            return {
                "texto": f"Error en el proveedor Gemini (HTTP {statusCode})",
                "tokens_entrada": 0,
                "tokens_salida": 0,
                "status": "error"
            }

        # 🔍 IMPRIMIR EL RAZONAMIENTO Y PASOS INTERMEDIOS DEL LLM
        debug("\n🗺️  [TRAZA DE PASOS Y RAZONAMIENTO DEL AGENTE]")
        debug("──────────────────────────────────────────────────")
        for mensaje in chat.history:
            for part in mensaje.parts:
                part_dict = type(part).to_dict(part) if hasattr(type(part), 'to_dict') else {}
                # Pasó 1: ¿El LLM decidió que necesitaba usar una herramienta?
                if 'function_call' in part_dict:
                    debug(f"🧠 [LLM PENSÓ]: Necesito extraer datos del sistema.")
                    debug(f"   ↳ 🛠️  Llamando a: '{part.function_call.name}'")
                    args = dict(part.function_call.args)
                    debug(f"   ↳ 📋 Argumentos calculados: {args}\n")
                
                # Paso 2: ¿Es la respuesta que tu código de Python (Qdrant) le inyectó de vuelta?
                elif 'function_response' in part_dict:
                    nombre_func = part_dict['function_response'].get('name', 'desconocida')
                    debug(f"⚙️  [PYTHON EJECUTÓ]: '{nombre_func}'")
                    debug(f"   ↳ 📥 Datos devueltos a Gemini con éxito.")
                    debug(f"   ↳ (Tu Qdrant ya le entregó el contexto a la IA)\n")
                
                # Paso 3: ¿Es texto plano? (Prompt inicial o respuesta final)
                elif 'text' in part_dict:
                    rol = "USUARIO (Prompt)" if mensaje.role == "user" else "GEMINI (Respuesta Final)"
                    debug(f"💬 [{rol}]: {part.text.strip()}\n")
        debug("──────────────────────────────────────────────────\n")

    # 📝 MODO NORMAL: Si no hay herramientas, se ejecuta el 'generate_content' clásico
    else:
        try:
            response = await chat_model.generate_content_async(
                contenidos_payload,
                generation_config=gen_config if gen_config else None
            )
        except googleExceptions.GoogleAPIError as e:
            # Captura errores oficiales de la API de Google (400, 429, 403, 500, etc.)
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
    
    debug(f"--- Info de la petición ---")
    debug(f"Tokens Entrada: {tokens_entrada} | Tokens Salida: {tokens_salida}")
    debug(f"───────────────────────────")

    return {
        "texto": response.text,
        "tokens_entrada": tokens_entrada,
        "tokens_salida": tokens_salida,
        "status" : "success"
    }