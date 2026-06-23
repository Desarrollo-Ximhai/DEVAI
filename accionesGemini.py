import json 
import google.generativeai as genai

from funciones import debug

def conectarGemini(key):
    debug("--Conectando Gemini")
    genai.configure(api_key=key)


def embed_with_gemini(text, dimension=3072, tipo="retrieval_document"):
    res = genai.embed_content(
        model='models/gemini-embedding-001',
        content=text,
        task_type=tipo,
        #task_type="retrieval_query",
        #task_type="retrieval_document",
        output_dimensionality=dimension
    )
    return res["embedding"] if "embedding" in res else None

def generate_response(prompt, model_name, archivos: list = None, configuracion = None, tools: list = None, system_instruction=None, history:list = None):
    debug('modelo en generate:', model_name)
    
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

        # system_instruction = """
        # Eres DEVAI, un asistente experto en ingeniería de datos.
        # CRÍTICO: NO conoces la estructura, tablas, llaves ni columnas de la base de datos actual del usuario. Todo tu conocimiento interno sobre este proyecto es CERO.
        # Por lo tanto, ante CUALQUIER pregunta del usuario que involucre tablas, dueños, lotes, consultas o lógica de negocio, es OBLIGATORIO que uses primero la herramienta 'buscar_conocimiento_base_datos'.
        # Está estrictamente prohibido adivinar o inventar nombres de tablas sin haber consultado la herramienta antes.
        # """

        

        chat_model = genai.GenerativeModel(model_name=model_name, tools=tools, system_instruction=system_instruction)
        chat = chat_model.start_chat(history=history, enable_automatic_function_calling=True)
        response = chat.send_message(contenidos_payload, generation_config=gen_config if gen_config else None)
        
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

        # historial_dict = [type(msg).to_dict(msg) for msg in chat.history]
        # debug("📁 JSON DEL HISTORIAL:")
        # debug(json.dumps(historial_dict, indent=2, ensure_ascii=False))

    # 📝 MODO NORMAL: Si no hay herramientas, se ejecuta el 'generate_content' clásico
    else:
        response = chat_model.generate_content(
            contenidos_payload,
            generation_config=gen_config if gen_config else None
        )

    # response = chat_model.generate_content(
    #     contenidos_payload,
    #     generation_config=gen_config if gen_config else None
    #     )
    
    uso_tokens = response.usage_metadata
    tokens_entrada = uso_tokens.prompt_token_count
    tokens_salida = uso_tokens.candidates_token_count
    
    debug(f"--- Info de la petición ---")
    debug(f"Tokens Entrada: {tokens_entrada} | Tokens Salida: {tokens_salida}")
    debug(f"───────────────────────────")
    #debug('Respuesta:')
    #debug(response.text)
    # # Opción A: Si solo necesitas el texto como antes, dejas esto:
    # return {"response": response.text, tokens_entrada}
    # return response.text
    
    return {
        "texto": response.text,
        "tokens_entrada": tokens_entrada,
        "tokens_salida": tokens_salida
    }