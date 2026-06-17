import google.generativeai as genai
import json 

from funciones import debug

def conectarGemini(key):
    print("--Conectando Gemini")
    genai.configure(api_key=key)

def embedForCode(text):
    print('--Embebiendo para codigo')
    res = genai.embed_content(
        model='models/gemini-embedding-001',
        content=text,
        #task_type="retrieval_document",
        task_type="retrieval_query",
        output_dimensionality=3072
    )
    return res["embedding"] if "embedding" in res else None

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

# def generate_response(prompt, model_name="models/gemini-3.1-flash-lite", archivos: list = None):
#     print('modelo en generate')
#     print(model_name)
#     chat_model = genai.GenerativeModel(model_name)
#     contenidos_payload = [prompt]
#     if archivos:
#         for arc in archivos:
#             contenidos_payload.append({
#                 "mime_type": arc["mime_type"],
#                 "data": arc["data"]
#             })
#     convo = chat_model.start_chat()
#     response = convo.send_message(contenidos_payload)
#     payload_total_tokens = contenidos_payload + [response.text]
#     tokens = chat_model.count_tokens(payload_total_tokens)
#     return response.text

import google.api_core.exceptions as google_exceptions
from fastapi import HTTPException  # ← Crucial importar esto

def generate_response(prompt, model_name="models/gemini-3.1-flash-lite", archivos: list = None, configuracion = None, tools: list = None, system_instruction=None, history:list = None):
    print('modelo en generate:', model_name)
    
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

    # 🛑 Función interna para lanzar los errores HTTP correctos hacia PHP
    def lanzar_error_api(e):
        print(f"❌ [ERROR EN GEMINI API]: {str(e)}")
        
        # Detectamos si es un error de cuota superada
        es_error_cuota = isinstance(e, google_exceptions.ResourceExhausted) or \
                         "quota" in str(e).lower() or \
                         "429" in str(e) or \
                         "resourceexhausted" in str(e).lower()
                         
        if es_error_cuota:
            # Lanzamos un HTTP 429 real con el mensaje personalizado
            raise HTTPException(
                status_code=429,
                detail="⚠️ [Error de Cuota] Has superado el límite de peticiones permitidas por minuto en Gemini. Por favor, espera unos segundos antes de enviar otro mensaje."
            )
        else:
            # Para cualquier otro error de la IA (ej. prompt bloqueado, error de modelo, etc.)
            raise HTTPException(
                status_code=500,
                detail=f"💥 [Error de Gemini]: No se pudo completar la solicitud de IA. Detalle: {str(e)}"
            )

    # 🤖 MODO AGENTE
    if tools:
        print("🤖 [INFO] Modo Agente activado. Orquestando llamadas automáticas...")
        chat_model = genai.GenerativeModel(model_name=model_name, tools=tools, system_instruction=system_instruction)
        chat = chat_model.start_chat(history=history, enable_automatic_function_calling=True)
        
        try:
            response = chat.send_message(contenidos_payload, generation_config=gen_config if gen_config else None)
        except Exception as e:
            lanzar_error_api(e) # Esto corta la ejecución y manda el 429 o 500
        
        # Trazas de pasos intermedios (Solo se ejecutan si todo salió bien)
        print("\n🗺️  [TRAZA DE PASOS Y RAZONAMIENTO DEL AGENTE]")
        print("──────────────────────────────────────────────────")
        for mensaje in chat.history:
            for part in mensaje.parts:
                part_dict = type(part).to_dict(part) if hasattr(type(part), 'to_dict') else {}
                if 'function_call' in part_dict:
                    print(f"🧠 [LLM PENSÓ]: Necesito extraer datos del sistema. Llamando a: '{part.function_call.name}'")
                elif 'function_response' in part_dict:
                    print(f"⚙️  [PYTHON EJECUTÓ]: '{part_dict['function_response'].get('name')}' -> Datos devueltos.")
                elif 'text' in part_dict:
                    rol = "USUARIO" if mensaje.role == "user" else "GEMINI"
                    print(f"💬 [{rol}]: {part.text.strip()}\n")
        print("──────────────────────────────────────────────────\n")

    # 📝 MODO NORMAL
    else:
        try:
            response = chat_model.generate_content(
                contenidos_payload,
                generation_config=gen_config if gen_config else None
            )
        except Exception as e:
            lanzar_error_api(e)
    
    # Extraemos metadata si todo fue exitoso
    uso_tokens = response.usage_metadata
    tokens_entrada = uso_tokens.prompt_token_count
    tokens_salida = uso_tokens.candidates_token_count
    
    return {
        "texto": response.text,
        "tokens_entrada": tokens_entrada,
        "tokens_salida": tokens_salida,
        "status": "success"
    }