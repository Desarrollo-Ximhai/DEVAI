import google.generativeai as genai
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

def embed_with_gemini(text, dimension=3072):
    res = genai.embed_content(
        model='models/gemini-embedding-001',
        content=text,
        task_type="retrieval_query",
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

def generate_response(prompt, model_name="models/gemini-3.1-flash-lite", archivos: list = None):
    print('modelo en generate:', model_name)
    
    chat_model = genai.GenerativeModel(model_name)
    contenidos_payload = [prompt]
    
    if archivos:
        for arc in archivos:
            contenidos_payload.append({
                "mime_type": arc["mime_type"],
                "data": arc["data"]
            })
    response = chat_model.generate_content(contenidos_payload)
    
    uso_tokens = response.usage_metadata
    tokens_entrada = uso_tokens.prompt_token_count
    tokens_salida = uso_tokens.candidates_token_count
    
    print(f"--- Info de la petición ---")
    print(f"Tokens Entrada: {tokens_entrada} | Tokens Salida: {tokens_salida}")
    print(f"───────────────────────────")
    #print('Respuesta:')
    #print(response.text)
    # # Opción A: Si solo necesitas el texto como antes, dejas esto:
    # return {"response": response.text, tokens_entrada}
    # return response.text
    
    return {
        "texto": response.text,
        "tokens_entrada": tokens_entrada,
        "tokens_salida": tokens_salida
    }