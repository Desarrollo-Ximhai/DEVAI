import google.generativeai as genai

def conectarGemini(key):
    print("--Conectando Gemini")
    genai.configure(api_key=key)

def embedForCode(text):
    print('--Embebiendo para codigo')
    res = genai.embed_content(
        model='models/gemini-embedding-001',
        content=text,
        task_type="retrieval_document",
        output_dimensionality=3072
    )
    return res["embedding"] if "embedding" in res else None

def embed_with_gemini(text, dimension=3072):
    res = genai.embed_content(
        model='models/gemini-embedding-001',
        content=text,
        task_type="retrieval_document",
        output_dimensionality=dimension
    )
    return res["embedding"] if "embedding" in res else None

def generate_response(prompt, model_name="models/gemini-3.1-flash-lite", archivos: list = None):
    print('modelo en generate')
    print(model_name)
    chat_model = genai.GenerativeModel(model_name)
    contenidos_payload = [prompt]
    if archivos:
        for arc in archivos:
            contenidos_payload.append({
                "mime_type": arc["mime_type"],
                "data": arc["data"]
            })
    convo = chat_model.start_chat()
    response = convo.send_message(contenidos_payload)
    payload_total_tokens = contenidos_payload + [response.text]
    
    # ─── EXTRACCIÓN DE TOKENS ──────────────────────────────────────
    # Accedemos a los metadatos de uso que Gemini ya calculó para nosotros
    uso_tokens = response.usage_metadata
    
    tokens_entrada = uso_tokens.prompt_token_count
    tokens_salida = uso_tokens.candidates_token_count
    tokens_totales = uso_tokens.total_token_count
    
    print(f"--- Métrica de Tokens ---")
    print(f"---- Response completo ---")
    print(response)
    print(f"Tokens de Entrada (Prompt + Archivos): {tokens_entrada}")
    print(f"Tokens de Salida (Respuesta): {tokens_salida}")
    print(f"Tokens Totales: {tokens_totales}")
    print(f"─────────────────────────")

    return response.text