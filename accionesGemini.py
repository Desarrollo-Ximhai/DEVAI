import google.generativeai as genai

def conectarGemini(key):
    genai.configure(api_key=key)

def embedForCode(text):
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
    tokens = chat_model.count_tokens(payload_total_tokens)
    return response.text