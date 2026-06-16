

# =================================================================
# Busqueda en QDRANT
# =================================================================

def buscar_conocimiento_base_datos(conceptos_a_buscar: str) -> str:
    """
    Busca esquemas de tablas, descripciones lógicas, relaciones de llaves foráneas 
    y lógica de negocio en la base de datos del proyecto actual.
    
    Args:
        conceptos_a_buscar: Términos o conceptos de negocio a buscar (ej: "mantenimientos", "pagos de lotes").
    Returns:
        Un string con las tablas y descripciones más relevantes encontradas en formato Markdown.
    """
    # 1. Recuperamos las variables globales o de contexto de tu app
    # (Puedes pasarlas por el entorno, una clase, o variables globales de tu script)
    global client, proyecto, collection_name 
    
    print(f"🤖 El Agente solicitó buscar en Qdrant: '{conceptos_a_buscar}' para el proyecto '{proyecto}'")

    # 2. Generamos el embedding a 768 en caliente usando la tarea correcta para consultas
    query_embedding768 = embed_with_gemini(
        conceptos_a_buscar, 
        dimension=768, 
        task_type="retrieval_query"  # 💡 Clave: Usar query aquí
    )

    # 3. Invocamos tu función original (la que ya tiene el Reranker integrado)
    # Seteamos un k generoso (40) para que tu prefetch híbrido funcione con fuerza
    puntos_ganadores = search_in_qdrant(
        client=client,
        collection_name=collection_name,
        user_query=conceptos_a_buscar,
        query_embedding=query_embedding768,
        proyecto=proyecto,
        k=40
    )

    # 4. Formateamos el resultado de los puntos ganadores (los 5 del reranker) 
    # en un string limpio para que el agente lo pueda leer fácilmente.
    contexto_para_el_agente = []
    for p in puntos_ganadores:
        texto_chunk = p.payload.get("text", "")
        contexto_para_el_agente.append(texto_chunk)

    # Unimos todo en un solo bloque de texto
    return "\n\n---\n\n".join(contexto_para_el_agente)