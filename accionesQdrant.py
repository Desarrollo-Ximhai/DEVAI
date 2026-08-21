# accionesQdrant.py
import asyncio
from collections import Counter, defaultdict
from datetime import datetime
import httpx
import json
import math

import os
import re
import uuid
from typing import Any, Dict, List, Optional, Tuple



import fitz  
from qdrant_client import AsyncQdrantClient
from qdrant_client.http.models import Filter, FieldCondition, MatchValue, PointStruct, SparseVector, Prefetch, Fusion, FusionQuery
from qdrant_client.http import models 
from fastembed.sparse import SparseTextEmbedding
from langsmith import traceable


from accionesGemini import conectarGemini, generate_response, embed_with_gemini
from funciones import debug

sparse_model = SparseTextEmbedding(model_name="Qdrant/bm25")

def conectarQdrant( qdrant_url, qdrant_api_key):
    client = AsyncQdrantClient(
        url= qdrant_url,  
        api_key=qdrant_api_key 
    )
    return client
@traceable(run_type="chain", name="RERANKER")
async def rerank( query_usuario, candidatos, top_n):
        
    RERANK_KEY= os.environ.get('RERANK_KEY') 
    RERANK_URL= os.environ.get('RERANK_URL') 

    if not candidatos:
        return []

    url = f"{RERANK_URL}"
    headers = {
        "Authorization": f"Bearer {RERANK_KEY}", 
        "Content-Type": "application/json"
    }

    documents = []         # Esta lista de strings planos va para la API de Jina
    mapeo_documentos = []  # Esta lista guardará el candidato original de Qdrant correspondiente

    for c in candidatos:
        text = c.payload.get("text", "") if hasattr(c, "payload") else c.get("text", "")
        
        if "# TABLA:" in text:
            sub_chunks = text.split("# TABLA:")
            for sub in sub_chunks:
                texto_limpio = sub.strip()
                if texto_limpio:
                    tabla_formateada = f"# TABLA: {texto_limpio}"
                    
                    # Añadimos el texto plano a la lista de Jina
                    documents.append(tabla_formateada)
                    # Guardamos la referencia al objeto original de Qdrant en la misma posición
                    mapeo_documentos.append(c)
        else:
            if text.strip():
                documents.append(text.strip())
                mapeo_documentos.append(c)

    data = {
        "model": "jina-reranker-v3",
        "query": query_usuario,
        "top_n": top_n,
        "documents": documents,
        "return_documents": True,
    }

    # debug(documents)
    
    #response = requests.post(url, headers=headers, data=json.dumps(data)) response vieja sincrona

    # debug('response')
    # debug(response.json()) 
    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(url, headers=headers, data=json.dumps(data), timeout=15.0)
            
            if response.status_code == 200:
                res_data = response.json()
                chunks_finales = []
                
                for hit in res_data.get("results", []):
                    idx = hit.get("index")
                    # Validamos que el índice sea correcto y esté dentro del rango mapeado
                    if idx is not None and idx < len(mapeo_documentos):
                        candidato_original = mapeo_documentos[idx]
                        # Evitamos duplicados por si dos sub-tablas del mismo punto de Qdrant rankearon alto
                        if candidato_original and candidato_original not in chunks_finales:
                            chunks_finales.append(candidato_original)
                            
                return chunks_finales
            else:
                debug(f"⚠️ Reranker respondió con error {response.status_code}, usando fallback.")
                try:
                    debug(response.json())
                except Exception:
                    debug(response.text)
                return candidatos[:top_n]
                
        except Exception as e:
            debug(f"🚨 Error en la petición asíncrona de Rerank: {str(e)}")
            return candidatos[:top_n]

class Qdrant:
    def __init__(self, client, collection, proyecto):
        self.client = client
        self.collection = collection
        self.proyecto = proyecto
        
    async def borrar_por_chat_id(self, collection_name: str, chat_id: int):
        """
        Borra todos los puntos en una colección que pertenezcan a un chat_id específico.
        """
        filtro = Filter(
            must=[
                FieldCondition(
                    key="chat_id",
                    match=MatchValue(value=chat_id)
                )
            ]
        )
        
        resultado = await self.client.delete(
            collection_name=collection_name,
            points_selector=filtro
        )
        return resultado

    async def borrar_por_point_id(self, collection_name: str, point_id: str):
        """
        Borra un punto específico de la colección dado su ID único (UUID).
        """
        resultado = await self.client.delete(
            collection_name=collection_name,
            points_selector=[point_id]
        )
        return resultado

    @traceable(run_type="chain", name="Search_in_QDRANT")
    async def search_in_qdrant(self, user_query, query_embedding, k, top_n):
        global sparse_model
        filtros = []

        if self.proyecto:
            filtros.append(
                FieldCondition(
                    key="project",
                    match=MatchValue(value=self.proyecto)
                )
            )
        
        #sparse_emb = list(sparse_model.embed(user_query))[0]
        sparse_emb_list = await asyncio.to_thread(lambda: list(sparse_model.embed(user_query)))
        sparse_emb = sparse_emb_list[0]
        qdrant_sparse_vector = SparseVector(
            indices=sparse_emb.indices.tolist(),
            values=sparse_emb.values.tolist()
        )
        results = await self.client.query_points(
            collection_name=self.collection,
            prefetch=[
                # Sub-petición 1: Búsqueda Semántica (Densa)
                Prefetch(query=query_embedding, limit=k),
                # Sub-petición 2: Búsqueda por Palabras Clave (Dispersa)
                Prefetch(query=qdrant_sparse_vector, using="text-sparse", limit=k)
            ],
            # El motor fusiona ambos rankings automáticamente usando RRF
            query=FusionQuery(fusion=Fusion.RRF),
            limit=k,
            query_filter=Filter(
                must=filtros
            )
        )
        debug(f"Busqueda en qdrant con k:{k}" )

        #Reranking
        return await rerank(user_query, results.points, top_n) 

    async def guardarShot(self, embed_fn, user_query, collection_memory, cot, respuesta, proyecto="default"):
        
        points = []
        uuids = []
        
        emb = await embed_fn(user_query,768)
        if emb is None:
            pass
        # debug('emb')
        # debug(emb)
        unUUUID = str(uuid.uuid4())
        uuids.append(unUUUID)
        points.append(
            PointStruct(
                id=unUUUID,
                vector=emb,
                payload={
                    "user_query": user_query,
                    "chain_of_thought": cot,
                    "project": proyecto,
                    "timestamp": datetime.utcnow().isoformat()
                }
            )
        )
        
        
        if not points:
            debug("⚠️ No se generaron embeddings para guardar memoria.")
       
        await self.client.upsert(
            collection_name=collection_memory,
            points=points,
            wait=True
        )
        debug(f"✅ Memoria guardada ({len(points)} puntos) para proyecto '{proyecto}'.")
        return uuids

    async def save_to_qdrant(self, embed_fn, user_query, collection_memory, respuesta, chat_id, proyecto="default"):
        
       
        textos = [
            {"role": "user", "text": user_query.strip()},
            {"role": "assistant", "text": respuesta.strip()},
        ]
        # debug('textos')
        # debug(textos)
        points = []
        uuids = []
        for item in textos:
            emb = await embed_fn(item["text"],768)
            if emb is None:
                continue
            # debug('emb')
            # debug(emb)
            unUUUID = str(uuid.uuid4())
            uuids.append(unUUUID)
            points.append(
                PointStruct(
                    id=unUUUID,
                    vector=emb,
                    payload={
                        "text": item["text"],
                        "chat_id": chat_id,
                        "role": item["role"],
                        "project": proyecto,
                        "timestamp": datetime.utcnow().isoformat()
                    }
                )
            )
        
        
        if not points:
            debug("⚠️ No se generaron embeddings para guardar memoria.")
       
        await self.client.upsert(
            collection_name=collection_memory,
            points=points,
            wait=True
        )
        debug(f"✅ Memoria guardada ({len(points)} puntos) para proyecto '{proyecto}'.")
        return uuids


    async def getProjectMemory(self, embed_fn, user_query, collection_memory, chat_id, proyecto="default", limit=5):
        filtros = [
            FieldCondition(
                key="role",
                match=MatchValue(value="assistant")
            ),
            
        ]
        if proyecto:
            filtros.append(
                FieldCondition(
                    key="project",
                    match=MatchValue(value=proyecto)
                )
            )

        if chat_id:
            filtros.append(
                FieldCondition(
                    key="chat_id",
                    match=MatchValue(value=chat_id)
                )
            )
        query_emb = await embed_fn(user_query,768)
        res = await self.client.query_points(
            collection_name=collection_memory,
            query=query_emb,
            limit=limit,
            with_payload=True,
            with_vectors=False,
            query_filter=Filter(
                must=filtros
            )
        )
        puntos = res.points

        return puntos

    async def embebirBaseDatos(self, descripcion, archivo, proyecto):
        GOOGLE_API_KEY= os.environ.get('KEY-FREE') 
        conectarGemini(GOOGLE_API_KEY)


        archivos_procesados = []
        chunks_de_base_datos = [] 
        sql_string =  archivo["data"].decode("utf-8", errors="ignore")
        chunks_base = chunk_schema(
            sql_string, 
            archivo["filename"], 
            proyecto
        )
    
        prompt = f"""
        Analiza este esquema SQL completo y entiende la lógica de negocio del sistema.
        Devuelve un objeto JSON estrictamente formateado donde las llaves sean los nombres de las tablas 
        y los valores sean las descripciones semánticas en español (qué hace la tabla y reglas de negocio deducidas).

        DESCRIPCIÓN GENERAL:
        {descripcion}


        SQL COMPLETO:
        {sql_string}
        """

        
        respuesta = await generate_response(prompt, "models/gemini-3.6-flash", json=True)
        diccionario_descripciones = json.loads(respuesta["texto"])
        #return respuesta
        for chunk in chunks_base:
            if chunk["metadata"]["type"] == "table":
                tabla_nombre = chunk["metadata"]["table"]
                
                descripcion_ia = diccionario_descripciones.get(tabla_nombre, "")
                sql_original = chunk["text"]

                chunk["text"] = f"# TABLA: {tabla_nombre}\n**Descripción Lógica:** {descripcion_ia}\n\n## SQL ORIGINAL:\n{sql_original}"            
                chunk["metadata"]["description"] = descripcion_ia
                
            chunks_de_base_datos.append(chunk)
        
        chunks_with_embeddings = []
        for chunk in chunks_de_base_datos:
            embedding = await embed_with_gemini(chunk["text"], 768, "retrieval_document")
            chunk_with_embedding = {
                "text": chunk['text'],
                "metadata": chunk['metadata'],
                "embedding": embedding
            }
            chunks_with_embeddings.append(chunk_with_embedding);


        # ---------- EJECUCIÓN ASÍNCRONA CONCURRENTE ----------
        # async def fetch_embedding_for_chunk(chunk):
        #     embedding = await embed_with_gemini(chunk["text"], 768, "retrieval_document")
        #     return {
        #         "text": chunk['text'],
        #         "metadata": chunk['metadata'],
        #         "embedding": embedding
        #     }
        
        # # Lanza todas las peticiones a la API al mismo tiempo
        # tareas = [fetch_embedding_for_chunk(chunk) for chunk in chunks_de_base_datos]
        # chunks_with_embeddings = await asyncio.gather(*tareas)
        # -----------------------------------------------------
        
        collection_name = "DevAI-DB"
        try:
            await self.client.delete(
                collection_name=collection_name,
                points_selector=models.Filter(
                    should=[
                        models.FieldCondition(
                            key="project",
                            match=models.MatchValue(value=proyecto)
                        ),
                    
                    ]
                )
            )
        except Exception as e:
            debug(f"[⚠️ ADVERTENCIA] No se pudo borrar o no existían puntos previos: {e}")

        points = []
        
        for chunk_data in chunks_with_embeddings:
            if chunk_data['embedding'] is not None:
                
                payload = {
                    "text": chunk_data['text'],
                    "metadata": chunk_data['metadata'],
                    "project": proyecto 
                }

                # Procesar Vector Disperso (BM25)
                sparse_embeddings = list(sparse_model.embed([chunk_data['text']]))
                sparse_emb = sparse_embeddings[0]
                qdrant_sparse_vector = SparseVector(
                    indices=sparse_emb.indices.tolist(),
                    values=sparse_emb.values.tolist()
                )

                vector_hibrido = {
                    "": chunk_data['embedding'],           
                    "text-sparse": qdrant_sparse_vector   
                }

                punto_id = str(uuid.uuid4())

                points.append(
                    PointStruct(
                        id=punto_id,
                        vector=vector_hibrido,
                        payload=payload
                    )
                )

        try:
            await self.client.upsert(
                collection_name=collection_name,
                wait=True,
                points=points
            )
            debug(f"✅ Se han subido exitosamente {len(points)} chunks actualizados a la colección '{collection_name}'.")
        except Exception as e:
            debug(f"[ERROR CRÍTICO] al subir los chunks a Qdrant: {e}")


        return True

    async def embebirArchivos(self, descripcion, archivos, proyecto):
        debug('ARCHIVOS PROCESADOS')
        debug(len(archivos))
        
        GOOGLE_API_KEY = os.environ.get('KEY-FREE') 
        conectarGemini(GOOGLE_API_KEY)

        chunks_totales = []

        # 1. Iterar sobre todos los archivos recibidos y chunkearlos
        for archivo in archivos:
            # Para PDFs, NO decodificamos a utf-8. Pasamos los bytes puros.
            pdf_bytes = archivo["data"] 
            filename = archivo["filename"]
            
            # Llamamos a nuestro nuevo método de chunking para PDFs
            chunks_pdf = chunk_pdf_document(
                pdf_bytes=pdf_bytes, 
                relative_path=filename, 
                project=proyecto
            )
            chunks_totales.extend(chunks_pdf)

        debug(f"Total de chunks generados a partir de los PDFs: {len(chunks_totales)}")

        # 2. ---------- EJECUCIÓN ASÍNCRONA CONCURRENTE (ACTIVADA) ----------
        async def fetch_embedding_for_chunk(chunk):
            embedding = await embed_with_gemini(chunk["text"], 768, "retrieval_document")
            return {
                "text": chunk['text'],
                "metadata": chunk['metadata'],
                "embedding": embedding
            }
        chunks_with_embeddings = []
        lote_size = 80 # Nos mantenemos seguros por debajo del límite de 100
        
        for i in range(0, len(chunks_totales), lote_size):
            lote = chunks_totales[i:i + lote_size]
            
            # Disparamos las peticiones del lote actual
            tareas = [fetch_embedding_for_chunk(chunk) for chunk in lote]
            resultados_lote = await asyncio.gather(*tareas)
            chunks_with_embeddings.extend(resultados_lote)
            
            # Si aún quedan más chunks por procesar, esperamos 60 segundos
            if i + lote_size < len(chunks_totales):
                debug(f"Lote procesado. Esperando 60 segundos para evitar el Rate Limit (429)...")
                await asyncio.sleep(60)
        
        
        # Disparamos todas las peticiones a Gemini al mismo tiempo
        # tareas = [fetch_embedding_for_chunk(chunk) for chunk in chunks_totales]
        # chunks_with_embeddings = await asyncio.gather(*tareas)
        # -------------------------------------------------------------------

        collection_name = "DevAI-Analisis"
        
        # 3. Limpieza de documentos viejos del mismo proyecto en Qdrant
        try:
            await self.client.delete(
                collection_name=collection_name,
                points_selector=models.Filter(
                    should=[
                        models.FieldCondition(
                            key="project",
                            match=models.MatchValue(value=proyecto)
                        ),
                    ]
                )
            )
        except Exception as e:
            debug(f"[⚠️ ADVERTENCIA] No se pudo borrar o no existían puntos previos: {e}")

        # 4. Preparación de vectores híbridos y subida a Qdrant
        points = []
        for chunk_data in chunks_with_embeddings:
            if chunk_data['embedding'] is not None:
                
                payload = {
                    "text": chunk_data['text'],
                    "metadata": chunk_data['metadata'],
                    "project": proyecto 
                }

                # Procesar Vector Disperso (BM25)
                # (Asegúrate de que sparse_model esté inicializado en el alcance de tu clase)
                sparse_embeddings = list(sparse_model.embed([chunk_data['text']]))
                sparse_emb = sparse_embeddings[0]
                qdrant_sparse_vector = SparseVector(
                    indices=sparse_emb.indices.tolist(),
                    values=sparse_emb.values.tolist()
                )

                # Vector Híbrido (Denso + Disperso)
                vector_hibrido = {
                    "": chunk_data['embedding'],           
                    "text-sparse": qdrant_sparse_vector   
                }

                punto_id = str(uuid.uuid4())

                points.append(
                    PointStruct(
                        id=punto_id,
                        vector=vector_hibrido,
                        payload=payload
                    )
                )

        # 5. Upsert final a la base de datos
        try:
            await self.client.upsert(
                collection_name=collection_name,
                wait=True,
                points=points
            )
            debug(f"✅ Se han subido exitosamente {len(points)} chunks actualizados a la colección '{collection_name}'.")
        except Exception as e:
            debug(f"[ERROR CRÍTICO] al subir los chunks a Qdrant: {e}")

        return True

def chunk_schema(sql, relative_path, project):

    chunks = []

    # Limpieza: Comentarios de bloque y de línea
    sql_clean = re.sub(r"/\*.*?\*/", "", sql, flags=re.DOTALL)
    sql_clean = re.sub(r"--.*$", "", sql_clean, flags=re.MULTILINE)

    # ---------- 1) CREATE TABLE ----------
    table_pattern = re.compile(
        r"(CREATE\s+TABLE\s+(?:IF\s+NOT\s+EXISTS\s+)?(?:`?(\w+)`?\.)?`?(\w+)`?\s*\(.*?\)\s*ENGINE\s*=\s*\w+.*?;)",
        re.IGNORECASE | re.DOTALL
    )

    tables = {}

    for match in table_pattern.finditer(sql_clean):
        full_create = match.group(1).strip()
        db_name = match.group(2)
        table_name = match.group(3)

        tables[table_name] = {
            "db_name": db_name,
            "create": full_create,
            "columns": [],
            "relations": []
        }

        chunks.append({
            "text": full_create,
            "metadata": {
                "chunker": "chunk_schema",
                "type": "table",
                "project": project,
                "path": relative_path,
                "db_name": db_name,
                "table": table_name,
                "symbol": f"table:{table_name}"
            }
        })

        # Extraer columnas
        for line in full_create.splitlines():
            line = line.strip().rstrip(",")
            col_match = re.match(r"`(\w+)`\s+([A-Z]+(?:\([^)]+\))?)", line, re.IGNORECASE)
            if col_match:
                tables[table_name]["columns"].append({
                    "name": col_match.group(1),
                    "type": col_match.group(2)
                })

    # ---------- 2) FOREIGN KEYS EN ALTER TABLE ----------
    # Ya no están anidados dentro del bucle de las columnas
    alter_fk_pattern = re.compile(
        r"ALTER\s+TABLE\s+(?:`?(\w+)`?\.)?`?(\w+)`?\s+(.*?);",
        re.IGNORECASE | re.DOTALL
    )

    fk_in_alter_pattern = re.compile(
        r"ADD\s+CONSTRAINT\s+`?(\w+)`?\s+FOREIGN\s+KEY\s+\(`?(\w+)`?\)\s+REFERENCES\s+(?:`?(\w+)`?\.)?`?(\w+)`?\s+\(`?(\w+)`?\)",
        re.IGNORECASE
    )
    
    seen_relations = set()

    for alter_match in alter_fk_pattern.finditer(sql_clean):
        db_name = alter_match.group(1)
        table_name = alter_match.group(2)
        alter_body = alter_match.group(3)

        if table_name not in tables:
            tables[table_name] = {
                "db_name": db_name,
                "create": "",
                "columns": [],
                "relations": []
            }

        for fk_match in fk_in_alter_pattern.finditer(alter_body):
            constraint_name = fk_match.group(1)
            column = fk_match.group(2)
            ref_db = fk_match.group(3)
            ref_table = fk_match.group(4)
            ref_column = fk_match.group(5)

            relation_key = (
                table_name,
                column,
                ref_table,
                ref_column,
            )

            if relation_key in seen_relations:
                continue

            seen_relations.add(relation_key)

            tables[table_name]["relations"].append({
                "constraint": constraint_name,
                "column": column,
                "ref_db": ref_db,
                "ref_table": ref_table,
                "ref_column": ref_column
            })

    # ---------- 3) Chunks de relaciones ----------
    # Indentación corregida para procesar todas las tablas
    for table_name, info in tables.items():
        if not info["relations"]:
            continue

        relaciones_unicas = {}

        for rel in info["relations"]:
            key = (
                table_name,
                rel.get("column"),
                rel.get("ref_table"),
                rel.get("ref_column")
            )
            relaciones_unicas[key] = rel

        lines = [f"Relaciones de la tabla `{table_name}`:"]

        for rel in relaciones_unicas.values():
            constraint = rel.get("constraint", "")
            prefix = f"- Constraint `{constraint}`: " if constraint else "- "
            lines.append(
                f"{prefix}`{table_name}`.`{rel['column']}` referencia "
                f"`{rel['ref_table']}`.`{rel['ref_column']}`"
            )

        chunks.append({
            "text": "\n".join(lines),
            "metadata": {
                "chunker": "chunk_schema",
                "type": "relationships",
                "project": project,
                "path": relative_path,
                "db_name": info.get("db_name"),
                "table": table_name,
                "symbol": f"relationships:{table_name}"
            }
        })

    # ---------- 4) CREATE VIEW ----------
    view_pattern = re.compile(
        r"(CREATE\s+(?:OR\s+REPLACE\s+)?VIEW\s+(?:`?(\w+)`?\.)?`?(\w+)`?\s+AS\s+.*?;)",
        re.IGNORECASE | re.DOTALL
    )

    for match in view_pattern.finditer(sql_clean):
        full_view = match.group(1).strip()
        db_name = match.group(2)
        view_name = match.group(3)

        chunks.append({
            "text": full_view,
            "metadata": {
                "chunker": "chunk_schema",
                "type": "view",
                "project": project,
                "path": relative_path,
                "db_name": db_name,
                "view": view_name,
                "symbol": f"view:{view_name}"
            }
        })

    # ---------- 5) Resumen general ----------
    if tables:
        lines = [f"Resumen del schema del proyecto `{project}`:"]
        for table_name, info in tables.items():
            columnas = ", ".join([c["name"] for c in info["columns"][:20]])
            extra = "..." if len(info["columns"]) > 20 else ""
            lines.append(f"- `{table_name}`: columnas {columnas}{extra}")

        chunks.append({
            "text": "\n".join(lines),
            "metadata": {
                "chunker": "chunk_schema",
                "type": "schema_summary",
                "project": project,
                "path": relative_path,
                "symbol": f"schema_summary:{project}"
            }
        })

    return chunks




# ============================================================
# CHUNKING PDF
# ============================================================

def chunk_pdf_document(
    pdf_bytes: bytes,
    relative_path: str,
    project: str,
    chunk_size: int = 1200,
    overlap: int = 200,
    min_chunk_size: int = 200,
) -> List[Dict[str, Any]]:
    """
    Chunker de PDFs orientado a RAG.

    Características:
    - Detecta tamaño de fuente predominante del body.
    - Detecta headings usando múltiples señales.
    - Mantiene secciones aunque crucen páginas.
    - Elimina headers / footers repetidos.
    - Evita cortar chunks demasiado temprano.
    - Fusiona chunks pequeños en lugar de descartarlos.
    - Conserva page_start / page_end.
    - Genera `embedding_text` enriquecido con documento + sección.
    - Genera un índice estructural separado.

    NOTA:
    `chunk_size`, `overlap` y `min_chunk_size` están en caracteres.
    """

    if chunk_size <= 0:
        raise ValueError("chunk_size debe ser > 0")

    if overlap < 0:
        raise ValueError("overlap debe ser >= 0")

    if overlap >= chunk_size:
        raise ValueError("overlap debe ser menor que chunk_size")

    if min_chunk_size < 0:
        raise ValueError("min_chunk_size debe ser >= 0")

    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    except Exception as e:
        print(f"Error abriendo PDF: {e}")
        return []

    try:
        if len(doc) == 0:
            return []

        # ----------------------------------------------------
        # 1. EXTRAER LÍNEAS ESTRUCTURADAS
        # ----------------------------------------------------

        pages = _extract_pdf_lines(doc)

        if not any(page["lines"] for page in pages):
            return []

        # ----------------------------------------------------
        # 2. DETECTAR HEADERS / FOOTERS REPETIDOS
        # ----------------------------------------------------

        repeated_noise = _detect_repeated_headers_footers(pages)

        for page in pages:
            page["lines"] = [
                line
                for line in page["lines"]
                if _normalize_repeated_line(line["text"]) not in repeated_noise
            ]

        # ----------------------------------------------------
        # 3. DETECTAR TAMAÑO DE BODY
        # ----------------------------------------------------

        body_font_size = _detect_body_font_size(pages)

        # ----------------------------------------------------
        # 4. DETECTAR TÍTULO DEL DOCUMENTO
        # ----------------------------------------------------

        fallback_title = os.path.basename(relative_path) or relative_path

        document_title = _detect_document_title(
            pages=pages,
            body_font_size=body_font_size,
            fallback=fallback_title,
        )

        # ----------------------------------------------------
        # 5. CONSTRUIR SECCIONES
        # ----------------------------------------------------

        sections = _build_sections(
            pages=pages,
            body_font_size=body_font_size,
            document_title=document_title,
        )

        # ----------------------------------------------------
        # 6. CHUNKEAR POR SECCIÓN
        # ----------------------------------------------------

        chunks: List[Dict[str, Any]] = []
        global_chunk_index = 0

        for section_index, section in enumerate(sections):
            section_title = section["title"]
            section_parts = section["parts"]

            if not section_parts:
                continue

            section_text, page_ranges = _join_parts_with_page_map(
                section_parts
            )

            if not section_text.strip():
                continue

            split_chunks = split_text_semantic_v3(
                text=section_text,
                chunk_size=chunk_size,
                overlap=overlap,
                min_chunk_size=min_chunk_size,
            )

            for section_chunk_index, split_chunk in enumerate(split_chunks):
                chunk_text = split_chunk["text"]
                start = split_chunk["start"]
                end = split_chunk["end"]

                page_start, page_end = _find_pages_for_char_range(
                    page_ranges=page_ranges,
                    chunk_start=start,
                    chunk_end=end,
                )

                # Texto que efectivamente se manda al modelo de embeddings.
                embedding_text = _build_embedding_text(
                    document_title=document_title,
                    section_title=section_title,
                    chunk_text=chunk_text,
                )

                chunks.append({
                    "text": chunk_text,

                    # Recomiendo crear el embedding usando este campo.
                    "embedding_text": embedding_text,

                    "metadata": {
                        "chunker": "semantic_pdf_v3",
                        "type": "content_chunk",

                        "project": project,
                        "path": relative_path,

                        "document_title": document_title,
                        "section": section_title,

                        "page_start": page_start,
                        "page_end": page_end,

                        # Para compatibilidad con pipelines que esperan page.
                        "page": page_start,

                        "chunk_index": global_chunk_index,
                        "section_index": section_index,
                        "section_chunk_index": section_chunk_index,

                        "total_pages": len(doc),

                        "char_count": len(chunk_text),
                        "embedding_char_count": len(embedding_text),
                    }
                })

                global_chunk_index += 1

        # ----------------------------------------------------
        # 7. CREAR ÍNDICE DEL DOCUMENTO
        # ----------------------------------------------------

        if sections:
            index_lines = []

            for section in sections:
                if not section["parts"]:
                    continue

                pages_in_section = sorted({
                    part["page"]
                    for part in section["parts"]
                })

                if not pages_in_section:
                    continue

                page_start = min(pages_in_section)
                page_end = max(pages_in_section)

                if page_start == page_end:
                    page_label = f"Pág. {page_start}"
                else:
                    page_label = f"Págs. {page_start}-{page_end}"

                index_lines.append(
                    f"- {section['title']} ({page_label})"
                )

            if index_lines:
                index_text = (
                    f"Documento: {document_title}\n\n"
                    "Estructura del documento:\n"
                    + "\n".join(index_lines)
                )

                chunks.append({
                    "text": index_text,
                    "embedding_text": index_text,
                    "metadata": {
                        "chunker": "semantic_pdf_v3",
                        "type": "document_index",

                        "project": project,
                        "path": relative_path,

                        "document_title": document_title,

                        "total_pages": len(doc),

                        "symbol": f"doc_index:{project}:{relative_path}",
                    }
                })

        return chunks

    finally:
        doc.close()


# ============================================================
# EXTRACTION
# ============================================================

def _extract_pdf_lines(doc) -> List[Dict[str, Any]]:
    """
    Extrae líneas conservando:
    - página
    - bbox
    - tamaño de fuente
    - bold
    - fuente
    - posición relativa
    """

    pages = []

    for page_num in range(len(doc)):
        page = doc[page_num]

        page_height = float(page.rect.height)
        page_width = float(page.rect.width)

        try:
            page_dict = page.get_text(
                "dict",
                sort=True,
            )
        except TypeError:
            # Compatibilidad con versiones viejas de PyMuPDF.
            page_dict = page.get_text("dict")

        lines_data = []

        for block in page_dict.get("blocks", []):
            if "lines" not in block:
                continue

            for line in block["lines"]:
                spans = line.get("spans", [])

                if not spans:
                    continue

                text = "".join(
                    span.get("text", "")
                    for span in spans
                ).strip()

                if not text:
                    continue

                bbox = line.get(
                    "bbox",
                    (
                        min(span["bbox"][0] for span in spans),
                        min(span["bbox"][1] for span in spans),
                        max(span["bbox"][2] for span in spans),
                        max(span["bbox"][3] for span in spans),
                    )
                )

                char_count = sum(
                    max(len(span.get("text", "").strip()), 1)
                    for span in spans
                )

                weighted_size = sum(
                    float(span.get("size", 0)) *
                    max(len(span.get("text", "").strip()), 1)
                    for span in spans
                )

                avg_size = (
                    weighted_size / char_count
                    if char_count
                    else 0
                )

                bold_chars = 0

                for span in spans:
                    span_text = span.get("text", "")

                    flags = int(span.get("flags", 0))
                    font_name = span.get("font", "").lower()

                    # PyMuPDF TEXT_FONT_BOLD = 16.
                    is_bold = bool(flags & 16)

                    # Fallback porque algunos PDFs describen mal flags.
                    if not is_bold:
                        is_bold = any(
                            marker in font_name
                            for marker in (
                                "bold",
                                "black",
                                "heavy",
                                "demi",
                                "semibold",
                            )
                        )

                    if is_bold:
                        bold_chars += max(len(span_text.strip()), 1)

                bold_ratio = (
                    bold_chars / char_count
                    if char_count
                    else 0
                )

                lines_data.append({
                    "text": _clean_line_text(text),

                    "page": page_num + 1,

                    "bbox": bbox,

                    "x0": float(bbox[0]),
                    "y0": float(bbox[1]),
                    "x1": float(bbox[2]),
                    "y1": float(bbox[3]),

                    "page_width": page_width,
                    "page_height": page_height,

                    "font_size": avg_size,
                    "bold_ratio": bold_ratio,

                    "spans": spans,
                })

        pages.append({
            "page": page_num + 1,
            "width": page_width,
            "height": page_height,
            "lines": lines_data,
        })

    return pages


# ============================================================
# BODY FONT DETECTION
# ============================================================

def _detect_body_font_size(
    pages: List[Dict[str, Any]]
) -> float:
    """
    Detecta aproximadamente el tamaño de fuente predominante.

    Se pondera por cantidad de caracteres para que un heading grande
    no tenga el mismo peso que un párrafo completo.
    """

    weights = defaultdict(int)

    for page in pages:
        for line in page["lines"]:
            text = line["text"].strip()

            if len(text) < 3:
                continue

            size = round(line["font_size"] * 2) / 2

            weights[size] += len(text)

    if not weights:
        return 10.0

    return max(
        weights.items(),
        key=lambda item: item[1]
    )[0]


# ============================================================
# HEADER / FOOTER REMOVAL
# ============================================================

def _detect_repeated_headers_footers(
    pages: List[Dict[str, Any]],
    top_ratio: float = 0.10,
    bottom_ratio: float = 0.10,
    min_page_ratio: float = 0.60,
) -> set:
    """
    Detecta líneas repetidas en la parte superior/inferior del PDF.

    Ejemplo:
        "ACME CONFIDENTIAL"
        "Manual de usuario"
        "Página 14 de 80"

    Los números se normalizan para detectar:
        Página 1
        Página 2
        Página 3
    como el mismo patrón.
    """

    total_pages = len(pages)

    if total_pages < 3:
        return set()

    occurrences = defaultdict(set)

    for page in pages:
        page_number = page["page"]
        height = page["height"]

        for line in page["lines"]:
            y0 = line["y0"]
            y1 = line["y1"]

            in_header = y0 <= height * top_ratio
            in_footer = y1 >= height * (1 - bottom_ratio)

            if not (in_header or in_footer):
                continue

            normalized = _normalize_repeated_line(
                line["text"]
            )

            if not normalized:
                continue

            # Evitamos considerar párrafos largos como header/footer.
            if len(normalized) > 160:
                continue

            occurrences[normalized].add(page_number)

    required_pages = max(
        3,
        math.ceil(total_pages * min_page_ratio)
    )

    repeated = {
        text
        for text, page_numbers in occurrences.items()
        if len(page_numbers) >= required_pages
    }

    return repeated


def _normalize_repeated_line(text: str) -> str:
    text = text.lower().strip()

    # Página 12 -> página #
    text = re.sub(r"\d+", "#", text)

    text = re.sub(r"\s+", " ", text)

    return text.strip()


# ============================================================
# HEADING DETECTION
# ============================================================

def _is_heading(
    line: Dict[str, Any],
    body_font_size: float,
) -> bool:
    text = line["text"].strip()

    if not text:
        return False

    # Los títulos suelen ser relativamente cortos.
    if len(text) > 180:
        return False

    # Evita números solos.
    if re.fullmatch(r"[\d\s./-]+", text):
        return False

    font_size = line["font_size"]
    bold_ratio = line["bold_ratio"]

    ratio = (
        font_size / body_font_size
        if body_font_size > 0
        else 1
    )

    score = 0

    # --------------------------
    # Tamaño de fuente
    # --------------------------

    if ratio >= 1.50:
        score += 4

    elif ratio >= 1.30:
        score += 3

    elif ratio >= 1.15:
        score += 2

    # --------------------------
    # Bold
    # --------------------------

    if bold_ratio >= 0.70:
        score += 2

    elif bold_ratio >= 0.35:
        score += 1

    # --------------------------
    # Longitud
    # --------------------------

    if len(text) <= 80:
        score += 1

    # --------------------------
    # Numeración de headings
    # --------------------------

    # 1 Introducción
    # 2.3 Arquitectura
    # 4.2.1 Autenticación
    if re.match(
        r"^\s*(?:\d+(?:\.\d+){0,5}|[IVXLC]+)[.)]?\s+\S+",
        text,
        re.IGNORECASE
    ):
        score += 2

    # --------------------------
    # Mayúsculas
    # --------------------------

    letters = [
        char
        for char in text
        if char.isalpha()
    ]

    if letters:
        uppercase_ratio = (
            sum(char.isupper() for char in letters)
            / len(letters)
        )

        if uppercase_ratio >= 0.85 and len(text) <= 100:
            score += 1

    # --------------------------
    # Señales negativas
    # --------------------------

    # Un título normalmente no termina como una frase completa.
    if text.endswith((".", ",", ";")):
        score -= 1

    # Texto de tamaño body sin bold necesita señales muy claras.
    if ratio < 1.10 and bold_ratio < 0.35:
        score -= 2

    return score >= 3


# ============================================================
# DOCUMENT TITLE
# ============================================================

def _detect_document_title(
    pages: List[Dict[str, Any]],
    body_font_size: float,
    fallback: str,
) -> str:
    """
    Intenta obtener el título a partir de la primera página.
    """

    if not pages:
        return fallback

    candidates = []

    # Normalmente basta con la primera página.
    first_page = pages[0]

    for index, line in enumerate(first_page["lines"]):
        text = line["text"].strip()

        if not text:
            continue

        if len(text) > 200:
            continue

        ratio = (
            line["font_size"] / body_font_size
            if body_font_size
            else 1
        )

        score = 0

        score += ratio * 3

        if line["bold_ratio"] >= 0.5:
            score += 1

        # Preferir cosas que aparecen arriba.
        vertical_position = (
            line["y0"] / first_page["height"]
            if first_page["height"]
            else 1
        )

        if vertical_position < 0.35:
            score += 2

        if len(text) <= 120:
            score += 1

        # Preferencia muy ligera por las primeras líneas.
        score -= index * 0.05

        candidates.append(
            (score, line["font_size"], text)
        )

    if not candidates:
        return fallback

    candidates.sort(reverse=True)

    best_score, best_size, best_text = candidates[0]

    # No aceptar como título cualquier línea body.
    if best_size < body_font_size * 1.10:
        return fallback

    return best_text


# ============================================================
# SECTION BUILDING
# ============================================================

def _build_sections(
    pages: List[Dict[str, Any]],
    body_font_size: float,
    document_title: str,
) -> List[Dict[str, Any]]:
    """
    Construye secciones que pueden abarcar múltiples páginas.
    """

    sections = []

    current_section = {
        "title": "General",
        "parts": [],
    }

    for page in pages:
        for line in page["lines"]:
            text = line["text"].strip()

            if not text:
                continue

            heading = _is_heading(
                line=line,
                body_font_size=body_font_size,
            )

            if heading:
                heading_text = _clean_heading(text)

                # Evitar que el título del documento se convierta
                # automáticamente en una sección vacía.
                if (
                    _normalize_for_comparison(heading_text)
                    == _normalize_for_comparison(document_title)
                    and not current_section["parts"]
                ):
                    continue

                # Guardar sección anterior si tiene contenido.
                if current_section["parts"]:
                    sections.append(current_section)

                current_section = {
                    "title": heading_text,
                    "parts": [],
                }

                continue

            current_section["parts"].append({
                "text": text,
                "page": line["page"],
            })

    if current_section["parts"]:
        sections.append(current_section)

    # Si por algún motivo todas quedaron vacías.
    if not sections:
        all_parts = []

        for page in pages:
            for line in page["lines"]:
                text = line["text"].strip()

                if text:
                    all_parts.append({
                        "text": text,
                        "page": line["page"],
                    })

        if all_parts:
            sections.append({
                "title": "General",
                "parts": all_parts,
            })

    return sections


# ============================================================
# TEXT + PAGE MAPPING
# ============================================================

def _join_parts_with_page_map(
    parts: List[Dict[str, Any]]
) -> Tuple[str, List[Dict[str, int]]]:
    """
    Une las líneas de una sección manteniendo un mapa:
        posición caracteres -> página
    """

    pieces = []
    ranges = []

    cursor = 0
    previous_page = None

    for part in parts:
        text = part["text"].strip()
        page = part["page"]

        if not text:
            continue

        if pieces:
            # Mantener doble salto cuando cambia de página.
            separator = (
                "\n\n"
                if previous_page != page
                else "\n"
            )

            pieces.append(separator)
            cursor += len(separator)

        start = cursor

        pieces.append(text)

        cursor += len(text)

        end = cursor

        ranges.append({
            "start": start,
            "end": end,
            "page": page,
        })

        previous_page = page

    return "".join(pieces), ranges


def _find_pages_for_char_range(
    page_ranges: List[Dict[str, int]],
    chunk_start: int,
    chunk_end: int,
) -> Tuple[Optional[int], Optional[int]]:
    """
    Obtiene las páginas tocadas por un rango de caracteres.
    """

    touched_pages = []

    for item in page_ranges:
        overlaps = (
            item["end"] > chunk_start
            and item["start"] < chunk_end
        )

        if overlaps:
            touched_pages.append(item["page"])

    if not touched_pages:
        return None, None

    return min(touched_pages), max(touched_pages)


# ============================================================
# SPLITTER
# ============================================================

def split_text_semantic_v3(
    text: str,
    chunk_size: int = 1200,
    overlap: int = 200,
    min_chunk_size: int = 200,
) -> List[Dict[str, Any]]:
    """
    Splitter recursivo-ish basado en límites naturales.

    Devuelve también offsets:
        {
            "text": "...",
            "start": 100,
            "end": 1260
        }

    Los offsets permiten determinar las páginas de cada chunk.
    """

    if not text:
        return []

    if overlap >= chunk_size:
        raise ValueError(
            "overlap debe ser menor que chunk_size"
        )

    separators = [
        "\n\n",
        "\n",
        ". ",
        "? ",
        "! ",
        "; ",
        ": ",
        ", ",
        " ",
    ]

    text_length = len(text)

    chunks = []

    start = 0

    # Evitamos aceptar cortes extremadamente pequeños.
    min_cut_ratio = 0.65

    while start < text_length:
        target_end = min(
            start + chunk_size,
            text_length
        )

        # Último chunk.
        if target_end >= text_length:
            raw_chunk = text[start:text_length]

            clean_text, left_trim, right_trim = _trim_with_offsets(
                raw_chunk
            )

            if clean_text:
                chunks.append({
                    "text": clean_text,
                    "start": start + left_trim,
                    "end": text_length - right_trim,
                })

            break

        min_end = int(
            start + chunk_size * min_cut_ratio
        )

        min_end = min(
            min_end,
            target_end
        )

        cut_candidates = []

        # Buscar todos los tipos de separador.
        for priority, separator in enumerate(separators):
            position = text.rfind(
                separator,
                min_end,
                target_end
            )

            if position == -1:
                continue

            cut_position = (
                position + len(separator)
            )

            cut_candidates.append({
                "position": cut_position,
                "priority": priority,
            })

        if cut_candidates:
            # Primero preferimos estar cerca del target.
            # En empate, preferimos separadores "más semánticos".
            cut_candidates.sort(
                key=lambda item: (
                    target_end - item["position"],
                    item["priority"],
                )
            )

            end = cut_candidates[0]["position"]

        else:
            end = target_end

        # Garantía absoluta de progreso.
        if end <= start:
            end = min(
                start + chunk_size,
                text_length
            )

        raw_chunk = text[start:end]

        clean_text, left_trim, right_trim = _trim_with_offsets(
            raw_chunk
        )

        if clean_text:
            chunks.append({
                "text": clean_text,
                "start": start + left_trim,
                "end": end - right_trim,
            })

        previous_start = start

        proposed_start = end - overlap

        # Ajustar overlap para evitar comenzar en medio de palabra.
        proposed_start = _move_to_word_boundary(
            text=text,
            position=proposed_start,
            max_lookahead=40,
        )

        start = max(
            proposed_start,
            previous_start + 1
        )

    # --------------------------------------------------------
    # Fusionar tail pequeño
    # --------------------------------------------------------

    if (
        len(chunks) >= 2
        and len(chunks[-1]["text"]) < min_chunk_size
    ):
        previous = chunks[-2]
        last = chunks[-1]

        # Evitamos duplicar el overlap al fusionar.
        combined_start = previous["start"]
        combined_end = last["end"]

        combined_text = text[
            combined_start:combined_end
        ].strip()

        chunks[-2] = {
            "text": combined_text,
            "start": combined_start,
            "end": combined_end,
        }

        chunks.pop()

    return chunks


# ============================================================
# EMBEDDING TEXT
# ============================================================

def _build_embedding_text(
    document_title: str,
    section_title: str,
    chunk_text: str,
) -> str:
    """
    Enriquece el embedding con contexto estructural.

    El texto que se muestra al usuario puede seguir siendo chunk_text.
    """

    pieces = []

    if document_title:
        pieces.append(
            f"Documento: {document_title}"
        )

    if (
        section_title
        and section_title != "General"
    ):
        pieces.append(
            f"Sección: {section_title}"
        )

    pieces.append(chunk_text)

    return "\n\n".join(pieces)


# ============================================================
# UTILS
# ============================================================

def _clean_line_text(text: str) -> str:
    """
    Limpia espacios horizontales SIN destruir estructura global.
    """

    text = text.replace("\u00a0", " ")

    text = re.sub(
        r"[ \t]+",
        " ",
        text
    )

    return text.strip()


def _clean_heading(text: str) -> str:
    text = _clean_line_text(text)

    # Evitar headings con puntos finales accidentales.
    text = text.rstrip()

    return text


def _normalize_for_comparison(text: str) -> str:
    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^\w\s]", "", text)

    return text.strip()


def _trim_with_offsets(
    text: str
) -> Tuple[str, int, int]:
    """
    strip(), pero devuelve cuánto eliminó de cada lado.
    """

    if not text:
        return "", 0, 0

    left = len(text) - len(text.lstrip())
    right = len(text) - len(text.rstrip())

    cleaned = text.strip()

    return cleaned, left, right


def _move_to_word_boundary(
    text: str,
    position: int,
    max_lookahead: int = 40,
) -> int:
    """
    Si el overlap cae en mitad de una palabra, avanza hasta
    el siguiente whitespace.

    Así evitamos:
        "...arquitec"
        "tectura del sistema..."

    cuando sea razonablemente posible.
    """

    if position <= 0:
        return 0

    if position >= len(text):
        return len(text)

    # Ya estamos en un límite.
    if text[position].isspace():
        return position

    upper = min(
        position + max_lookahead,
        len(text)
    )

    for i in range(position, upper):
        if text[i].isspace():
            return i + 1

    return position
