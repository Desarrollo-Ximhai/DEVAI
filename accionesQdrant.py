# accionesQdrant.py
import asyncio
from datetime import datetime
import httpx
import json
import os
import re
import uuid

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
        
        # Disparamos todas las peticiones a Gemini al mismo tiempo
        tareas = [fetch_embedding_for_chunk(chunk) for chunk in chunks_totales]
        chunks_with_embeddings = await asyncio.gather(*tareas)
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


def chunk_pdf_document(pdf_bytes, relative_path, project, chunk_size=800, overlap=150):
    """
    Procesa un archivo PDF en bytes, extrae su estructura jerárquica
    y genera chunks enriquecidos con metadata de contexto (página, títulos, proyecto).
    """
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    chunks = []
    
    current_section = "Introducción / General"
    full_document_summary = []

    # Regex para identificar posibles títulos/secciones en el texto
    # (Líneas cortas en mayúsculas o con numeraciones como "1.2 Uso de Logo")
    header_pattern = re.compile(r"^(?:[0-9]+\.|\b[A-ZÁÉÍÓÚÑ\s]{4,}\b)", re.MULTILINE)

    for page_num in range(len(doc)):
        page = doc[page_num]
        text = page.get_text("text").strip()

        if not text:
            continue

        # Detectar si la página inicia con una nueva sección
        lines = text.split("\n")
        first_line = lines[0].strip() if lines else ""
        
        if len(first_line) < 60 and (header_pattern.match(first_line) or first_line.isupper()):
            current_section = first_line

        full_document_summary.append(f"Pág {page_num + 1}: {first_line[:40]}...")

        # Split semántico por tamaño de caracteres con ventana deslizante (Overlap)
        words = text.split()
        start = 0

        while start < len(words):
            end = start + chunk_size
            chunk_words = words[start:end]
            chunk_text = " ".join(chunk_words)

            chunks.append({
                "text": chunk_text,
                "metadata": {
                    "chunker": "chunk_pdf_document",
                    "type": "content_chunk",
                    "project": project,
                    "path": relative_path,
                    "page": page_num + 1,
                    "section": current_section,
                    "total_pages": len(doc),
                    "symbol": f"doc:{project}:p{page_num + 1}"
                }
            })

            # Avanzamos aplicando el traslape (overlap) para mantener contexto
            start += (chunk_size - overlap)

    # ---------- Resumen General del PDF ----------
    if chunks:
        resumen_texto = f"Resumen del documento PDF `{relative_path}` ({len(doc)} páginas):\n"
        resumen_texto += "\n".join(full_document_summary[:30]) # Primeras 30 referencias
        if len(full_document_summary) > 30:
            resumen_texto += "\n..."

        chunks.append({
            "text": resumen_texto,
            "metadata": {
                "chunker": "chunk_pdf_document",
                "type": "doc_summary",
                "project": project,
                "path": relative_path,
                "total_pages": len(doc),
                "symbol": f"doc_summary:{project}"
            }
        })

    doc.close()
    return chunks

def chunk_schemaViejo(sql, relative_path, project):

    chunks = []

    # Limpieza ligera
    sql_clean = re.sub(r"/\*.*?\*/", "", sql, flags=re.DOTALL)

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

        # columnas
        for line in full_create.splitlines():
            line = line.strip().rstrip(",")

            col_match = re.match(r"`(\w+)`\s+([A-Z]+(?:\([^)]+\))?)", line, re.IGNORECASE)
            if col_match:
                tables[table_name]["columns"].append({
                    "name": col_match.group(1),
                    "type": col_match.group(2)
                })

            # ---------- FOREIGN KEYS EN ALTER TABLE ----------
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

    # ---------- 2) Chunks de relaciones ----------
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

    # ---------- 3) CREATE VIEW ----------
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

    # ---------- 4) Resumen general ----------
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