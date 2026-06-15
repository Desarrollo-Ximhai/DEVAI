# accionesQdrant.py
import uuid
import os
from qdrant_client import QdrantClient
from qdrant_client.http.models import Filter, FieldCondition, MatchValue, PointStruct, SparseVector, Prefetch, Fusion, FusionQuery
from qdrant_client.http import models 
from datetime import datetime
from fastembed.sparse import SparseTextEmbedding
from funciones import debug
sparse_model = SparseTextEmbedding(model_name="Qdrant/bm25")
import requests
import re



def conectarQdrant(qdrant_url, qdrant_api_key):
    try:
        qdrant_api_key = qdrant_api_key
        print("Qdrant API Key obtenida de los secretos de Colab.")
    except:
        qdrant_api_key = None
        print("No se encontró 'QDRANT_API_KEY' en los secretos de Colab. Si tu instancia de Qdrant requiere una API Key, asegúrate de haberla guardado correctamente.")

    # Conecta con tu Qdrant (local o remoto)
    client = QdrantClient(
        url= qdrant_url,  # o tu URL remota
        api_key=qdrant_api_key # Pasa la API key durante la inicialización
    )
    return client

def borrar_por_chat_id(client: QdrantClient, collection_name: str, chat_id: int):
    """
    Borra todos los puntos en una colección que pertenezcan a un chat_id específico.
    """
    # Filtro para encontrar puntos donde el chat_id coincida
    filtro = Filter(
        must=[
            FieldCondition(
                key="chat_id",
                match=MatchValue(value=chat_id)
            )
        ]
    )
    
    resultado = client.delete(
        collection_name=collection_name,
        points_selector=filtro
    )
    return resultado

def borrar_por_point_id(client: QdrantClient, collection_name: str, point_id: str):
    """
    Borra un punto específico de la colección dado su ID único (UUID).
    """
    resultado = client.delete(
        collection_name=collection_name,
        points_selector=[point_id]
    )
    return resultado


def rerank_con_langsearch(query_usuario, candidatos, top_n=4):
    """
    Usa la API de LangSearch para reordenar los chunks de Qdrant.
    Consumo de RAM local = 0 MB.
    """
    RERANK_KEY= os.environ.get('RERANK_KEY') 
    RERANK_URL= os.environ.get('RERANK_URL') 

    if not candidatos:
        return []

    url = f"{RERANK_URL}"
    headers = {
        "Authorization": f"Bearer {RERANK_KEY}", 
        "Content-Type": "application/json"
    }

    # Extraemos solo las cadenas de texto limpias de los candidatos de Qdrant
    documentos = [
        c.payload.get("text", "") if hasattr(c, "payload") else c.get("text", "")
        for c in candidatos
    ]

    payload = {
        "model": "langsearch-reranker-v1",
        "query": query_usuario,
        "top_n": top_n,
        "return_documents": False, # No necesitamos que nos devuelva el texto, solo los índices
        "documents": documentos
    }

    try:
        response = requests.post(url, json=payload, headers=headers, timeout=5)
        if response.status_code == 200:
            res_data = response.json()
            
            # Re-mapeamos los índices ganadores a tus objetos originales de Qdrant
            chunks_finales = []
            for hit in res_data.get("results", []):
                idx = hit.get("index")
                if idx is not None and idx < len(candidatos):
                    chunks_finales.append(candidatos[idx])
            
            return chunks_finales
        else:
            print(f"⚠️ LangSearch respondió con error {response.status_code}, usando fallback.")
            return candidatos[:top_n]
            
    except Exception as e:
        print(f"⚠️ Falló la conexión con LangSearch: {e}")
        # Tu RAG no se muere si se cae la API, solo usa los primeros por defecto
        return candidatos[:top_n]


#Busqueda anterior, solo era sobre los vectores, ahora lo hacemos tambien de manera dispersa(palabras clave)
# def search_in_qdrant(client, collection_name, query_embedding, k=5):
#     results = client.query_points(
#         collection_name=collection_name,
#         query=query_embedding,
#         limit=k,
#         )

#     return results.points 



def search_in_qdrant(client, collection_name, user_query, query_embedding, proyecto, k):
    global sparse_model
    filtros = []
    if proyecto:
        filtros.append(
            FieldCondition(
                key="project",
                match=MatchValue(value=proyecto)
            )
        )
    
    sparse_emb = list(sparse_model.embed(user_query))[0]
    qdrant_sparse_vector = SparseVector(
        indices=sparse_emb.indices.tolist(),
        values=sparse_emb.values.tolist()
    )
    results = client.query_points(
        collection_name=collection_name,
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
    return rerank_con_langsearch(user_query, results.points, 5) 

def save_to_qdrant(client, embed_fn, user_query, collection_memory, respuesta, chat_id, proyecto="default"):
    
    textos = [
        {"role": "user", "text": user_query.strip()},
        {"role": "assistant", "text": respuesta.strip()},
    ]

    points = []
    uuids = []
    for item in textos:
        emb = embed_fn(item["text"],768)
        if emb is None:
            continue
        unUUUID = uuid.uuid4()
        uuids.append(unUUUID)
        points.append(
            PointStruct(
                id=str(unUUUID),
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
        print("⚠️ No se generaron embeddings para guardar memoria.")
        return

    client.upsert(
        collection_name=collection_memory,
        points=points,
        wait=True
    )
    print(f"✅ Memoria guardada ({len(points)} puntos) para proyecto '{proyecto}'.")
    return uuids


def getProjectMemory(client, embed_fn, user_query, collection_memory, chat_id, proyecto="default", limit=5):
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
    query_emb = embed_fn(user_query,768)
    res = client.query_points(
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

    # ordenar cronológicamente
    # puntos.sort(
    #     key=lambda x: x.payload.get("timestamp", 0)
    # )
    
    return puntos

def embebirBaseDatos(descripcion, archivo, proyecto):
    archivos_procesados = []
    chunks_de_base_datos = [] 
    # Transformamos los bytes puros en un string de Python
    sql_string =  archivo["data"].decode("utf-8", errors="ignore")
    
    # 1. Ejecutamos tu función de chunking pasando el string directo
    chunks_base = chunk_schema(
        sql_string, 
        archivo["filename"], 
        proyecto
    )
    print('chunks_base')
    print(chunks_base)
    return chunks_base
    
    # # 2. El flujo con la IA: Iteramos tus chunks para enriquecerlos
    # for chunk in chunks_base:
    #     if chunk["metadata"]["type"] == "table":
    #         tabla_nombre = chunk["metadata"]["table"]
            
    #         # Aquí llamas a la función que le pide la descripción a Gemini
    #         # descripcion_ia = await pedir_descripcion_a_gemini(chunk["text"])
    #         descripcion_ia = "Descripción generada por el LLM para esta tabla..." 
            
    #         # Fusionamos el Markdown de la IA con el SQL original (Formato Híbrido)
    #         chunk["text"] = f"# TABLA: {tabla_nombre}\n{descripcion_ia}\n\n## SQL ORIGINAL:\n{chunk['text']}"
        
    #     # Guardamos el chunk ya procesado en nuestra lista
    #     chunks_de_base_datos.append(chunk)
        
    # print(f"🧬 Se procesó el archivo SQL '{value.filename}' en {len(chunks_de_base_datos)} chunks estructurados.")

def chunk_schema(sql, relative_path, project):
   
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


    for table_name, info in tables.items():
      print(table_name, len(info["relations"]))

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