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
import json

from accionesGemini import conectarGemini, generate_response, embed_with_gemini

def conectarQdrant( qdrant_url, qdrant_api_key):
        client = QdrantClient(
            url= qdrant_url,  
            api_key=qdrant_api_key 
        )
        return client
def rerank_con_langsearch( query_usuario, candidatos, top_n=15):
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
        documentos = []
        for c in candidatos:
            # 1. Extraer el texto y el ID original del punto
            text = c.payload.get("text", "") if hasattr(c, "payload") else c.get("text", "")
            punto_id = c.id if hasattr(c, "id") else c.get("id", "unknown_id")
            
            # 2. Si el texto viene con múltiples tablas metidas en el mismo payload
            if "# TABLA:" in text:
                sub_chunks = text.split("# TABLA:")
                sub_indice = 0
                
                for sub in sub_chunks:
                    texto_limpio = sub.strip()
                    if texto_limpio:
                        # Reconstruimos el tag '# TABLA:' que se borró en el split
                        tabla_formateada = f"# TABLA: {texto_limpio}"
                        
                        documentos.append({
                            "text": tabla_formateada,
                            "id": f"{punto_id}_{sub_indice}" # Ej: UUID-ORIGINAL_0
                        })
                        sub_indice += 1
            else:
                # Si el punto ya venía atómico o es otro tipo de contexto
                if text.strip():
                    documentos.append({
                        "text": text.strip(),
                        "id": str(punto_id)
                    })
        payload = {
            "query": query_usuario,
            "top_n": top_n,
            "documents": documentos
        }

        
        response = requests.post(url, json=payload, headers=headers, timeout=5)
        print('response)')
        print(response) 
        if response.status_code == 200:
            res_data = response.json()
            
            chunks_finales = []
            
            # 1. Creamos un diccionario para buscar rápido los candidatos originales por su ID de Qdrant
            #    Ej: {"id_de_qdrant": objeto_candidato}
            candidatos_by_id = {
                str(c.id if hasattr(c, "id") else c.get("id")): c 
                for c in candidatos
            }
            
            for hit in res_data.get("results", []):
                # 2. La API compatible con OpenAI/Chutes te devuelve el "document" o su "id"
                #    Dependiendo de la API, suele venir en hit["document"]["id"] o hit["id"]
                doc_id = hit.get("document", {}).get("id") or hit.get("id")
                
                if doc_id:
                    # 3. Rompemos el ID para quitarle el sufijo '_índice' (ej: "UUID_0" -> "UUID")
                    qdrant_id = str(doc_id).split("_")[0]
                    
                    # 4. Recuperamos el objeto original completo de Qdrant
                    candidato_original = candidatos_by_id.get(qdrant_id)
                    
                    # Evitamos duplicados en 'chunks_finales' por si dos tablas del mismo punto fueron relevantes
                    if candidato_original and candidato_original not in chunks_finales:
                        chunks_finales.append(candidato_original)
            
           
            
            return chunks_finales
        
        # 🔍 AQUÍ ESTÁ EL AJUSTE PARA INVESTIGAR EL ERROR 500
        else:
            print(f"⚠️ LangSearch respondió con error {response.status_code}, usando fallback.")
            print("──────────────────────────────────────────────────")
            print("🚨 [DETALLE DEL ERROR DE LANGSEARCH]:")
            try:
                # Intentamos leer si la API mandó un JSON con el mensaje de error
                print(response.json())
            except Exception:
                # Si no es un JSON, imprimimos el HTML o texto plano crudo que mandó el servidor
                print(response.text)
            print("──────────────────────────────────────────────────")
            
            return candidatos[:top_n]
class Qdrant:
    def __init__(self, client, collection, proyecto):
        self.client = client
        self.collection = collection
        self.proyecto = proyecto
        

    

    def borrar_por_chat_id(self, collection_name: str, chat_id: int):
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
        
        resultado = self.client.delete(
            collection_name=collection_name,
            points_selector=filtro
        )
        return resultado

    def borrar_por_point_id(self, collection_name: str, point_id: str):
        """
        Borra un punto específico de la colección dado su ID único (UUID).
        """
        resultado = self.client.delete(
            collection_name=collection_name,
            points_selector=[point_id]
        )
        return resultado


    


    #Busqueda anterior, solo era sobre los vectores, ahora lo hacemos tambien de manera dispersa(palabras clave)
    # def search_in_qdrant(client, collection_name, query_embedding, k=5):
    #     results = client.query_points(
    #         collection_name=collection_name,
    #         query=query_embedding,
    #         limit=k,
    #         )

    #     return results.points 



    def search_in_qdrant(self, user_query, query_embedding, k):
        global sparse_model
        filtros = []

        if self.proyecto:
            filtros.append(
                FieldCondition(
                    key="project",
                    match=MatchValue(value=self.proyecto)
                )
            )
        
        sparse_emb = list(sparse_model.embed(user_query))[0]
        qdrant_sparse_vector = SparseVector(
            indices=sparse_emb.indices.tolist(),
            values=sparse_emb.values.tolist()
        )
        results = self.client.query_points(
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
        return rerank_con_langsearch(user_query, results.points, 10) 

    def save_to_qdrant(self, embed_fn, user_query, collection_memory, respuesta, chat_id, proyecto="default"):
        
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

        self.client.upsert(
            collection_name=collection_memory,
            points=points,
            wait=True
        )
        print(f"✅ Memoria guardada ({len(points)} puntos) para proyecto '{proyecto}'.")
        return uuids


    def getProjectMemory(self, embed_fn, user_query, collection_memory, chat_id, proyecto="default", limit=5):
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
        res = self.client.query_points(
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

    def embebirBaseDatos(self, descripcion, archivo, proyecto):
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

        SQL COMPLETO:
        {sql_string}
        """

        
        respuesta = generate_response(prompt,configuracion={"tipo": "application/json"})
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
            embedding = embed_with_gemini(chunk["text"], 768, "retrieval_document")
            chunk_with_embedding = {
                "text": chunk['text'],
                "metadata": chunk['metadata'],
                "embedding": embedding
            }
            chunks_with_embeddings.append(chunk_with_embedding);

        collection_name = "DevAI-DB"
        try:
            self.client.delete(
                collection_name=collection_name,
                points_selector=models.Filter(
                    should=[
                        # Opción A: El proyecto está en la raíz del payload
                        models.FieldCondition(
                            key="project",
                            match=models.MatchValue(value=proyecto)
                        ),
                    
                    ]
                )
            )
        except Exception as e:
            print(f"[⚠️ ADVERTENCIA] No se pudo borrar o no existían puntos previos: {e}")

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
            self.client.upsert(
                collection_name=collection_name,
                wait=True,
                points=points
            )
            print(f"✅ Se han subido exitosamente {len(points)} chunks actualizados a la colección '{collection_name}'.")
        except Exception as e:
            print(f"[ERROR CRÍTICO] al subir los chunks a Qdrant: {e}")


        return True


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