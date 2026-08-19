# -*- coding: utf-8 -*-
from datetime import datetime
import json
import os
import time
from typing import Any, Optional
import uuid

from fastapi import FastAPI, Request, UploadFile, Depends, HTTPException, Header
from fastapi.responses import StreamingResponse
from langsmith import traceable
from pydantic import BaseModel
from qdrant_client import QdrantClient
from qdrant_client.http.models import Filter, FieldCondition, MatchValue, PointStruct
import tiktoken
import uvicorn

from accionesQdrant import Qdrant, conectarQdrant
from accionesGemini import conectarGemini, generate_response, generate_response_streaming, embed_with_gemini
from accionesChutes import  generate_response_chutes_streaming
from accionesLiteLLM import generate_response_litellm_streaming, generate_response_litellm_simple
from funciones import debug, crawl_site_async
from tools import sqlTools, codigoTools, systemTools, shotsTools, fileTools

ADMIN_KEY = os.environ.get("ADMIN_API_KEY")
def verificar_clave(api_key: str = Header(...)):
    if api_key != ADMIN_KEY:
        raise HTTPException(status_code=403, detail="No autorizado: Clave inválida")

LITELLM_PROXY_KEY = os.environ["LITELLM_PROXY_KEY"]
LITELLM_PROXY_URL = os.environ["LITELLM_PROXY_URL"]
QDRANT_URL = os.environ["QDRANT_URL"]
QDRANT_API_KEY = os.environ.get("QDRANT_API_KEY") 
KEY_FREE2 = os.environ.get("GOOGLE_API_KEY2") 
GOOGLE_API_KEY= os.environ.get('KEY-FREE') 
CHUTES_API_KEY= os.environ.get('CHUTES_API_KEY') 

conectarGemini(GOOGLE_API_KEY)

def optimizar_y_aplanar_historial(historial: Any, max_tokens: int):
    """
    Parsea, limpia y aplana el historial sin importar si viene como 
    String JSON o como lista de diccionarios desde PHP.
    """
    if isinstance(historial, str):
        try:
            historial = json.loads(historial)
        except Exception:
            return []
            
    if not isinstance(historial, list):
        return []

    encoding = tiktoken.get_encoding("cl100k_base")
    historial_plano_final = []
    tokens_acumulados = 0
    
    for turno in reversed(historial):
        
        if isinstance(turno, dict):
            u_text = turno.get("user", "")
            a_text = turno.get("assistant", "")
        else:
            continue
            
        tokens_user = len(encoding.encode(str(u_text))) + 4
        tokens_assistant = len(encoding.encode(str(a_text))) + 4
        tokens_turno = tokens_user + tokens_assistant
        
        if tokens_acumulados + tokens_turno > max_tokens:
            break
            
        componentes_turno = [
            {"role": "user", "content": u_text},
            {"role": "assistant", "content": a_text}
        ]
        
        historial_plano_final = componentes_turno + historial_plano_final
        tokens_acumulados += tokens_turno
    return historial_plano_final

@traceable
async def agenteGemini(historialModificado, objTools, objShots, objCodigo, objSystem, query, model_name, archivos, system_instruction):
    historial_gemini = []
    for turno in historialModificado:
        # Gemini exige 'model' en lugar de 'assistant'
        rol_gemini = "model" if turno["role"] == "assistant" else "user"
        historial_gemini.append({
            "role": rol_gemini,
            "parts": [turno["content"]]  
        })
    lista_tools = [
        objSystem.buscar_herramientas_personalizadas_php, 
        objSystem.ejecutar_herramienta_personalizada_php, 
        objShots.buscar_ejemplos_few_shots,
        objTools.buscar_conocimiento_base_datos, 
        objTools.ejecutar_consulta_php
        ]
    if(objCodigo):
        lista_tools.append(objCodigo.buscar_conocimiento_fragmentos_codigo)

    async for paso in generate_response_streaming(
        prompt=query,
        model_name=model_name,
        archivos=archivos,
        tools=lista_tools,
        system_instruction=system_instruction,
        history=historial_gemini
    ):
        yield paso

@traceable
async def agenteChutes(historialModificado, objTools, objShots, objCodigo, objSystem, query, model_name, archivos, system_instruction):
    historial_chutes = []
    for turno in historialModificado:
        rol_chutes = "assistant" if turno["role"] == "assistant" else "user"
        historial_chutes.append({
            "role": rol_chutes,
            "content": turno["content"]  # En OpenAI es directo string, sin el "parts" de Gemini
        })

    tools_schemas = [
        {
            "type": "function",
            "function": {
                "name": "buscar_herramientas_personalizadas_php",
                "description": "Obtiene el catálogo completo de todas las funciones y herramientas personalizadas de negocio disponibles en el servidor PHP, incluyendo sus nombres, descripciones y los parámetros exactos requeridos para su ejecución.",
                "parameters": {
                    "type": "object",
                    "properties": {} 
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "ejecutar_herramienta_personalizada_php",
                "description": "Ejecuta una función específica en el backend de PHP. PROHIBIDO: No inventes parámetros. Los argumentos enviados en el objeto 'argumentos' deben coincidir estrictamente con los tipos de datos y nombres requeridos por el contrato obtenido previamente mediante la herramienta 'buscar_herramientas_personalizadas_php'.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "nombre_funcion": {
                            "type": "string",
                            "description": "El nombre exacto de la función a ejecutar (ej: 'aplicar_descuento_lote')."
                        },
                        "argumentos": {
                            "type": "object",
                            "description": "Un objeto (JSON) con los parámetros requeridos por la función, tal como los especificó el catálogo."
                        }
                    },
                    "required": [
                        "nombre_funcion",
                        "argumentos"
                    ]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "buscar_conocimiento_base_datos",
                "description": "Busca esquemas de tablas, descripciones lógicas, relaciones de llaves foráneas y lógica de negocio en la base de datos del proyecto actual",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Términos o conceptos de negocio a buscar (ej: 'mantenimientos', 'pagos')."
                        }
                    },
                    "required": ["query"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "ejecutar_consulta_php",
                "description": "Ejecuta una consulta SQL estrictamente SELECT en el servidor de producción para recuperar filas de datos reales.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "sql": {
                            "type": "string",
                            "description": "Sentencia SQL SELECT limpia completa y válida (ej: 'SELECT nombre, saldo FROM clientes WHERE saldo > 10000 LIMIT 20;')."
                        }
                    },
                    "required": ["sql"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "buscar_ejemplos_few_shots",
                "description": "Busca ejemplos históricos (few-shots) de cómo el sistema ha resuelto exitosamente peticiones similares en el pasado. Útil para entender qué herramientas usar, cómo encadenarlas y cómo corregir errores SQL o lógicos.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "La intención o pregunta actual del usuario (ej: 'lista de lotes y dueños')."
                        }
                    },
                    "required": ["query"]
                }
            }
        }
    ]
    tool_functions = {
        "buscar_herramientas_personalizadas_php": objSystem.buscar_herramientas_personalizadas_php,
        "ejecutar_herramienta_personalizada_php": objSystem.ejecutar_herramienta_personalizada_php,
        "buscar_ejemplos_few_shots": objShots.buscar_ejemplos_few_shots,
        "buscar_conocimiento_base_datos": objTools.buscar_conocimiento_base_datos,
        "ejecutar_consulta_php": objTools.ejecutar_consulta_php,
    }

    if(objCodigo):
        tools_schemas.append({
            "type": "function",
            "function": {
                "name": "buscar_conocimiento_fragmentos_codigo",
                "description": "Busca fragmentos de código, clases, métodos y controladores dentro del framework de desarrollo del usuario para entender cómo interactuar o programar con sus sistemas.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Concepto técnico, nombre de clase, método o funcionalidad a buscar en el código (ej: 'cómo usar la clase objAjuste', 'sintaxis de selects en el framework')."
                        }
                    },
                    "required": ["query"]
                }
            }
        })
        tool_functions["buscar_conocimiento_fragmentos_codigo"] = objCodigo.buscar_conocimiento_fragmentos_codigo

    async for paso in generate_response_chutes_streaming(
        prompt=query,
        model_name=model_name,
        api_key=CHUTES_API_KEY,
        archivos=archivos,
        tools_schemas=tools_schemas,
        tool_functions=tool_functions,
        system_instruction=system_instruction,
        history=historial_chutes
    ):
        yield paso

@traceable
async def agenteLitellm(historialModificado, objTools, objShots, objCodigo, objFile, objSystem, query, model_name, archivos, system_instruction):
    historial_litellm = []
    for turno in historialModificado:
        rol_litellm = "assistant" if turno["role"] == "assistant" else "user"
        historial_litellm.append({
            "role": rol_litellm,
            "content": turno["content"]
        })

    tools_schemas = [
        {
            "type": "function",
            "function": {
                "name": "buscar_herramientas_personalizadas_php",
                "description": "Obtiene el catálogo completo de todas las funciones y herramientas personalizadas de negocio disponibles en el servidor PHP, incluyendo sus nombres, descripciones y los parámetros exactos requeridos para su ejecución.",
                "parameters": {
                    "type": "object",
                    "properties": {} 
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "ejecutar_herramienta_personalizada_php",
                "description": "Ejecuta una función específica en el backend de PHP. PROHIBIDO: No inventes parámetros. Los argumentos enviados en el objeto 'argumentos' deben coincidir estrictamente con los tipos de datos y nombres requeridos por el contrato obtenido previamente mediante la herramienta 'buscar_herramientas_personalizadas_php'.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "nombre_funcion": {
                            "type": "string",
                            "description": "El nombre exacto de la función a ejecutar (ej: 'aplicar_descuento_lote')."
                        },
                        "argumentos": {
                            "type": "object",
                            "description": "Un objeto (JSON) con los parámetros requeridos por la función, tal como los especificó el catálogo."
                        }
                    },
                    "required": ["nombre_funcion", "argumentos"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "buscar_conocimiento_base_datos",
                "description": "Busca esquemas de tablas, descripciones lógicas, relaciones de llaves foráneas y lógica de negocio en la base de datos del proyecto actual",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Términos o conceptos de negocio a buscar (ej: 'mantenimientos', 'pagos')."
                        }
                    },
                    "required": ["query"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "ejecutar_consulta_php",
                "description": "Ejecuta una consulta SQL estrictamente SELECT en el servidor de producción para recuperar filas de datos reales.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "sql": {
                            "type": "string",
                            "description": "Sentencia SQL SELECT limpia completa y válida (ej: 'SELECT nombre, saldo FROM clientes WHERE saldo > 10000 LIMIT 20;')."
                        }
                    },
                    "required": ["sql"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "buscar_ejemplos_few_shots",
                "description": "Busca ejemplos históricos (few-shots) de cómo el sistema ha resuelto exitosamente peticiones similares en el pasado. Útil para entender qué herramientas usar, cómo encadenarlas y cómo corregir errores SQL o lógicos.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "La intención o pregunta actual del usuario (ej: 'lista de lotes y dueños')."
                        }
                    },
                    "required": ["query"]
                }
            }
        }
    ]
    
    tool_functions = {
        "buscar_herramientas_personalizadas_php": objSystem.buscar_herramientas_personalizadas_php,
        "ejecutar_herramienta_personalizada_php": objSystem.ejecutar_herramienta_personalizada_php,
        "buscar_ejemplos_few_shots": objShots.buscar_ejemplos_few_shots,
        "buscar_conocimiento_base_datos": objTools.buscar_conocimiento_base_datos,
        "ejecutar_consulta_php": objTools.ejecutar_consulta_php,
    }

    if(objCodigo):
        tools_schemas.append({
            "type": "function",
            "function": {
                "name": "buscar_conocimiento_fragmentos_codigo",
                "description": "Busca fragmentos de código, clases, métodos y controladores dentro del framework de desarrollo del usuario para entender cómo interactuar o programar con sus sistemas.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Concepto técnico, nombre de clase, método o funcionalidad a buscar en el código (ej: 'cómo usar la clase objAjuste', 'sintaxis de selects en el framework')."
                        }
                    },
                    "required": ["query"]
                }
            }
        })
        tool_functions["buscar_conocimiento_fragmentos_codigo"] = objCodigo.buscar_conocimiento_fragmentos_codigo

    if(objFile):
        tools_schemas.append({
            "type": "function",
            "function": {
                "name": "buscar_conocimiento_archivos",
                "description": "Busca información relevante en los manuales de marca, documentos PDF, políticas, servicios e información general del negocio del proyecto actual.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Términos, preguntas o conceptos a buscar (ej: 'uso de logo', 'horarios', 'garantía', 'servicios')."
                        }
                    },
                    "required": ["query"]
                }
            }
        },)
        tool_functions["buscar_conocimiento_archivos"] = objFile.buscar_conocimiento_archivos

    # 🚀 Ejecutamos el streaming a través de LiteLLM
    async for paso in generate_response_litellm_streaming(
        prompt=query,
        model_name=model_name,
        proxy_key=LITELLM_PROXY_KEY, 
        proxy_url=LITELLM_PROXY_URL, 
        archivos=archivos,
        tools_schemas=tools_schemas,
        tool_functions=tool_functions,
        system_instruction=system_instruction,
        history=historial_litellm
    ):
        yield paso


app = FastAPI()

@app.get("/health")
async def health():
    return {
        "status": "ok"
    }

@traceable
@app.post("/devaiAgent", dependencies=[Depends(verificar_clave)])
async def devai_endpoint(request: Request):
    client = conectarQdrant(QDRANT_URL, QDRANT_API_KEY)

    default_system_instruction = """
        Eres un asistente de desarrollo extremadamente preciso y especializado en interpretar código PHP, HTML y SQL dentro de un framework personalizado.

        Tu objetivo es resolver las consultas del usuario utilizando de forma proactiva las herramientas (Tools) a tu disposición para consultar la base de datos de conocimiento, esquemas, análisis y código fuente real.

        REGLAS CRÍTICAS DE OPERACIÓN:
        1. **Veracidad Estricta:** No inventes, asumas, ni completes nada que no esté explícitamente en la información recuperada por tus herramientas. Si la información no está ahí, no existe para ti.
        2. **Insuficiencia de Información:** Si tras ejecutar tus herramientas consideras que no hay suficiente información para responder con certeza, detén tu análisis y responde claramente que no es posible contestar, detallando con precisión qué dato o fragmento te hace falta.
        3. **Privacidad del Contexto:** No menciones de qué fragmento de código, tabla exacta o tool provino la información. No digas cosas como "según el chunk recuperado...". Simplemente asimila el conocimiento y responde formalmente.
        4. **Reglas del Framework (Interfaz/Vistas):** Al analizar o generar código de vistas, NO inventes inputs ni etiquetas HTML estándar, a menos que se te pida. Utiliza siempre la clase 'Ximhai' o los ejemplos de código reales obtenidos mediante tus herramientas para guiar la estructura.
        5. **Eficiencia SQL Estricta (PROHIBIDO BUCLES N+1):** Cuando uses la herramienta 'ejecutar_consulta_php', está TERMINANTEMENTE PROHIBIDO realizar múltiples consultas consecutivas o individuales para procesar listas de elementos. Si necesitas datos de varios registros o validar una lista, debes estructurar UNA SOLA CONSULTA limpia utilizando operadores como 'IN', 'BETWEEN' o agrupaciones mediante 'INNER JOIN'. Maximiza la eficiencia y minimiza las llamadas al servidor.

        REGLAS DE FORMATO Y RESPUESTA:
        - Responde de forma concreta, profesional y directa al grano.
        - No repitas estas instrucciones del sistema ni hagas resúmenes innecesarios de todo lo que encontraste.
        - No generes estructuras de código incompletas o "falsas".
        - Todo código fuente generado o citado debe ir estrictamente encasillado dentro de bloques de marcado triple: ```.
        """ 

    #Primero, los datos del request. Todos con sus propios defaults
    form_data = await request.form()
    query = form_data.get("query", "")
    memoria = form_data.get("memoria", "DevAI-Memory")
    chat_id = int(form_data.get("chat_id", 0))
    codigo = form_data.get("codigo", "DEVAI-embeddings")
    bd = form_data.get("basedatos", form_data.get("bd", "DevAI-DB"))
    archivo = form_data.get("analisis", form_data.get("archivo", "DevAI-Analisis"))
    proyecto = form_data.get("proyecto", "default")
    proveedor = form_data.get("proveedor", "gemini")
    system_instruction = form_data.get("system_instruction", default_system_instruction)
    url = form_data.get("url", "https://ximhai.com")
    conCodigo = form_data.get("conCodigo", False)
    conArchivos = form_data.get("conArchivos", False)
    
    model_name = form_data.get("model_name", "models/gemini-3.1-flash-lite") 
    historial = form_data.get("historial", "")
    max_tokens = int(form_data.get("max_tokens", 6000))
    guardarFewShot = form_data.get("fewShot", False)

    datosExtra = form_data.get("datosExtra", False)
    
    #Por si vienen archivos
    archivos_procesados = []
    for key, value in form_data.items():
        if key.startswith("files[") and hasattr(value, "filename"):
            contenido_bytes = await value.read()
            archivos_procesados.append({
                "mime_type": value.content_type,   
                "data": contenido_bytes          
            })
    #El historial lo aplanamos al numero de tokens que traemos por defecto
    historialModificado = optimizar_y_aplanar_historial(historial, max_tokens)

    
    KEY= os.environ.get(f"{proyecto}_KEY", 'KEY-FREE') 
    debug('KEY')
    debug(KEY)
    if(KEY == 'KEY-FREE'):
        KEY = GOOGLE_API_KEY
    conectarGemini(KEY)

    objQdrant = Qdrant(
        client=client,  
        collection=bd,
        proyecto=proyecto
    )
    objMemoria = Qdrant(
        client=client,
        collection=memoria,
        proyecto=proyecto
    )
    objShotsQ = Qdrant(
        client=client,
        collection='DevAI-FewShots',
        proyecto=proyecto
    )
    
    objQdrantCodigo = Qdrant(
        client=client,  
        collection=codigo,
        proyecto=None
    )

    objQdrantFile = Qdrant(
        client=client,  
        collection=archivo,
        proyecto=proyecto
    )


    #Objetos de tools, ya con el objeto de Qdrant para el tema de la collection y url para ejecutar la consulta en php.
    objTools = sqlTools(objQdrant=objQdrant, url=url)
    objShots = shotsTools(objQdrant=objShotsQ)
    objCodigo = codigoTools(objQdrant=objQdrantCodigo)
    objFile = fileTools(objQdrant=objQdrantFile)
    objSystem = systemTools(url=url, datosExtra = datosExtra)

    if(conCodigo == False):
        objCodigo = None
    if(conArchivos == False):
        objFile = None

    queryAux = query
    query = f"<mainQuery>{query}</mainQuery"
    
  
    async def generar_eventos_stream():
        tokens_entrada_acumulados = 0
        tokens_salida_acumulados = 0
        textoRespuesta = ""
        extraInfo = {
            "name": f"agenteLiteLLM{proyecto}" ,
            "metadata": {
                "chat_id": chat_id,
                "modelo": model_name

            }
        }
        if(proveedor == 'gemini'):
            streamingTexto = agenteGemini(historialModificado, objTools , objShots , objCodigo, objSystem, query, model_name, archivos_procesados, system_instruction, langsmith_extra={"name": f"agenteGemini{proyecto}"})
        elif (proveedor == 'litellm'):
            streamingTexto = agenteLitellm(historialModificado, objTools , objShots , objCodigo, objFile, objSystem, query, model_name, archivos_procesados, system_instruction, langsmith_extra= extraInfo)
        else:
            streamingTexto = agenteChutes(historialModificado, objTools , objShots , objCodigo, objSystem, query, model_name, archivos_procesados, system_instruction, langsmith_extra={"name": f"agenteChutes{proyecto}"})

        async for chunk in streamingTexto:
            #debug('chunk en agente gemini:')
            #debug(chunk)
            if chunk.get("type") == "error":
                yield f"{json.dumps({ 'type': 'error', 'content': chunk['content']}, ensure_ascii=False)}\n\n"
            #CoT
            if chunk.get("type") == "thought":
                #debug(f"El LLM penso :" + chunk['content'])
                yield f"{json.dumps({'type': 'thought', 'content': chunk['content']}, ensure_ascii=False)}\n\n"

            elif chunk.get("type") == "token":
                textoRespuesta += chunk["content"]  # Buffer para guardar posteriormente en Qdrant
                yield f"{json.dumps({'type': 'token', 'content': chunk['content']}, ensure_ascii=False)}\n\n"

            elif chunk.get("type") == "metrics":
                tokens_entrada_acumulados = chunk["tokens_entrada"]
                tokens_salida_acumulados = chunk["tokens_salida"]
                cot = chunk['chain_of_thought']

        debug(f"Streaming finalizado. Guardando memoria... Chars: {len(textoRespuesta)}")
        debug(f" TOKENS en Agentic , TokIn+: {tokens_entrada_acumulados}, TokOut+: {tokens_salida_acumulados}")
        # Guardado en la memoria de Qdrant  
        uuids = []
        #debug(f"Texto Respuesta: {textoRespuesta.strip()}")

        uuids = await objMemoria.save_to_qdrant(
            embed_fn=embed_with_gemini,
            user_query=queryAux,
            collection_memory=memoria,
            respuesta=textoRespuesta.strip(),
            chat_id=chat_id,
            proyecto=proyecto
        )

        if(guardarFewShot):
            uuids = await objMemoria.guardarShot(
                embed_fn=embed_with_gemini,
                user_query=queryAux,
                collection_memory="DevAI-FewShots",
                cot = cot,
                respuesta=textoRespuesta.strip(),
                proyecto=proyecto
            )

        respuesta_final_metadata = {
            "type": "final_metadata",
            "uuids": uuids, 
            "tokens_entrada": tokens_entrada_acumulados, 
            "tokens_salida": tokens_salida_acumulados,
        }
        yield f"{json.dumps(respuesta_final_metadata, ensure_ascii=False)}\n\n"

    return StreamingResponse(generar_eventos_stream(), media_type="text/event-stream")


#
# =================================================================
# NUEVO APARTADO: ENDPOINT PARA PROMPTS LIBRES (SIN RAG / QDRANT)
# =================================================================

class FreePromptRequest(BaseModel):
    prompt: str
    model_name: str 

@app.post("/prompt", dependencies=[Depends(verificar_clave)])
async def free_prompt_endpoint(request: FreePromptRequest):
    conectarGemini(KEY_FREE2)
    try:
        if not request.prompt:
            respuesta = {'error': "No se recibió un prompt válido"  }
            return {"response": respuesta}
        if not request.model_name:
            respuesta = {'error': "No se recibió un modelo válido"  }
            return {"response": respuesta}


        response = await generate_response(request.prompt, request.model_name)

        response = response["texto"].strip()
        
        return {"response": response}
        
    except Exception as e:
        return {"error": str(e)}, 500


# =================================================================
# NUEVO APARTADO: Acciones de QDRANT para borrar puntos 
# =================================================================
class BorrarChatRequest(BaseModel):
    collection_name: str
    chat_id: int

class BorrarPuntoRequest(BaseModel):
    collection_name: str
    point_id: str

@app.post("/borrar_chat", dependencies=[Depends(verificar_clave)])
async def endpoint_borrar_chat(request: BorrarChatRequest):
    client = conectarQdrant(QDRANT_URL, QDRANT_API_KEY)
    """Endpoint para borrar todo el historial de un chat por ID."""
    try:
        res = await borrar_por_chat_id(client, request.collection_name, request.chat_id)
        return {"status": "success", "result": res.status}
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.post("/borrar_punto", dependencies=[Depends(verificar_clave)])
async def endpoint_borrar_punto(request: BorrarPuntoRequest):
    client = conectarQdrant(QDRANT_URL, QDRANT_API_KEY)
    """Endpoint para borrar un único punto por su ID."""
    try:
        res = await borrar_por_point_id(client, request.collection_name, request.point_id)
        return {"status": "success", "result": res.status}
    except Exception as e:
        return {"status": "error", "message": str(e)}



#
# =================================================================
# NUEVO APARTADO: Para poner una nueva base de datos
# =================================================================

@app.post("/nueva-bd", dependencies=[Depends(verificar_clave)])
async def devai_endpoint(request: Request):
    client = conectarQdrant(QDRANT_URL, QDRANT_API_KEY)
    
    
    form_data = await request.form()
    
    descripcion = form_data.get("descripcion", "")
    proyecto = form_data.get("proyecto", "")
    
    objQdrant = Qdrant(
        client=client,  
        collection='DevAI-DB',
        proyecto=proyecto
    )

    archivos_procesados = []
    for key, value in form_data.items():
        if key.startswith("files[") and hasattr(value, "filename"):
            contenido_bytes = await value.read()
            archivos_procesados.append({
                "mime_type": value.content_type,   
                "data": contenido_bytes,
                "filename": value.filename
            })
    archivo = archivos_procesados[0]
    respuesta =  await objQdrant.embebirBaseDatos(descripcion, archivo, proyecto)
    return {"response": respuesta}



#
# =================================================================
# NUEVO APARTADO: Para embebir archivos
# =================================================================
@app.post("/nuevo-archivo", dependencies=[Depends(verificar_clave)])
async def devai_endpoint(request: Request):
    client = conectarQdrant(QDRANT_URL, QDRANT_API_KEY)
    form_data = await request.form()
    
    descripcion = form_data.get("descripcion", "")
    proyecto = form_data.get("proyecto", "")
    model_name = form_data.get("model_name", "models/gemini-3.1-flash-lite")

    objQdrant = Qdrant(
        client=client,  
        collection='DevAI-DB',
        proyecto=proyecto
    )

    archivos_procesados = []
    for key, value in form_data.items():
        if key.startswith("files[") and hasattr(value, "filename"):
            contenido_bytes = await value.read()
            archivos_procesados.append({
                "mime_type": value.content_type,   
                "data": contenido_bytes,
                "filename": value.filename
            })
    
    respuesta =  await objQdrant.embebirArchivos(descripcion, archivos_procesados, proyecto)
    return {"response": respuesta}



#
# =================================================================
# NUEVO APARTADO: Para hacer crawl
# =================================================================
@app.post("/crawl", dependencies=[Depends(verificar_clave)])
async def endpoint_crawl(request: Request):
    form_data = await request.form()
    
    url = form_data.get("url", "https://ximhai.com")
    modelo = form_data.get("modelo", "gemini-lite")
    system_instruction = form_data.get("system_instruction", None)
    if(system_instruction == None or system_instruction == ''):
        return {
                "status": "error",
                "mensaje": "No se recibió un system_instruction válido"
            }

    
    # Aquí puedes ajustar cuántas páginas quieres y de a cuántas concurrentes
    paginas_extraidas = await crawl_site_async(
        base_url=url, 
        max_paginas=50, 
        max_concurrencia=5 # Descargará 5 páginas a la vez
    ) 
    textos_limpios = []
    for pagina in paginas_extraidas:
        prompt = f"{system_instruction}. \n\n TEXTO:{pagina}"
        texto = await generate_response_litellm_simple( prompt = prompt , model_name= modelo, proxy_key=LITELLM_PROXY_KEY, proxy_url=LITELLM_PROXY_URL )
        if(texto.type == 'error'):
            pass
        textos_limpios.append(texto) 

    return {
        "status": "success",
        "total_paginas": len(paginas_extraidas),
        "data": paginas_extraidas,
        "dataLimpia" : textos_limpios
    }