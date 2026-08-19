import asyncio
import httpx
from urllib.parse import urljoin, urlparse
import xml.etree.ElementTree as ET

from bs4 import BeautifulSoup
import trafilatura

def debug(debug):
    showLogs = True
    if(showLogs):
        print(debug)

async def extraer_urls_sitemap_async(base_url, client):
    """Descarga y extrae el sitemap de forma asíncrona."""
    urls = set()
    sitemap_url = urljoin(base_url, "/sitemap.xml")
    try:
        response = await client.get(sitemap_url, timeout=10.0)
        if response.status_code == 200:
            root = ET.fromstring(response.text)
            for elem in root.iter():
                if 'loc' in elem.tag:
                    urls.add(elem.text.strip())
    except Exception as e:
        debug(f"Sitemap no disponible o error: {e}")
    return urls

async def procesar_pagina_async(url, client, base_url, dominio_base):
    """Descarga una página, extrae su texto limpio y descubre nuevos enlaces."""
    try:
        response = await client.get(url, timeout=10.0, follow_redirects=True)
        if response.status_code != 200:
            return None, set()

        html = response.text
        
        # 1. Trafilatura necesita el HTML en texto para extraer
        texto_limpio = trafilatura.extract(html, output_format='markdown')
        resultado = None
        #if texto_limpio and len(texto_limpio) > 50:
        resultado = {"url": url, "texto": texto_limpio}

        # 2. Extraer nuevos enlaces con BeautifulSoup
        nuevos_enlaces = set()
        soup = BeautifulSoup(html, 'html.parser')
        for link in soup.find_all('a', href=True):
            next_url = urljoin(base_url, link['href']).split('#')[0]
            # Filtrar para quedarse en el dominio
            if urlparse(next_url).netloc == dominio_base:
                nuevos_enlaces.add(next_url)
                
        return resultado, nuevos_enlaces
    except Exception as e:
        debug(f"Error descargando {url}: {e}")
        return None, set()

async def crawl_site_async(base_url, max_paginas=20, max_concurrencia=5):
    """
    Crawler concurrente. Descarga las páginas en lotes para no bloquear el servidor.
    """
    urls_visitadas = set()
    contenidos_extraidos = []
    dominio_base = urlparse(base_url).netloc
    
    # Abrimos un cliente HTTP asíncrono
    async with httpx.AsyncClient(verify=False) as client:
        # Llenar la cola inicial con el sitemap y la página base
        urls_por_visitar = await extraer_urls_sitemap_async(base_url, client)
        urls_por_visitar.add(base_url)
        
        while urls_por_visitar and len(contenidos_extraidos) < max_paginas:
            # Tomamos un lote de URLs para procesar al mismo tiempo (Concurrencia)
            lote_actual = []
            while urls_por_visitar and len(lote_actual) < max_concurrencia:
                url = urls_por_visitar.pop()
                if url not in urls_visitadas:
                    lote_actual.append(url)
                    urls_visitadas.add(url)
            
            if not lote_actual:
                break
                
            debug(f"Procesando lote de {len(lote_actual)} URLs concurrentes...")
            
            # Ejecutamos las descargas de este lote AL MISMO TIEMPO
            tareas = [procesar_pagina_async(u, client, base_url, dominio_base) for u in lote_actual]
            resultados_lote = await asyncio.gather(*tareas)
            
            # Recopilamos el texto extraído y los enlaces descubiertos
            for resultado_texto, enlaces_descubiertos in resultados_lote:
                if resultado_texto:
                    contenidos_extraidos.append(resultado_texto)
                
                # Añadimos los enlaces descubiertos a la cola de pendientes
                for enlace in enlaces_descubiertos:
                    if enlace not in urls_visitadas:
                        urls_por_visitar.add(enlace)
                        
    return contenidos_extraidos[:max_paginas] # Asegurar límite exacto