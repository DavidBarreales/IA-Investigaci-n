# =========================
# IMPORTS Y CONFIGURACIÓN
# =========================
import os  # Operaciones de sistema de archivos
import re  # Expresiones regulares
import fitz  # PyMuPDF para manipulación de PDFs
import torch  # Operaciones con tensores y modelos en GPU
import faiss  # Búsqueda vectorial eficiente
import pdfplumber  # Extracción de texto y palabras de PDFs
import requests  # Llamadas HTTP
import numpy as np  # Operaciones numéricas
import logging  # Control de logs
from PIL import Image  # Procesamiento de imágenes
from sentence_transformers import SentenceTransformer  # Embeddings semánticos
from transformers import (
    MarianMTModel, MarianTokenizer,  # Traducción
    Pix2StructProcessor, Pix2StructForConditionalGeneration  # Modelo visual
)
from langchain.text_splitter import RecursiveCharacterTextSplitter  # División de texto
import pytesseract  # OCR
from collections import Counter  # Utilidades de conteo
import base64  # Codificación de imágenes

# =========================
# PARÁMETROS GLOBALES
# =========================

ALTURA_CLIP = 600       # Altura mínima para recorte de imágenes

# Configuración de Tesseract OCR
pytesseract.pytesseract.tesseract_cmd = r"C:\\Program Files\\Tesseract-OCR\\tesseract.exe"
logging.getLogger("fitz").setLevel(logging.ERROR)
logging.getLogger("pdfminer").setLevel(logging.ERROR)

# =========================
# CARGA DE MODELOS
# =========================
# Modelo de embeddings para búsquedas semánticas
embed_model = SentenceTransformer("intfloat/multilingual-e5-large", device='cuda')
# Modelos de traducción inglés-español
trans_tokenizer = MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-es")
trans_model = MarianMTModel.from_pretrained("Helsinki-NLP/opus-mt-en-es")
trans_model.to("cuda")
# Modelo visual principal: Pix2Struct orientado a documentos
try:
    pix2struct_processor = Pix2StructProcessor.from_pretrained("google/pix2struct-docvqa-base")
    pix2struct_model = Pix2StructForConditionalGeneration.from_pretrained("google/pix2struct-docvqa-base").to("cuda")
    PIX2STRUCT_MODE = 'cuda'
except Exception as e:
    print(f"[WARN] No se pudo cargar Pix2Struct en GPU: {e}. Intentando en CPU...")
    pix2struct_processor = Pix2StructProcessor.from_pretrained("google/pix2struct-docvqa-base")
    pix2struct_model = Pix2StructForConditionalGeneration.from_pretrained("google/pix2struct-docvqa-base").to("cpu")
    PIX2STRUCT_MODE = 'cpu'

# =========================
# FUNCIONES AUXILIARES
# =========================
def romano_a_arabigo(romano):
    """Convierte números romanos a arábigos (para identificar figuras/tablas)."""
    romano = romano.upper()
    mapa = {'I': 1, 'V': 5, 'X': 10, 'L': 50, 'C': 100, 'D': 500, 'M': 1000}
    total = 0
    prev = 0
    for letra in reversed(romano):
        valor = mapa.get(letra, 0)
        if valor < prev:
            total -= valor
        else:
            total += valor
            prev = valor
    return str(total)

def traducir_al_espanol(texto):
    """Traduce texto al español usando MarianMT."""
    inputs = trans_tokenizer([texto], return_tensors="pt", padding=True).to("cuda")
    translated = trans_model.generate(**inputs, max_length=60)
    return trans_tokenizer.decode(translated[0], skip_special_tokens=True)

def describir_imagen(ruta_imagen):
    """
    Dada una ruta de imagen, extrae una descripción combinando OCR, Pix2Struct y LLaVA.
    - El OCR extrae texto, categorías y posibles valores.
    - Pix2Struct intenta describir visualmente la imagen.
    - LLaVA (vía Ollama) da una segunda opinión visual.
    La salida es una descripción combinada y robusta.
    """
    imagen = Image.open(ruta_imagen).convert("RGB")
    # --- OCR heurístico ---
    texto_ocr = pytesseract.image_to_string(imagen, lang='eng+spa').strip()
    lineas = [l.strip() for l in texto_ocr.splitlines() if l.strip()]
    titulo = next((l for l in lineas if re.match(r'^(gr[aá]fico|figura|tabla)[\s:.-]*', l, re.IGNORECASE)), None)
    fuente = next((l for l in lineas if re.search(r'fuente', l, re.IGNORECASE)), None)
    eje_y = next((l for l in lineas if re.search(r'(euros|millones)', l, re.IGNORECASE)), None)
    eje_x = next((l for l in lineas if re.search(r'201\d', l)), None)
    leyenda = next((l for l in lineas if "juego online" in l.lower() or "margen" in l.lower()), None)
    partes_ocr = []
    if titulo: partes_ocr.append(f"El gráfico se titula: {titulo}")
    if eje_x: partes_ocr.append(f"El eje X representa: {eje_x}")
    if eje_y: partes_ocr.append(f"El eje Y muestra: {eje_y}")
    if leyenda: partes_ocr.append(f"La leyenda incluye: {leyenda}")
    if fuente: partes_ocr.append(f"La fuente es: {fuente}")
    # OCR mejorado: buscar valores numéricos, años, porcentajes y categorías
    valores = re.findall(r'\b\d{1,3}(?:[.,]\d{3})*(?:[.,]\d+)?\b', texto_ocr)
    if valores:
        partes_ocr.append(f"Valores numéricos detectados: {', '.join(valores[:8])}{'...' if len(valores)>8 else ''}")
    anios = re.findall(r'20\d{2}', texto_ocr)
    if anios:
        partes_ocr.append(f"Años detectados: {', '.join(sorted(set(anios)))}")
    porcentajes = re.findall(r'\d{1,3}%|\d{1,3},\d+%', texto_ocr)
    if porcentajes:
        partes_ocr.append(f"Porcentajes detectados: {', '.join(porcentajes)}")
    categorias = re.findall(r'([A-Z][a-záéíóúüñ]{2,}(?: [A-Z][a-záéíóúüñ]{2,})*)', texto_ocr)
    if categorias:
        partes_ocr.append(f"Posibles categorías: {', '.join(sorted(set(categorias[:6])))}{'...' if len(categorias)>6 else ''}")
    # Extraer texto literal detectado por OCR (si hay suficiente)
    if len(texto_ocr) > 20:
        partes_ocr.append(f"Texto extraído literal OCR: {texto_ocr[:300]}{'...' if len(texto_ocr)>300 else ''}")
    descripcion_ocr = ". ".join(partes_ocr) or "[OCR] Sin texto descriptivo suficiente."
    descripcion_ocr = "[OCR] " + traducir_al_espanol(descripcion_ocr)

    # --- Pix2Struct ---
    imagen_pix2 = imagen
    if imagen_pix2.width < 800:
        imagen_pix2 = imagen_pix2.resize((imagen_pix2.width * 2, imagen_pix2.height * 2))
    # PROMPT GENÉRICO
    prompt_pix2struct = (
        "Describe con detalle el contenido visual de la imagen. No inventes información. Si no puedes describirla, responde: 'No se puede describir'."
    )
    device = PIX2STRUCT_MODE
    inputs = pix2struct_processor(images=imagen_pix2, text=prompt_pix2struct, return_tensors="pt").to(device)
    generated_ids = pix2struct_model.generate(**inputs, max_new_tokens=800)
    descripcion_pix2struct_raw = pix2struct_processor.decode(generated_ids[0], skip_special_tokens=True).strip()
    advertencia = ""
    if descripcion_pix2struct_raw.lower().startswith(prompt_pix2struct[:40].lower()) or len(descripcion_pix2struct_raw) < 30:
        advertencia = "[ADVERTENCIA: La descripción de Pix2Struct puede ser irrelevante o vacía.]\n"
    descripcion_pix2struct = f"[Pix2Struct] {advertencia}{traducir_al_espanol(descripcion_pix2struct_raw)}"

    # --- LLaVA vía Ollama ---
    prompt_llava = (
        "La imagen es una captura de pantalla de un PDF. "
        "La imagen principal siempre se encuentra centrada. "
        "El texto superior es el título de la imagen. "
        "El texto inferior, que suele comenzar por 'Fuente:', es la fuente. "
        "Describe únicamente lo que ves en la imagen, sin suponer contexto externo. "
        "Indica si hay texto visible y transcríbelo literalmente. "
        "Si hay un gráfico, describe su tipo (barras, líneas, circular, etc.), ejes, leyenda y título solo si son claramente visibles. "
        "El titulo de la imagen se encuentra siempre en la parte superior de la imagen. "
        "No deduzcas ni inventes información. Si no puedes identificar el contenido, responde: 'No se puede describir'."
    )
    descripcion_llava = "[LLaVA] Error al conectar con Ollama."
    try:
        import requests
        import io
        buffered = io.BytesIO()
        # Usar la imagen original, sin reescalar para LLaVA
        imagen.save(buffered, format="PNG")
        img_b64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
        payload = {
            "model": "llava",
            "prompt": prompt_llava,
            "images": [img_b64],
            "stream": False
        }
        r = requests.post("http://localhost:11434/api/generate", json=payload, timeout=120)
        if r.status_code == 200:
            resp = r.json()
            desc = resp.get("response") or resp.get("description") or resp.get("result") or resp.get("text")
            if desc:
                descripcion_llava = "[LLaVA] " + traducir_al_espanol(desc.strip())
            else:
                descripcion_llava = "[LLaVA] Sin respuesta del modelo."
        else:
            descripcion_llava = f"[LLaVA] Error HTTP {r.status_code}"
    except Exception as e:
        descripcion_llava = f"[LLaVA] Error: {str(e)}"

    # --- Salida combinada ---
    return f"{descripcion_pix2struct}\n{descripcion_llava}\n{descripcion_ocr}"


def extraer_claves_visuales(respuesta):
    """
    Extrae claves como 'figura_1', 'tabla_2' de una respuesta textual para identificar elementos visuales.
    """
    claves_raw = re.findall(
        r'(imagen|figura|gr[aá]fico|tabla)[\s\-.:#nº]*\s*(\d{1,3}|[ivxlcdmIVXLCDM]{1,6})',
        respuesta, flags=re.IGNORECASE
    )
    claves = []
    for tipo, num in claves_raw:
        if not num.isdigit():
            num = romano_a_arabigo(num)
        claves.append(f"{tipo.lower()}_{num}")
    return claves

def es_probable_indice(texto):
    """
    Heurística para detectar si una página es un índice:
    - Al menos 5 líneas con muchos puntos y terminadas en número.
    - O bien, al menos 5 líneas (>=60%) que empiezan por numeración tipo 1., 1.1, 2.3.4, etc. y son largas (>30 caracteres).
    """
    texto = texto.lower()
    lineas = [l for l in texto.splitlines() if l.strip()]
    if len(lineas) < 5:
        return False
    # Heurística 1: líneas con muchos puntos y terminadas en número
    lineas_puntos_num = [l for l in lineas if re.search(r'\.{4,}\s*\d+$', l)]
    prop_puntos_num = len(lineas_puntos_num) / len(lineas) if lineas else 0
    long_media_puntos_num = np.mean([len(l) for l in lineas_puntos_num]) if lineas_puntos_num else 0
    if len(lineas_puntos_num) >= 5 and prop_puntos_num >= 0.6 and long_media_puntos_num > 30:
        return True
    # Heurística 2: líneas que empiezan por numeración tipo 1., 1.1, 2.3.4, etc. y son largas
    patron_numeracion = re.compile(r'^(\d+\.)+(\d+)?\s+')
    lineas_numeradas = [l for l in lineas if patron_numeracion.match(l) and len(l) > 30]
    prop_numeradas = len(lineas_numeradas) / len(lineas) if lineas else 0
    if len(lineas_numeradas) >= 5 and prop_numeradas >= 0.6:
        return True
    return False

def build_indice_visual(pdf_path):
    """
    Construye un índice de elementos visuales (figuras, tablas, etc.) por página a partir del texto del PDF.
    """
    indice = {}
    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages):
            texto = page.extract_text() or ""
            if es_probable_indice(texto):
                print(f"[IGNORADO] Página {i+1} detectada como índice.")
                continue
            matches_simple = re.findall(r'(imagen|figura|gr[aá]fico|tabla)[\s\-\.:#nº]*\s*(\d{1,3}|[ivxlcdmIVXLCDM]{1,6})', texto, flags=re.IGNORECASE)
            for tipo, num in matches_simple:
                num_norm = num if num.isdigit() else romano_a_arabigo(num)
                clave = f"{tipo.lower()}_{num_norm}"
                indice[clave] = i + 1
            matches_pag = re.findall(r'(imagen|figura|gr[aá]fico|tabla)\s+(\d+|[ivxlcdmIVXLCDM]{1,6})[^\n]{0,60}?\(en la página (\d+)\)', texto, flags=re.IGNORECASE)
            for tipo, num, pag in matches_pag:
                num_norm = num if num.isdigit() else romano_a_arabigo(num)
                clave = f"{tipo.lower()}_{num_norm}"
                indice[clave] = int(pag)
    return indice

def guardar_visuales_relevantes(pdf_path, paginas_objetivo, respuesta, output_dir=None):
    """
    Extrae y guarda imágenes relevantes de las páginas objetivo del PDF, recortando encabezados y pies si es posible.
    Aplica recorte horizontal automático y llama a describir_imagen para cada imagen extraída.
    Devuelve un diccionario con la ruta y descripción de cada imagen.
    """
    resultados = {}
    if output_dir is None:
        output_dir = input("Introduce el directorio donde guardar las imágenes extraídas:\n> ").strip()
    os.makedirs(output_dir, exist_ok=True)
    consulta_emb = embed_model.encode(respuesta, convert_to_tensor=True)
    # Extraer encabezado solicitado (tipo, número y parte del título si hay)
    encabezado_pat = re.search(r'(tabla|figura|imagen|gr[aá]fico)\s*(\d+|[ivxlcdmIVXLCDM]{1,6})(:|\.|\s+)([^\n]*)', respuesta, re.IGNORECASE)
    encabezado_solicitado = None
    if encabezado_pat:
        tipo, num, _, titulo = encabezado_pat.groups()
        num = num if num.isdigit() else romano_a_arabigo(num)
        encabezado_solicitado = {
            "tipo": tipo.lower(),
            "num": num,
            "titulo": titulo.strip().lower() if titulo else ""
        }
    with fitz.open(pdf_path) as doc, pdfplumber.open(pdf_path) as plumber:
        for i in paginas_objetivo:
            page = doc[i - 1]
            plumb_page = plumber.pages[i - 1]
            texto = plumb_page.extract_text() or ""
            if es_probable_indice(texto):
                print(f"[IGNORADO] Página {i} detectada como índice durante extracción.")
                continue
            words = plumb_page.extract_words()
            # Buscar todos los encabezados de figura en la página
            patron_linea = re.compile(rf"(imagen|figura|gr[aá]fico|tabla)[\s\-\.:#nº]*(\d{{1,3}}|[ivxlcdmIVXLCDM]{{1,6}})[\s\-\.:]*(.+)", re.IGNORECASE)
            lineas = texto.splitlines()
            candidatos = []
            for linea in lineas:
                m = patron_linea.match(linea.strip())
                if m:
                    tipo, num, titulo = m.groups()
                    num_norm = num if num.isdigit() else romano_a_arabigo(num)
                    candidatos.append({
                        "tipo": tipo.lower(),
                        "num": num_norm,
                        "titulo": titulo.strip().lower(),
                        "linea": linea.strip()
                    })
            # Elegir el encabezado más parecido al solicitado
            mejor = None
            mejor_score = -1
            if encabezado_solicitado and candidatos:
                for c in candidatos:
                    score = 0
                    if c["tipo"] == encabezado_solicitado["tipo"]:
                        score += 2
                    if c["num"] == encabezado_solicitado["num"]:
                        score += 3
                    # Coincidencia parcial de título (si hay)
                    if encabezado_solicitado["titulo"] and encabezado_solicitado["titulo"][:8] in c["titulo"]:
                        score += 2
                    if score > mejor_score:
                        mejor = c
                        mejor_score = score
            elif candidatos:
                mejor = candidatos[0]
            if mejor:
                # Buscar la posición Y del encabezado elegido
                tokens_linea = mejor["linea"].split()
                y_encabezado = None
                for idx in range(len(words) - len(tokens_linea) + 1):
                    bloque = [w["text"] for w in words[idx:idx+len(tokens_linea)]]
                    bloque_limpio = " ".join(bloque).lower().replace(",", "").replace(":", "").replace("-", "").replace(".", "").strip()
                    linea_limpia = mejor["linea"].lower().replace(",", "").replace(":", "").replace("-", "").replace(".", "").strip()
                    if bloque_limpio.startswith(linea_limpia[:20]) or linea_limpia.startswith(bloque_limpio[:20]):
                        y_encabezado = float(words[idx]["top"])
                        break
                if y_encabezado is None:
                    print(f"[WARN] Encabezado '{mejor['linea']}' no encontrado con la heurística optimizada.")
                # Buscar el pie de figura (ej. 'Fuente:') SOLO si está debajo del encabezado
                y_fuente = None
                if y_encabezado is not None:
                    for w in words:
                        if re.match(r"fuente:?", w["text"].lower()):
                            y_candidato = float(w["top"])
                            if y_candidato > y_encabezado:
                                y_fuente = y_candidato
                                break
                    if y_fuente is None:
                        altura_pagina = page.rect.height
                        if altura_pagina - y_encabezado > ALTURA_CLIP:
                            siguientes = [w for w in words if float(w["top"]) > y_encabezado + 20]
                            if siguientes:
                                y_fuente = float(siguientes[0]["top"])
                                print(f"[INFO] No se encontró 'Fuente', usando la primera línea de texto tras encabezado en y={y_fuente}")
                            else:
                                y_fuente = altura_pagina
                        else:
                            y_fuente = altura_pagina
                    else:
                        y_fuente = min(page.rect.height, y_fuente + 40)
                    if y_fuente - y_encabezado < 20:
                        print(f"[WARN] Recorte inválido: y_encabezado={y_encabezado}, y_fuente={y_fuente}. Se guardará la página completa.")
                        ruta = os.path.join(output_dir, f"pagina_{i}_captura.png")
                        page.get_pixmap(dpi=300).save(ruta)
                        # Recorte horizontal automático para el usuario
                        imagen = Image.open(ruta)
                        bbox = imagen.getbbox()
                        if bbox:
                            imagen = imagen.crop(bbox)
                            imagen.save(ruta)
                        clave = f"{i}_captura"
                    else:
                        ruta = os.path.join(output_dir, f"pagina_{i}_clip.png")
                        clip = fitz.Rect(0, y_encabezado, page.rect.width, y_fuente)
                        page.get_pixmap(dpi=300, clip=clip).save(ruta)
                        # Recorte horizontal automático para el usuario
                        imagen = Image.open(ruta)
                        bbox = imagen.getbbox()
                        if bbox:
                            imagen = imagen.crop(bbox)
                            imagen.save(ruta)
                        print(f"[INFO] Se guardó recorte ajustado de la página {i} desde y={y_encabezado:.2f} hasta y={y_fuente:.2f}")
                        clave = f"{i}_clip"
                    descripcion = describir_imagen(ruta)
                    resultados[clave] = (ruta, descripcion)
                else:
                    ruta = os.path.join(output_dir, f"pagina_{i}_captura.png")
                    page.get_pixmap(dpi=300).save(ruta)
                    # Recorte horizontal automático para el usuario
                    imagen = Image.open(ruta)
                    bbox = imagen.getbbox()
                    if bbox:
                        imagen = imagen.crop(bbox)
                        imagen.save(ruta)
                    print(f"[INFO] No se detectó encabezado en la página {i}, se guardó captura completa.")
                    clave = f"{i}_captura"
                    descripcion = describir_imagen(ruta)
                    resultados[clave] = (ruta, descripcion)
            else:
                ruta = os.path.join(output_dir, f"pagina_{i}_captura.png")
                page.get_pixmap(dpi=300).save(ruta)
                # Recorte horizontal automático para el usuario
                imagen = Image.open(ruta)
                bbox = imagen.getbbox()
                if bbox:
                    imagen = imagen.crop(bbox)
                    imagen.save(ruta)
                print(f"[INFO] No se detectó encabezado en la página {i}, se guardó captura completa.")
                clave = f"{i}_captura"
                descripcion = describir_imagen(ruta)
                resultados[clave] = (ruta, descripcion)
    return resultados

def query_mistral(prompt):
    """
    Llama a un modelo Mistral local vía API para responder preguntas sobre fragmentos del PDF.
    """
    try:
        r = requests.post("http://localhost:11434/api/generate", json={
            "model": "mistral",
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0.0}
        })
        return r.json().get("response", "No encontrado en el documento").strip()
    except Exception as e:
        return f"[ERROR] {str(e)}"
    except Exception as e:
        return f"[ERROR] {str(e)}"

def consulta_pide_visual(query):
    """
    Siempre devuelve True: ahora se analizan imágenes para cualquier consulta.
    """
    return True

def main():
    """
    Flujo principal del programa:
    1. Pide la ruta al PDF.
    2. Pide el directorio de salida UNA SOLA VEZ.
    3. Extrae texto de cada página, ignorando índices.
    4. Genera embeddings y construye un índice semántico.
    5. En bucle, recibe consultas del usuario, busca fragmentos relevantes y pregunta a Mistral.
    6. Si la respuesta menciona una figura/tabla, extrae y describe la imagen correspondiente.
    """
    pdf_path = input("Ruta al PDF:\n> ").strip()
    output_dir = input("Introduce el directorio donde guardar las imágenes extraídas:\n> ").strip()
    chunks = []
    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages):
            texto = page.extract_text()
            if not texto:
                continue
            if es_probable_indice(texto):
                print(f"[IGNORADO] Página {i+1} detectada como índice en procesamiento inicial.")
                continue
            splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=300)
            for chunk in splitter.split_text(texto):
                chunks.append({"text": chunk, "page": i + 1})
    textos = [c["text"] for c in chunks]
    embeddings = embed_model.encode(textos, convert_to_numpy=True, show_progress_bar=True)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    indice = build_indice_visual(pdf_path)
    while True:
        query = input("\nConsulta ('salir' para terminar):\n>> ").strip()
        if query.lower() in ["salir", "exit", "quit"]:
            break
        query_embedding = embed_model.encode([f"query: {query}"], convert_to_numpy=True)
        distances, indices_r = index.search(query_embedding, 8)
        top_chunks = [chunks[i] for i in indices_r[0]]
        contexto = "\n\n".join(f"(Página {c['page']})\n{c['text']}" for c in top_chunks)
        prompt = (
            "Eres un experto en comprensión documental. Tu única fuente de información son los fragmentos extraídos de un documento PDF. "
            "No puedes inventar, deducir ni interpolar datos. Tu tarea es identificar si la información solicitada aparece en una tabla, figura, imagen o gráfico. "
            "Si es así, debes citar exactamente el nombre que le da el documento (como 'Figura 1', 'Tabla 4', etc.) y en qué página aparece. "
            "Si no puedes identificar una tabla o figura explícita con esa información, responde exactamente: 'No encontrado en el documento'. "
            "No debes generar tú mismo una tabla o resumen de datos. No debes deducir valores ni convertir texto en formato tabular. "
            "No des explicaciones o reformules datos si no están explícitamente presentes como elemento visual en el documento. "
            f"\n\n=== Fragmentos extraídos ===\n{contexto}\n\n=== Pregunta del usuario ===\n{query}\n\n=== Respuesta experta ==="
        )
        respuesta = query_mistral(prompt)
        print(f"\n[RESPUESTA]: {respuesta}")
        # Solo extraer y analizar imágenes si la consulta lo pide explícitamente
        if not consulta_pide_visual(query):
            print("[INFO] La consulta no solicita explícitamente una imagen, gráfico, figura o tabla. No se extraerán imágenes.")
            continue
        if not re.search(r'(tabla|figura|imagen|gr[áa]fico)\s+(\d+|[ivxlcdm]{1,6})', respuesta.lower()):
            print("[INFO] La respuesta no menciona ninguna figura o tabla concreta. Se descarta.")
            continue
        claves = extraer_claves_visuales(respuesta)
        # Buscar página explícita en la respuesta
        paginas_explicitas = re.findall(r'en la página\s+(\d+)', respuesta.lower())
        paginas = []
        fallback_paginas = []
        if paginas_explicitas:
            paginas = sorted(set([int(p) for p in paginas_explicitas]))
            # Fallback: si hay claves y alguna clave está en el índice pero NO en la lista de páginas explícitas, la guardamos para fallback
            for k in claves:
                if k in indice and indice[k] not in paginas:
                    fallback_paginas.append(indice[k])
        else:
            paginas = sorted(set(indice[k] for k in claves if k in indice))
        resultados = guardar_visuales_relevantes(pdf_path, paginas, respuesta, output_dir=output_dir)
        # Fallback: si en todas las páginas principales no se detectó encabezado, probar en las de fallback
        encabezado_detectado = any("clip" in clave for clave in resultados)
        if not encabezado_detectado and fallback_paginas:
            print(f"[INFO] No se detectó encabezado en las páginas principales. Probando en páginas alternativas del índice: {fallback_paginas}")
            resultados_fallback = guardar_visuales_relevantes(pdf_path, fallback_paginas, respuesta, output_dir=output_dir)
            # Solo mostrar si se detecta encabezado (clip)
            for clave, (ruta, desc) in resultados_fallback.items():
                if "clip" in clave:
                    print(f"[IMAGEN][FALLBACK]: {ruta}\n[DESCRIPCIÓN]: {desc}\n")
            # Si tampoco hay clip, mostrar los fallback como captura
            if not any("clip" in clave for clave in resultados_fallback):
                for clave, (ruta, desc) in resultados_fallback.items():
                    print(f"[IMAGEN][FALLBACK]: {ruta}\n[DESCRIPCIÓN]: {desc}\n")
        for clave, (ruta, desc) in resultados.items():
            print(f"[IMAGEN]: {ruta}\n[DESCRIPCIÓN]: {desc}\n")

if __name__ == "__main__":
    main()
