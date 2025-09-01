# IA-Investigacion
Herramienta en Python para el análisis de PDFs con NLP y visión por computador. Permite extraer texto, identificar tablas y gráficos y generar descripciones automáticas. Proyecto en fase beta, la descripción de imágenes aún no es del todo fiable.


Este repositorio contiene un script en Python que permite realizar consultas sobre documentos PDF y obtener respuestas basadas en texto y, cuando es posible, en elementos visuales como tablas y gráficos.

Características

-Extracción de texto con pdfplumber y OCR (Tesseract).

-Indexación semántica con Sentence Transformers y FAISS.

-Traducción automática con MarianMT.

-Procesamiento visual con Pix2Struct y LLaVA vía Ollama.

-Detección de índices para evitar ruido en las búsquedas.

-Generación de descripciones de imágenes (en desarrollo).

Aviso: El proyecto está en fase beta. La descripción de imágenes puede ser incompleta o incorrecta.
