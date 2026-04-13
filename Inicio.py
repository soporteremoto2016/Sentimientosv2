import streamlit as st
from textblob import TextBlob
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.probability import FreqDist
import nltk
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
import pandas as pd
import numpy as np

# ─── Lexicón de sentimientos en español ───────────────────────
PALABRAS_POSITIVAS = {
    "bueno", "buena", "buenos", "buenas", "excelente", "excelentes",
    "maravilloso", "maravillosa", "increíble", "increíbles", "fantástico",
    "fantástica", "genial", "positivo", "positiva", "feliz", "alegre",
    "contento", "contenta", "satisfecho", "satisfecha", "perfecto", "perfecta",
    "mejor", "mejora", "mejorado", "mejorar", "prometedor", "prometedora",
    "exitoso", "exitosa", "éxito", "logro", "logros", "avance", "avances",
    "innovación", "innovador", "innovadora", "potencial", "oportunidad",
    "oportunidades", "beneficio", "beneficios", "útil", "útiles", "eficiente",
    "eficientes", "efectivo", "efectiva", "poderoso", "poderosa", "capaz",
    "capaces", "brillante", "brillantes", "sobresaliente", "notable",
    "admirable", "valioso", "valiosa", "importante", "fascinante", "fascinantes",
    "interesante", "interesantes", "fácil", "rápido", "rápida", "preciso",
    "precisa", "seguro", "segura", "confiable", "confiables", "robusto",
    "robusta", "gran", "grande", "grandes", "mejor", "óptimo", "óptima",
    "gusta", "encanta", "amo", "amas", "aman", "recomienda", "recomendamos",
    "amor", "pasión", "entusiasmo", "esperanza", "progreso", "crecimiento",
    "transformación", "mejorar", "superar", "ganar", "triunfar", "destacar",
    "liderar", "empoderar", "inspirar", "grandioso", "maravilla", "maravillosa", "maravilloso"
}

PALABRAS_NEGATIVAS = {
    "malo", "mala", "malos", "malas", "terrible", "terribles", "horrible",
    "horribles", "pésimo", "pésima", "peor", "deficiente", "deficientes",
    "problema", "problemas", "error", "errores", "falla", "fallas", "falló",
    "fracaso", "fracasos", "fracasó", "grave", "graves", "negativo", "negativa",
    "difícil", "difíciles", "imposible", "imposibles", "complicado", "complicada",
    "preocupante", "preocupantes", "peligroso", "peligrosa", "riesgo", "riesgos",
    "desafío", "desafíos", "obstáculo", "obstáculos", "limitación", "limitaciones",
    "lento", "lenta", "ineficiente", "ineficientes", "inútil", "inútiles",
    "obsoleto", "obsoleta", "anticuado", "anticuada", "desactualizado",
    "frustrante", "frustrantes", "decepcionante", "decepcionantes",
    "insatisfecho", "insatisfecha", "insuficiente", "insuficientes",
    "inadecuado", "inadecuada", "incorrecto", "incorrecta", "equivocado",
    "equivocada", "complejo", "compleja", "confuso", "confusa", "molesto",
    "molesta", "triste", "tristeza", "miedo", "temor", "angustia", "ansiedad",
    "desastre", "catástrofe", "crisis", "deterioro", "declive", "caída",
    "pérdida", "perdida", "daño", "daños", "aburrido", "aburrida", "mediocre"
}

INTENSIFICADORES = {"muy", "bastante", "extremadamente", "totalmente", "completamente",
                    "absolutamente", "increíblemente", "enormemente", "sumamente"}
NEGACIONES = {"no", "nunca", "jamás", "tampoco", "ni", "ningún", "ninguna", "sin"}


def analizar_con_lexico(texto: str) -> dict:
    tokens = texto.lower().split()
    score = 0.0
    total_sentiment_words = 0
    negacion_activa = False

    for i, token in enumerate(tokens):
        token_limpio = ''.join(c for c in token if c.isalpha() or c == 'é' or c == 'á'
                               or c == 'í' or c == 'ó' or c == 'ú' or c == 'ü' or c == 'ñ')

        if token_limpio in NEGACIONES:
            negacion_activa = True
            continue

        intensificador = 1.0
        if i > 0:
            prev = ''.join(c for c in tokens[i-1] if c.isalpha())
            if prev in INTENSIFICADORES:
                intensificador = 1.5

        if token_limpio in PALABRAS_POSITIVAS:
            delta = 1.0 * intensificador
            score += -delta if negacion_activa else delta
            total_sentiment_words += 1
            negacion_activa = False
        elif token_limpio in PALABRAS_NEGATIVAS:
            delta = -1.0 * intensificador
            score += -delta if negacion_activa else delta
            total_sentiment_words += 1
            negacion_activa = False
        else:
            negacion_activa = False

    if total_sentiment_words > 0:
        polaridad = max(-1.0, min(1.0, score / (total_sentiment_words * 1.5)))
    else:
        polaridad = 0.0

    num_palabras = len([t for t in tokens if len(t) > 2])
    subjetividad = min(1.0, total_sentiment_words / max(num_palabras, 1) * 3)

    return round(polaridad, 4), round(subjetividad, 4)

@st.cache_resource
def descargar_nltk():
    for recurso in ['punkt', 'punkt_tab', 'stopwords']:
        try:
            nltk.download(recurso, quiet=True)
        except:
            pass

descargar_nltk()

st.set_page_config(
    page_title="Análisis de Sentimientos",
    page_icon="🧠",
    layout="centered"
)

# ─── Estilos CSS ──────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;700;800&family=DM+Sans:wght@300;400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

.stApp {
    background: #0d0d12;
    color: #e8e8f0;
}

.main-header {
    text-align: center;
    padding: 2.5rem 0 1.5rem 0;
}
.main-header h1 {
    font-family: 'Syne', sans-serif;
    font-weight: 800;
    font-size: 2.6rem;
    letter-spacing: -1px;
    background: linear-gradient(135deg, #a78bfa, #60a5fa, #34d399);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0.4rem;
}
.main-header p {
    color: #6b7280;
    font-size: 1rem;
    font-weight: 300;
}

.result-card {
    border-radius: 18px;
    padding: 2rem;
    margin: 1.5rem 0;
    border: 1px solid rgba(255,255,255,0.08);
    background: rgba(255,255,255,0.04);
    backdrop-filter: blur(10px);
}

.sentiment-badge {
    display: inline-block;
    font-family: 'Syne', sans-serif;
    font-size: 1.8rem;
    font-weight: 800;
    padding: 0.5rem 1.8rem;
    border-radius: 50px;
    margin-bottom: 1rem;
    letter-spacing: -0.5px;
}
.badge-positivo { background: linear-gradient(135deg, #059669, #34d399); color: #fff; }
.badge-negativo { background: linear-gradient(135deg, #dc2626, #f87171); color: #fff; }
.badge-neutro   { background: linear-gradient(135deg, #4b5563, #9ca3af); color: #fff; }

.metric-row {
    display: flex;
    gap: 1rem;
    margin-top: 1rem;
}
.metric-box {
    flex: 1;
    background: rgba(255,255,255,0.06);
    border-radius: 12px;
    padding: 1rem 1.2rem;
    border: 1px solid rgba(255,255,255,0.06);
    text-align: center;
}
.metric-box .label {
    font-size: 0.72rem;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    color: #6b7280;
    margin-bottom: 0.3rem;
}
.metric-box .value {
    font-family: 'Syne', sans-serif;
    font-size: 1.7rem;
    font-weight: 700;
}

.oracion-row {
    display: flex;
    align-items: flex-start;
    gap: 0.8rem;
    padding: 0.75rem 1rem;
    border-radius: 10px;
    margin-bottom: 0.5rem;
    background: rgba(255,255,255,0.03);
    border-left: 3px solid transparent;
}
.oracion-positivo { border-color: #34d399; }
.oracion-negativo { border-color: #f87171; }
.oracion-neutro   { border-color: #6b7280; }

/* Textarea - AQUÍ SE CAMBIÓ EL COLOR A NEGRO */
.stTextArea textarea {
    background: rgba(255,255,255,0.9) !important; /* Fondo más claro para que se vea el texto negro */
    border: 1px solid rgba(255,255,255,0.12) !important;
    border-radius: 14px !important;
    color: #000000 !important; /* Texto en color negro */
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.95rem !important;
    resize: vertical;
}

.stButton > button {
    background: linear-gradient(135deg, #7c3aed, #4f46e5) !important;
    color: white !important;
    border: none !important;
    border-radius: 50px !important;
    padding: 0.65rem 2.5rem !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 700 !important;
    font-size: 1rem !important;
    letter-spacing: 0.5px !important;
    transition: all 0.2s ease !important;
    width: 100%;
}
.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 25px rgba(124,58,237,0.5) !important;
}

.ejemplo-chip {
    display: inline-block;
    background: rgba(255,255,255,0.06);
    border: 1px solid rgba(255,255,255,0.1);
    border-radius: 20px;
    padding: 0.3rem 0.9rem;
    font-size: 0.82rem;
    cursor: pointer;
    margin: 0.2rem;
    color: #9ca3af;
}

hr { border-color: rgba(255,255,255,0.07) !important; }

.streamlit-expanderHeader {
    font-family: 'Syne', sans-serif;
    color: #a78bfa !important;
}
</style>
""", unsafe_allow_html=True)


# ─── Funciones ────────────────────────────────────────────────

def analizar_sentimiento_oracion(oracion: str) -> dict:
    pol, sub = None, None
    metodo = "lexicón"

    try:
        from deep_translator import GoogleTranslator
        texto_en = GoogleTranslator(source='es', target='en').translate(oracion)
        if texto_en:
            blob = TextBlob(texto_en)
            pol_tb = blob.sentiment.polarity
            sub_tb = blob.sentiment.subjectivity
            if pol_tb != 0.0 or sub_tb != 0.0:
                pol, sub = pol_tb, sub_tb
                metodo = "traducción"
    except Exception:
        pass

    if pol is None:
        try:
            blob_es = TextBlob(oracion)
            pol_es = blob_es.sentiment.polarity
            sub_es = blob_es.sentiment.subjectivity
            if pol_es != 0.0:
                pol, sub = pol_es, sub_es
                metodo = "textblob-es"
        except Exception:
            pass

    if pol is None or pol == 0.0:
        pol_lex, sub_lex = analizar_con_lexico(oracion)
        if pol_lex != 0.0:
            if pol is not None and pol != 0.0:
                pol = (pol + pol_lex) / 2
                sub = (sub + sub_lex) / 2
            else:
                pol, sub = pol_lex, sub_lex
                metodo = "lexicón"

  pol = pol if pol is not None else 0.0
    sub = sub if sub is not None else 0.0

    if pol > 0.1:
        etiqueta, emoji, css_clase = "Positivo", "🟢", "positivo"
    elif pol < -0.1:
        etiqueta, emoji, css_clase = "Negativo", "🔴", "negativo"
    else:
        etiqueta, emoji, css_clase = "Neutro", "⚪", "neutro"

    # AQUÍ ESTÁ EL DICCIONARIO QUE CAUSABA EL ERROR
    return {
        "polaridad": round(pol, 4),
        "subjetividad": round(sub, 4),
        "etiqueta": etiqueta,
        "emoji": emoji,
        "css": css_clase,
        "metodo": metodo
    }
