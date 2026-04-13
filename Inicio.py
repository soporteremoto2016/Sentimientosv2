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
    "liderar", "empoderar", "inspirar", "grandioso", "maravilla"
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
        token_limpio = ''.join(c for c in token if c.isalpha() or c in 'éáíóúüñ')

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

# ─── Estilos CSS Personalizados ────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;700;800&family=DM+Sans:wght@300;400;500&display=swap');

html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }

.stApp { background: #0d0d12; color: #e8e8f0; }

.main-header { text-align: center; padding: 2.5rem 0 1.5rem 0; }
.main-header h1 {
    font-family: 'Syne', sans-serif;
    font-weight: 800;
    font-size: 2.6rem;
    background: linear-gradient(135deg, #a78bfa, #60a5fa, #34d399);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

/* Ajuste de colores en Badges */
.sentiment-badge {
    display: inline-block;
    font-family: 'Syne', sans-serif;
    font-size: 1.8rem;
    font-weight: 800;
    padding: 0.5rem 1.8rem;
    border-radius: 50px;
    margin-bottom: 1rem;
    color: #ffffff !important; /* Texto siempre blanco */
    box-shadow: 0 4px 15px rgba(0,0,0,0.3);
}
.badge-positivo { background: linear-gradient(135deg, #059669, #34d399); }
.badge-negativo { background: linear-gradient(135deg, #dc2626, #f87171); }
.badge-neutro   { background: linear-gradient(135deg, #4b5563, #9ca3af); }

.result-card {
    border-radius: 18px;
    padding: 2rem;
    margin: 1.5rem 0;
    border: 1px solid rgba(255,255,255,0.08);
    background: rgba(255,255,255,0.04);
}

.metric-box {
    background: rgba(255,255,255,0.06);
    border-radius: 12px;
    padding: 1rem;
    text-align: center;
}
.metric-box .value { font-family: 'Syne', sans-serif; font-size: 1.7rem; font-weight: 700; }

.stButton > button {
    background: linear-gradient(135deg, #7c3aed, #4f46e5) !important;
    color: white !important;
    border-radius: 50px !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 700 !important;
}
</style>
""", unsafe_allow_html=True)

def analizar_sentimiento_oracion(oracion: str) -> dict:
    pol, sub = None, None
    metodo = "lexicón"

    # Capa 1: Traducción
    try:
        from deep_translator import GoogleTranslator
        texto_en = GoogleTranslator(source='es', target='en').translate(oracion)
        if texto_en:
            blob = TextBlob(texto_en)
            if blob.sentiment.polarity != 0.0 or blob.sentiment.subjectivity != 0.0:
                pol, sub = blob.sentiment.polarity, blob.sentiment.subjectivity
                metodo = "traducción"
    except: pass

    # Capa 2: Lexicón (Siempre disponible)
    if pol is None or pol == 0.0:
        pol_lex, sub_lex = analizar_con_lexico(oracion)
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

    return {
        "polaridad": round(pol, 4),
        "subjetividad": round(sub, 4),
        "etiqueta": etiqueta,
        "emoji": emoji,
        "css": css_clase,
        "metodo": metodo
    }

# El resto de tus funciones de visualización (gauge_chart, barras_oraciones) 
# y la sección de INTERFAZ se mantienen igual, el CSS ya se encargará de los cambios de color.

# ─── INTERFAZ (Continuación) ──────────────────────────────────
st.markdown("""
<div class="main-header">
    <h1>🧠 Análisis de Sentimientos</h1>
    <p>Detecta la polaridad emocional de textos en español</p>
</div>
""", unsafe_allow_html=True)

texto_usuario = st.text_area("Escribe tu texto:", height=150, label_visibility="collapsed")
analizar = st.button("✨ Analizar sentimiento")

if analizar and texto_usuario.strip():
    res = analizar_sentimiento_oracion(texto_usuario)
    
    st.markdown(f"""
    <div class="result-card">
        <div style="text-align:center;">
            <span class="sentiment-badge badge-{res['css']}">{res['emoji']} {res['etiqueta']}</span>
        </div>
        <div class="metric-row" style="display: flex; gap: 10px; margin-top: 20px;">
            <div class="metric-box" style="flex:1;">
                <div style="color: #6b7280; font-size: 0.7rem;">POLARIDAD</div>
                <div class="value" style="color:{'#34d399' if res['polaridad']>0.1 else '#f87171' if res['polaridad']<-0.1 else '#9ca3af'}">
                    {res['polaridad']:+.3f}
                </div>
            </div>
            <div class="metric-box" style="flex:1;">
                <div style="color: #6b7280; font-size: 0.7rem;">SUBJETIVIDAD</div>
                <div class="value" style="color:#60a5fa">{res['subjetividad']:.3f}</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
