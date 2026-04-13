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

# ─── Funciones de Análisis ────────────────────────────────────

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

# ─── Configuración de página ──────────────────────────────────
st.set_page_config(
    page_title="Análisis de Sentimientos",
    page_icon="🧠",
    layout="centered"
)

# ─── Estilos CSS (CORREGIDOS PARA VISIBILIDAD) ────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;700;800&family=DM+Sans:wght@300;400;500&display=swap');

html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }

.stApp {
    background: #0d0d12;
    color: #e8e8f0;
}

/* Header principal */
.main-header { text-align: center; padding: 2.5rem 0 1.5rem 0; }
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
.main-header p { color: #6b7280; font-size: 1rem; font-weight: 300; }

/* CORRECCIÓN: CUADRO DE TEXTO VISIBLE */
.stTextArea textarea {
    background-color: #1a1a21 !important; /* Fondo oscuro */
    color: #ffffff !important;           /* Letras blancas */
    border: 1px solid rgba(255,255,255,0.15) !important;
    border-radius: 14px !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 1rem !important;
    line-height: 1.6 !important;
}

.stTextArea textarea::placeholder {
    color: rgba(255,255,255,0.3) !important;
}

/* Botón */
.stButton > button {
    background: linear-gradient(135deg, #7c3aed, #4f46e5) !important;
    color: white !important;
    border: none !important;
    border-radius: 50px !important;
    padding: 0.65rem 2.5rem !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 700 !important;
    transition: all 0.2s ease !important;
    width: 100%;
}
.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 25px rgba(124,58,237,0.5) !important;
}

/* Otros estilos de UI */
.result-card { border-radius: 18px; padding: 2rem; margin: 1.5rem 0; border: 1px solid rgba(255,255,255,0.08); background: rgba(255,255,255,0.04); backdrop-filter: blur(10px); }
.sentiment-badge { display: inline-block; font-family: 'Syne', sans-serif; font-size: 1.8rem; font-weight: 800; padding: 0.5rem 1.8rem; border-radius: 50px; margin-bottom: 1rem; color: #fff !important; }
.badge-positivo { background: linear-gradient(135deg, #059669, #34d399); }
.badge-negativo { background: linear-gradient(135deg, #dc2626, #f87171); }
.badge-neutro   { background: linear-gradient(135deg, #4b5563, #9ca3af); }

.metric-box { flex: 1; background: rgba(255,255,255,0.06); border-radius: 12px; padding: 1rem; border: 1px solid rgba(255,255,255,0.06); text-align: center; }
.metric-box .label { font-size: 0.7rem; text-transform: uppercase; color: #6b7280; margin-bottom: 0.3rem; }
.metric-box .value { font-family: 'Syne', sans-serif; font-size: 1.6rem; font-weight: 700; }
</style>
""", unsafe_allow_html=True)

# ─── Lógica de Procesamiento ──────────────────────────────────

def analizar_sentimiento_oracion(oracion: str) -> dict:
    pol, sub = None, None
    metodo = "lexicón"
    try:
        from deep_translator import GoogleTranslator
        texto_en = GoogleTranslator(source='es', target='en').translate(oracion)
        if texto_en:
            blob = TextBlob(texto_en)
            pol_tb, sub_tb = blob.sentiment.polarity, blob.sentiment.subjectivity
            if pol_tb != 0.0 or sub_tb != 0.0:
                pol, sub, metodo = pol_tb, sub_tb, "traducción"
    except: pass

    if pol is None or pol == 0.0:
        pol_lex, sub_lex = analizar_con_lexico(oracion)
        pol, sub, metodo = pol_lex, sub_lex, "lexicón"

    pol = pol if pol is not None else 0.0
    sub = sub if sub is not None else 0.0
    
    if pol > 0.1: et, em, cl = "Positivo", "🟢", "positivo"
    elif pol < -0.1: et, em, cl = "Negativo", "🔴", "negativo"
    else: et, em, cl = "Neutro", "⚪", "neutro"

    return {"polaridad": pol, "subjetividad": sub, "etiqueta": et, "emoji": em, "css": cl, "metodo": metodo}

def gauge_chart(polaridad: float):
    fig, ax = plt.subplots(figsize=(5, 2.5), subplot_kw={'aspect': 'equal'})
    fig.patch.set_facecolor('#0d0d12')
    ax.set_facecolor('#0d0d12')
    theta = np.linspace(np.pi, 0, 300)
    ax.plot(np.cos(theta), np.sin(theta), color='#1f2937', linewidth=15, solid_capstyle='round')
    color = '#34d399' if polaridad > 0.1 else '#f87171' if polaridad < -0.1 else '#9ca3af'
    angulo = np.pi - (polaridad + 1) / 2 * np.pi
    ax.plot(np.cos(np.linspace(np.pi, angulo, 100)), np.sin(np.linspace(np.pi, angulo, 100)), color=color, linewidth=15, solid_capstyle='round')
    ax.text(0, 0.2, f'{polaridad:+.2f}', color='white', fontsize=18, ha='center', fontweight='bold')
    ax.axis('off')
    return fig

# ─── Interfaz Principal ───────────────────────────────────────

st.markdown("""
<div class="main-header">
    <h1>🧠 Análisis de Sentimientos</h1>
    <p>Detecta la polaridad emocional de textos en español · NLP con NLTK & TextBlob</p>
</div>
""", unsafe_allow_html=True)

EJEMPLOS = {
    "😊 Positivo": "La inteligencia artificial es una tecnología maravillosa.",
    "😞 Negativo": "Este sistema tiene errores graves y un rendimiento terrible.",
    "😐 Neutro": "El modelo procesa los datos de entrada de forma binaria.",
}

st.markdown("**Ejemplos rápidos:**")
cols = st.columns(len(EJEMPLOS))
texto_ejemplo = ""
for i, (label, txt) in enumerate(EJEMPLOS.items()):
    if cols[i].button(label, key=f"ej_{i}"):
        texto_ejemplo = txt

texto_usuario = st.text_area(
    "Escribe o pega tu texto en español:",
    value=texto_ejemplo,
    height=150,
    placeholder="Escribe aquí tu texto para analizar sus sentimientos (Ej: Hoy es un gran día...)",
    label_visibility="collapsed"
)

st.markdown("<br>", unsafe_allow_html=True)
if st.button("✨  Analizar sentimiento"):
    if texto_usuario.strip():
        with st.spinner("Analizando..."):
            res = analizar_sentimiento_oracion(texto_usuario)
            oraciones = sent_tokenize(texto_usuario, language='spanish')
            
        st.markdown(f"""
        <div class="result-card">
            <div style="text-align:center; margin-bottom:1rem;">
                <span class="sentiment-badge badge-{res['css']}">{res['emoji']} {res['etiqueta']}</span>
            </div>
            <div style="display:flex; gap:10px;">
                <div class="metric-box">
                    <div class="label">Polaridad</div>
                    <div class="value" style="color:{'#34d399' if res['polaridad']>0.1 else '#f87171' if res['polaridad']<-0.1 else '#9ca3af'}">{res['polaridad']:+.3f}</div>
                </div>
                <div class="metric-box">
                    <div class="label">Subjetividad</div>
                    <div class="value" style="color:#60a5fa">{res['subjetividad']:.3f}</div>
                </div>
                <div class="metric-box">
                    <div class="label">Oraciones</div>
                    <div class="value" style="color:#a78bfa">{len(oraciones)}</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        st.pyplot(gauge_chart(res['polaridad']), use_container_width=True)
    else:
        st.warning("⚠️ Escribe algo antes de analizar.")

st.markdown("<br><br><div style='text-align:center;color:#4b5563;font-size:0.8rem;'>Análisis de Sentimientos · NLTK + TextBlob</div>", unsafe_allow_html=True)
