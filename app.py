import os
import re
import json
import base64
from io import BytesIO
from datetime import datetime
from typing import Optional, List, Dict, Tuple

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
from dotenv import load_dotenv
from openai import OpenAI
from fpdf import FPDF

from main import (
    find_column,
    classify_ads,
    clean_dutch_number,
    AD_NAME_COLUMNS,
    CAMPAIGN_COLUMNS,
    TONE_INSTRUCTION,
)

load_dotenv()

CATEGORY_NL = {
    "High Performer": "Top Presteerder",
    "Average":        "Gemiddeld",
    "Underperformer": "Onderpresteerder",
    "No Data":        "Geen Data",
}

PRIORITY_COLOURS = {
    # (background, text, label)
    "Hoog":   ("#e63946", "#fff", "Must-test"),
    "Middel": ("#1d6fa5", "#fff", "Strong contender"),
    "Laag":   ("#6c757d", "#fff", "Experimental"),
}

colours_map = {
    "High Performer": "background-color: #d4edda; color: #155724",
    "Average":        "background-color: #fff3cd; color: #856404",
    "Underperformer": "background-color: #f8d7da; color: #721c24",
    "No Data":        "background-color: #e2e3e5; color: #383d41",
}

SPEND_COLUMNS = [
    "Besteed bedrag (EUR)", "Amount spent (EUR)", "Besteed bedrag",
    "Amount spent", "Kosten", "Spend",
]

ROAS_COLUMNS_KPI = [
    "ROAS (rendement op advertentie-uitgaven) voor aankoop",
    "Purchase ROAS (return on ad spend)", "ROAS",
    "Rendement op advertentie-uitgaven",
]

MATRIX_GROUPS = {
    "Hoog":   {"label": "Directe Actie",  "bg": "#00573C", "color": "#ffffff"},
    "Middel": {"label": "Testen",          "bg": "#33B784", "color": "#ffffff"},
    "Laag":   {"label": "Lange Termijn",   "bg": "#96D247", "color": "#1a3a2a"},
}

# ---------------------------------------------------------------------------
# Rocket launch animation (Phase 2)
# ---------------------------------------------------------------------------

_ROCKET_ANIM_HTML = """
<style>
@keyframes _rocketRise   { 0%{transform:translateY(80px) scale(0.85);opacity:.6} 100%{transform:translateY(0) scale(1);opacity:1} }
@keyframes _moonPulse    { 0%,100%{text-shadow:0 0 18px rgba(255,220,80,.5)} 50%{text-shadow:0 0 38px rgba(255,220,80,.95)} }
@keyframes _starBlink    { 0%,100%{opacity:1} 50%{opacity:.15} }
@keyframes _trailFade    { 0%{opacity:.8} 100%{opacity:0;transform:scaleY(1.6)} }
.ok-space-bg {
    background:linear-gradient(180deg,#000814 0%,#001d3d 55%,#003566 100%);
    border-radius:18px; padding:52px 32px 44px; text-align:center;
    min-height:310px; position:relative; overflow:hidden;
}
.ok-stars-row { font-size:.8rem; letter-spacing:14px; animation:_starBlink 2.2s ease-in-out infinite; color:#e8e8e8; display:block; margin-bottom:10px; }
.ok-moon-icon { font-size:3.2rem; animation:_moonPulse 2.6s ease-in-out infinite; display:block; margin-bottom:6px; }
.ok-rocket-wrap { display:inline-block; animation:_rocketRise 1s cubic-bezier(.22,1,.36,1) forwards; }
.ok-rocket-icon { font-size:4rem; display:block; }
.ok-launch-title { color:#33B784; font-size:1.45rem; font-weight:800; font-family:Rubik,sans-serif; margin-top:20px; margin-bottom:6px; }
.ok-launch-sub   { color:#adb5bd; font-size:.88rem; font-family:Rubik,sans-serif; }
</style>
<div class="ok-space-bg">
  <span class="ok-stars-row">✦ ✦ ✦ ✦ ✦ ✦ ✦ ✦ ✦</span>
  <span class="ok-moon-icon">🌕</span>
  <div class="ok-rocket-wrap"><span class="ok-rocket-icon">🚀</span></div>
  <div class="ok-launch-title">AI-analyse is gestart!</div>
  <div class="ok-launch-sub">De campagnedata wordt geanalyseerd — even geduld a.u.b.</div>
</div>
"""


def fmt_nl(value: float, decimals: int = 2, prefix: str = "") -> str:
    """
    Format a float in Dutch locale: period as thousands separator, comma as decimal.
    e.g. 1234.56 → '1.234,56'   |   2.67 → '2,67'
    """
    formatted = f"{value:,.{decimals}f}"           # '1,234.56' (English)
    formatted = formatted.replace(",", "X").replace(".", ",").replace("X", ".")  # Dutch
    return f"{prefix}{formatted}"


# ---------------------------------------------------------------------------
# Pagina-instellingen
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Online Klik | Meta Ads Analyser",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Brand CSS + Rubik font
# ---------------------------------------------------------------------------

st.markdown(
    """
    <link href="https://fonts.googleapis.com/css2?family=Rubik:wght@400;500;600;700;800&display=swap" rel="stylesheet">
    <style>
        /* Global font */
        html, body, [class*="css"] {
            font-family: 'Rubik', sans-serif !important;
        }

        /* Main header */
        h1, h2, h3, h4 {
            font-family: 'Rubik', sans-serif !important;
        }

        /* Primary buttons → Actiegroen */
        .stButton > button[kind="primary"],
        button[data-testid="baseButton-primary"] {
            background-color: #33B784 !important;
            border-color: #33B784 !important;
            color: #ffffff !important;
            font-family: 'Rubik', sans-serif !important;
            font-weight: 600 !important;
            border-radius: 8px !important;
        }
        .stButton > button[kind="primary"]:hover,
        button[data-testid="baseButton-primary"]:hover {
            background-color: #00573C !important;
            border-color: #00573C !important;
        }

        /* Download buttons */
        .stDownloadButton > button {
            background-color: #00573C !important;
            border-color: #00573C !important;
            color: #ffffff !important;
            font-family: 'Rubik', sans-serif !important;
            font-weight: 500 !important;
            border-radius: 8px !important;
        }
        .stDownloadButton > button:hover {
            background-color: #33B784 !important;
            border-color: #33B784 !important;
        }

        /* Sidebar */
        [data-testid="stSidebar"] {
            background-color: #f5faf7 !important;
        }
        [data-testid="stSidebar"] .stMarkdown p,
        [data-testid="stSidebar"] .stMarkdown li {
            font-size: 0.88rem !important;
        }

        /* Success messages → Limoengroen tint */
        .stAlert[data-baseweb="notification"][kind="positive"],
        div[data-testid="stNotification"][class*="success"] {
            background-color: #edfaf3 !important;
            border-left-color: #33B784 !important;
        }

        /* Metric labels */
        [data-testid="stMetricLabel"] {
            font-family: 'Rubik', sans-serif !important;
            font-size: 0.78rem !important;
            font-weight: 600 !important;
            color: #00573C !important;
        }

        /* Tab labels */
        .stTabs [data-baseweb="tab"] {
            font-family: 'Rubik', sans-serif !important;
            font-weight: 600 !important;
        }
        .stTabs [aria-selected="true"] {
            color: #00573C !important;
            border-bottom-color: #33B784 !important;
        }

        /* Page header bar */
        .ok-header {
            background: linear-gradient(135deg, #00573C 0%, #33B784 100%);
            color: #ffffff;
            padding: 20px 28px 18px 28px;
            border-radius: 12px;
            margin-bottom: 8px;
        }
        .ok-header h1 {
            font-family: 'Rubik', sans-serif !important;
            font-size: 1.7rem !important;
            font-weight: 800 !important;
            margin: 0 0 2px 0 !important;
            color: #ffffff !important;
        }
        .ok-header p {
            margin: 0 !important;
            font-size: 0.85rem !important;
            opacity: 0.85;
        }

        /* Footer */
        .ok-footer {
            text-align: center;
            padding: 24px 0 12px 0;
            color: #6c757d;
            font-size: 0.82rem;
        }
        .ok-footer strong {
            color: #00573C;
            font-weight: 700;
            letter-spacing: 0.04em;
        }

        /* Briefing card containers */
        [data-testid="stVerticalBlockBorderWrapper"] {
            border: 1px solid #d5e8dd !important;
            border-radius: 14px !important;
            background: #ffffff !important;
        }
        [data-testid="stVerticalBlockBorderWrapper"]:hover {
            border-color: #33B784 !important;
            box-shadow: 0 2px 12px rgba(0,87,60,.08) !important;
            transition: box-shadow .2s, border-color .2s;
        }

        /* Upload drop-zones */
        [data-testid="stFileUploaderDropzone"] {
            background: #f5faf7 !important;
            border: 2px dashed #33B784 !important;
            border-radius: 10px !important;
        }

        /* Popover button */
        [data-testid="stPopover"] button {
            background: #f5faf7 !important;
            border: 1px solid #33B784 !important;
            color: #00573C !important;
            font-weight: 600 !important;
            border-radius: 8px !important;
            width: 100% !important;
        }
        [data-testid="stPopover"] button:hover {
            background: #edfaf3 !important;
        }

        /* Hide Streamlit deploy button */
        #MainMenu, footer, header { visibility: hidden; }

        /* Subtle tab bar */
        .stTabs [data-baseweb="tab-list"] {
            border-bottom: 2px solid #e0ede6;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Zijbalk
# ---------------------------------------------------------------------------

with st.sidebar:
    st.markdown(
        "<div style='padding:8px 0 6px 0'>"
        "<span style='font-size:1.15rem;font-weight:800;color:#00573C;font-family:Rubik,sans-serif'>"
        "Online Klik</span>"
        "<span style='font-size:0.73rem;color:#6c757d;display:block;margin-top:2px'>"
        "Meta Ads Analyser</span></div>",
        unsafe_allow_html=True,
    )
    st.divider()

    _phase_now = st.session_state.get("phase", "UPLOAD")

    if _phase_now == "UPLOAD":
        st.markdown(
            "**Hoe werkt het?**\n\n"
            "1. Upload je Meta Ads CSV-export\n"
            "2. Upload je bannerafbeeldingen\n"
            "3. Koppel je banners aan de toppers\n"
            "4. Klik op **Matches Bevestigen & Analyse Starten**\n\n"
            "De AI classificeert je advertenties, koppelt de visuals "
            "aan de juiste advertenties en genereert 10 creatieve briefings."
        )

    elif _phase_now == "MATCHER":
        st.markdown(
            "**Pre-Flight Check**\n\n"
            "De AI heeft een voorzet gemaakt op basis van je bestandsnamen. "
            "Controleer de koppelingen, pas aan waar nodig, en bevestig.\n\n"
            "_Geen API-aanroepen tijdens dit scherm — alles is instant._"
        )
        if st.button("↩️ Terug naar upload", use_container_width=True):
            st.session_state.phase = "UPLOAD"
            st.rerun()

    elif _phase_now == "RESULTS" and "results" in st.session_state:
        _sr = st.session_state["results"]

        # ── Downloads ─────────────────────────────────────────────────────
        st.markdown(
            "<div style='font-size:0.72rem;font-weight:700;color:#00573C;"
            "text-transform:uppercase;letter-spacing:0.06em;margin-bottom:6px'>"
            "📥 Downloads</div>",
            unsafe_allow_html=True,
        )
        _pdf_cached = _sr.get("pdf_bytes", b"")
        if _pdf_cached:
            st.download_button(
                "📄 Download Rapport (PDF)",
                data=_pdf_cached,
                file_name="online_klik_analyserapport.pdf",
                mime="application/pdf",
                use_container_width=True,
            )
        st.download_button(
            "⬇️ Download Briefings (Markdown)",
            data=_sr.get("concepts_md", "").encode("utf-8"),
            file_name="creatieve_briefings.md",
            mime="text/markdown",
            use_container_width=True,
        )
        st.download_button(
            "📊 Download CSV",
            data=_sr["df"].to_csv(index=False).encode("utf-8"),
            file_name="geclassificeerde_advertenties.csv",
            mime="text/csv",
            use_container_width=True,
        )
        st.divider()

        # ── Master Prompt how-to ───────────────────────────────────────────
        with st.expander("💡 Master Prompt gebruiken", expanded=False):
            st.markdown(
                "1. Ga naar **Tab 3 → Creatieve Briefings**\n"
                "2. Lees de **Hook**, **Primary Text** en **Headline**\n"
                "3. Kopieer de **Master Prompt** (XML-blok)\n"
                "4. Ga naar **[Nano Banana Pro](https://nanobananapro.com)** of **[Freepik](https://freepik.com)**\n"
                "5. Klik **🖼️ Zie referentieafbeelding** in de kaart → upload die foto\n"
                "6. Plak de prompt en genereer je visual\n\n"
                "_De prompt integreert je product naadloos in de nieuwe scène._"
            )

        if st.button("🔄 Opnieuw Analyseren", use_container_width=True, type="primary"):
            st.session_state.pop("results", None)
            st.session_state.pop("final_matches", None)
            # Clear stale MATCHER selectbox values so defaults are recomputed
            for _k in [k for k in st.session_state if k.startswith("match_")]:
                del st.session_state[_k]
            st.session_state.phase = "MATCHER"
            st.rerun()
        if st.button("↩️ Nieuwe analyse starten", use_container_width=True):
            for _k in ("results", "_csv_bytes", "_csv_name", "_imgs", "final_matches"):
                st.session_state.pop(_k, None)
            for _k in [k for k in st.session_state if k.startswith("match_")]:
                del st.session_state[_k]
            st.session_state.phase = "UPLOAD"
            st.rerun()

    st.divider()
    with st.expander("🛠️ Developer Settings", expanded=False):
        api_key = st.text_input(
            "OpenAI API-sleutel",
            value=os.getenv("OPENAI_API_KEY", ""),
            type="password",
            help="Jouw sleutel wordt niet opgeslagen buiten deze sessie.",
        )
        st.caption("Sleutel is alleen actief tijdens deze sessie.")
        st.caption(
            "☁️ **Cloud-deployment:** stel de sleutel in als omgevingsvariabele via het "
            "Streamlit Cloud dashboard (App settings → Secrets) of het Vercel dashboard "
            "(Settings → Environment Variables) onder de naam `OPENAI_API_KEY`."
        )

# ---------------------------------------------------------------------------
# Session state — initialise + refresh-recovery guards
# ---------------------------------------------------------------------------

if "phase" not in st.session_state:
    st.session_state.phase = "UPLOAD"

# Guard 1: pipeline finished but phase wasn't advanced (mid-rerun edge case)
if st.session_state.phase == "LANCERING" and "results" in st.session_state:
    st.session_state.phase = "RESULTS"

# Guard 2: user refreshed during LANCERING — session memory is gone
if st.session_state.phase == "LANCERING" and not st.session_state.get("_csv_bytes"):
    st.session_state.phase = "UPLOAD"
    st.session_state["_refresh_warning"] = True

# Guard 3: user refreshed during MATCHER — session memory is gone
if st.session_state.phase == "MATCHER" and not st.session_state.get("_csv_bytes"):
    st.session_state.phase = "UPLOAD"
    st.session_state["_refresh_warning"] = True

# ---------------------------------------------------------------------------
# Koptekst (always visible)
# ---------------------------------------------------------------------------

st.markdown(
    "<div class='ok-header'>"
    "<h1>Online Klik | Meta Ads Analyser</h1>"
    "<p>Upload je campagnedata en banners — de AI analyseert wat werkt, waarom het werkt, "
    "en genereert direct nieuwe creatieve briefings op basis van de winnende patronen.</p>"
    "</div>",
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Main content area — swaps between Phase 1 / Phase 2 / Phase 3
# ---------------------------------------------------------------------------

_main_area = st.empty()

# ---------------------------------------------------------------------------
# Hulpfuncties
# ---------------------------------------------------------------------------

def load_csv_from_upload(file) -> pd.DataFrame:
    """
    Reads an uploaded CSV regardless of delimiter (comma, semicolon, tab)
    or encoding (UTF-8, latin-1, cp1252). Uses sep=None + Python engine for
    auto-detection so Meta Ads exports from any locale are handled correctly.
    """
    file.seek(0)
    raw = file.read()
    for enc in ("utf-8", "latin-1", "cp1252"):
        try:
            return pd.read_csv(
                BytesIO(raw),
                sep=None,
                engine="python",
                encoding=enc,
                encoding_errors="replace",
                on_bad_lines="skip",
            )
        except Exception:
            continue
    raise ValueError(
        "Het CSV-bestand kon niet worden gelezen. "
        "Gebruik a.u.b. een originele Meta Ads Manager export."
    )


def encode_uploaded_image(file) -> Tuple[str, str]:
    ext = os.path.splitext(file.name)[1].lower()
    mime = "image/jpeg" if ext in (".jpg", ".jpeg") else f"image/{ext.lstrip('.')}"
    file.seek(0)
    return base64.b64encode(file.read()).decode("utf-8"), mime


def describe_image(client: OpenAI, image_file, ad_name: str) -> str:
    b64, mime = encode_uploaded_image(image_file)
    system_msg = (
        "Je bent een senior Meta Ads creatief strateeg. "
        "Je geeft uitsluitend feitelijke, directe analyse. "
        "Je antwoord begint ALTIJD direct met punt 1 — geen inleiding, geen begroeting, "
        "geen bevestiging, geen samenvatting vooraf. Eerste karakter is '1'."
    )
    prompt = (
        f"Analyseer de banneradvertentie '{ad_name}'.\n\n"
        "1. **Hook / Headline** — Wat ziet de kijker als eerste? Voordeel-gedreven, nieuwsgierigheidsgericht of aanbod-gebaseerd?\n"
        "2. **Visuele Stijl** — Minimalistisch, gedurfd, lifestyle, productgericht?\n"
        "3. **Kleurenpalet** — Dominante kleuren en de emotie die ze uitstralen.\n"
        "4. **Productpresentatie** — Close-up, in context, flatlay?\n"
        "5. **Persoon / Model** — Aanwezig? Uitdrukking, houding, doelgroepkenmerken.\n"
        "6. **CTA** — Tekst en prominentie.\n"
        "7. **Eerste Indruk** — Welk gevoel of verlangen wekt dit op? Één zin.\n\n"
        "Gebruik opsommingstekens onder elke kop. Wees specifiek, geen inleiding."
    )
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": system_msg},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:{mime};base64,{b64}", "detail": "high"},
                    },
                ],
            },
        ],
        max_tokens=900,
    )
    return response.choices[0].message.content


def compare_creatives(
    client: OpenAI,
    high: List[Dict],
    underperformers: List[Dict],
    no_data: List[Dict],
) -> str:
    def fmt(ads):
        return "\n\n".join(
            f"### {a['name']} ({CATEGORY_NL.get(a['category'], a['category'])})\n{a['description']}"
            for a in ads
        )

    sections = []
    if high:
        sections.append("## TOP PRESTEERDERS\n" + fmt(high))
    if underperformers:
        sections.append("## ONDERPRESTEERDERS\n" + fmt(underperformers))
    if no_data:
        sections.append("## GEEN PRESTATIEDATA\n" + fmt(no_data))

    context = "\n\n---\n\n".join(sections)
    system_msg = f"Je bent een senior Meta Ads creatief strateeg. {TONE_INSTRUCTION}"
    prompt = (
        "Hieronder staan visuele beschrijvingen van advertenties uit dezelfde campagne, "
        "gegroepeerd per prestatieniveau.\n\n"
        f"{context}\n\n---\n\n"
        "Schrijf op basis van uitsluitend de visuele en creatieve elementen een strategische analyse "
        "met de volgende secties:\n\n"
        "## 1. Waarom de Winnaar Wint\n"
        "Benoem de specifieke creatieve keuzes die de hogere prestatie veroorzaken. Verwijs naar advertentienamen.\n\n"
        "## 2. Waarom de Verliezer Verliest\n"
        "Benoem de specifieke creatieve zwaktes. Wat ontbreekt of slaagt er niet in de scroll te stoppen? "
        "Verwijs naar advertentienamen.\n\n"
        "## 3. Het Beslissende Patroon\n"
        "Het belangrijkste creatieve verschil in één vetgedrukte, pakkende zin.\n\n"
        "## 4. Aanbevelingen\n"
        "Geef 3 concrete, specifieke verbeteringen. Gebruik voor elke aanbeveling dit formaat:\n\n"
        "**Aanbeveling N:** [actie]\n"
        "**Prioriteit:** Hoog | Middel | Laag\n"
        "**Verwachte Impact:** [waarom dit werkt en welk resultaat je verwacht]\n\n"
        "Vermijd generiek advies."
    )
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": system_msg},
            {"role": "user", "content": prompt},
        ],
        max_tokens=1400,
    )
    return response.choices[0].message.content




def generate_concepts(
    client: OpenAI, analysis_report: str, image_filenames: Optional[List[str]] = None
) -> List[Dict]:
    """
    Returns a list of concept dicts with keys:
    nummer, titel, prioriteit, rationale,
    hook, primary_text, headline, visuele_omschrijving,
    referentie_afbeelding, master_prompt
    """
    image_filenames = image_filenames or []
    filenames_block = (
        "Beschikbare referentie-afbeeldingen (exacte bestandsnamen — alleen gebruiken voor het veld "
        "referentie_afbeelding, NIET als tekst in de master_prompt):\n"
        + "\n".join(f"  - {fn}" for fn in image_filenames)
        if image_filenames
        else "Geen afbeeldingen geüpload."
    )

    system_msg = (
        "Je bent een senior copywriter en creatief directeur voor een high-end Nederlandse sieraden-"
        "webshop. Je schrijft Instagram-waardige advertentieteksten: pakkend, menselijk, direct. "
        + TONE_INSTRUCTION
    )
    prompt = (
        "Hieronder staat een visueel analyserapport van een 'Summer Sale'-campagne.\n\n"
        f"---\n{analysis_report}\n---\n\n"
        f"{filenames_block}\n\n"
        "Genereer op basis van de winnende patronen 10 nieuwe Meta-advertentieconcepten.\n\n"

        "COPYWRITING-STIJLGIDS — volg dit strikt:\n"
        "TOON: Spreek de lezer aan met 'je/jouw', nooit 'u'. Licht maar elegant. Menselijk, niet corporate.\n"
        "HOOK (thumb-stopper): De eerste zin die de scroll stopt. Persoonlijk, prikkelend of emotioneel. "
        "Max. 10 woorden. VERBODEN: 'Ontdek de magie', 'Shop nu', 'Tijdelijk aanbod', "
        "'Begin je sieradenreis', 'Verander je look'. "
        "GOED: 'Een cadeautje voor jezelf, waarom niet?', 'De ring waar iedereen naar vraagt.', "
        "'Jouw zomerlook mist nog één ding.'\n"
        "PRIMARY TEXT: 2–4 zinnen. Vertel een mini-verhaal of spreek een herkenbaar gevoel aan. "
        "Vermeld het aanbod (korting/gratis cadeau) op een natuurlijke manier — niet als een marktkoopman. "
        "VERBODEN: generieke openers als 'Wij bieden aan...', 'Profiteer nu van...', 'Mis het niet!'.\n"
        "HEADLINE (klik-trigger): Max. 8 woorden. Concreet en actiegericht zonder robotachtige "
        "Call-to-Actions. GOED: 'Jouw nieuwe favoriet voor de zomer', 'De details die je look afmaken', "
        "'Gratis cadeau bij elke bestelling'. VERBODEN: 'Shop de collectie', 'Ontdek nu'.\n\n"

        "Retourneer een geldig JSON-object met een 'concepten' array van precies 10 objecten. "
        "Elk object heeft:\n"
        "- nummer (int): 1–10\n"
        "- titel (string): interne naam voor het concept\n"
        "- prioriteit (string): 'Hoog', 'Middel' of 'Laag'\n"
        "- rationale (string): precies 2 zinnen in het Nederlands die uitleggen waarom dit concept zal converteren — "
        "gebaseerd op de winnende patronen in de analyse. Eerste zin = het gedragsinzicht (waarom de doelgroep reageert). "
        "Tweede zin = de verwachte zakelijke impact (CTR, ROAS, of betrokkenheid).\n"
        "- hook (string): thumb-stopper, max. 10 woorden\n"
        "- primary_text (string): 2–4 zinnen, menselijk en on-brand, inclusief het aanbod\n"
        "- headline (string): klik-trigger, max. 8 woorden\n"
        "- visuele_omschrijving (string): volledige art direction briefing voor de ontwerper — "
        "hero-afbeelding, model (uitdrukking, houding, demografisch profiel), productplaatsing, "
        "kleurenpalet, typografiestijl, sfeer, grafische elementen, beeldverhouding\n"
        "- referentie_afbeelding (string): kies de EXACTE bestandsnaam (uit de lijst hierboven) die het beste "
        "past als visuele referentie voor dit concept. "
        "Als het concept product-gericht is, kies de meest passende afbeelding. "
        "Als het concept service- of sfeer-gericht is zonder duidelijk product, gebruik de waarde 'geen'.\n"
        "- master_prompt (string): een rijke, beschrijvende XML-prompt (minimaal 60 woorden) voor "
        "Freepik of Nano Banana Pro. STRIKTE REGELS:\n"
        "  (1) NOOIT bestandsnamen gebruiken in de prompt-tekst — beschrijf het product VISUEEL "
        "      (bv. 'a delicate gold ring with a single diamond', 'a structured leather handbag in cognac'). "
        "      Als meerdere producten zichtbaar kunnen zijn, specificeer: 'Focus on: [product description] — "
        "      [kleur, materiaal, vorm]'.\n"
        "  (2) Formatteer de output ALTIJD als dit exacte XML-blok (geen extra tekst eromheen):\n"
        "      <prompt>\n"
        "      [Scene: beschrijf de visuele scène in detail — locatie, sfeer, props, achtergrond]\n"
        "      [Style: lighting type, camera angle, composition, Online Klik aesthetic — clean, professional, "
        "      conversion-focused, minimalist luxury, 8K commercial photography]\n"
        "      [Integration: Seamlessly integrate the product from the provided reference image into this scene.]\n"
        "      </prompt>\n"
        "  (3) De [Scene] sectie beschrijft de gewenste advertentiebeeldopbouw — NIET het product zelf.\n"
        "  (4) De [Style] sectie dekt verplicht: lighting (bv. soft diffused studio light), "
        "      camera angle (bv. close-up hero shot), composition (bv. rule of thirds, negative space).\n"
        "  (5) De [Integration] regel is altijd letterlijk: "
        "      'Seamlessly integrate the product from the provided reference image into this scene.'\n\n"
        "Extra regels:\n"
        "- Bouw voort op de winnende patronen: dubbel aanbod, elegante close-ups, aspirationeel model.\n"
        "- Varieer de haakjes: nieuwsgierigheid, social proof, urgentie, storytelling, voordeel.\n"
        "- Elk concept voelt uniek aan — geen herhalingen in toon of structuur.\n"
        "- Alle tekstvelden behalve master_prompt in het Nederlands.\n"
        "- Retourneer ALLEEN het JSON-object, geen extra tekst."
    )
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": system_msg},
            {"role": "user", "content": prompt},
        ],
        max_tokens=6000,
        response_format={"type": "json_object"},
    )
    raw = response.choices[0].message.content
    parsed = json.loads(raw)
    # GPT may wrap the array in a key — unwrap if needed
    if isinstance(parsed, dict):
        for v in parsed.values():
            if isinstance(v, list):
                return v
    return parsed


# ---------------------------------------------------------------------------
# Cached API wrappers — re-runs skip the API and return from cache instantly
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner=False, ttl=3600)
def _describe_image_cached(b64: str, mime: str, ad_name: str, api_key: str) -> str:
    """GPT-4o vision call — cached 1 h by (image content, ad name)."""
    _c = OpenAI(api_key=api_key)
    system_msg = (
        "Je bent een senior Meta Ads creatief strateeg. "
        "Je geeft uitsluitend feitelijke, directe analyse. "
        "Je antwoord begint ALTIJD direct met punt 1 — geen inleiding, geen begroeting, "
        "geen bevestiging, geen samenvatting vooraf. Eerste karakter is '1'."
    )
    prompt = (
        f"Analyseer de banneradvertentie '{ad_name}'.\n\n"
        "1. **Hook / Headline** — Voordeel-gedreven, nieuwsgierigheidsgericht of aanbod-gebaseerd?\n"
        "2. **Visuele Stijl** — Minimalistisch, gedurfd, lifestyle, productgericht?\n"
        "3. **Kleurenpalet** — Dominante kleuren en de emotie die ze uitstralen.\n"
        "4. **Productpresentatie** — Close-up, in context, flatlay?\n"
        "5. **Persoon / Model** — Aanwezig? Uitdrukking, houding, doelgroepkenmerken.\n"
        "6. **CTA** — Tekst en prominentie.\n"
        "7. **Eerste Indruk** — Welk gevoel of verlangen wekt dit op? Één zin.\n\n"
        "Gebruik opsommingstekens onder elke kop. Wees specifiek, geen inleiding."
    )
    response = _c.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": system_msg},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url",
                     "image_url": {"url": f"data:{mime};base64,{b64}", "detail": "high"}},
                ],
            },
        ],
        max_tokens=900,
    )
    return response.choices[0].message.content


def _build_creatives_context(
    high: List[Dict], underperformers: List[Dict], no_data: List[Dict]
) -> str:
    """Build the grouped-ads context string that is fed to the comparison prompt."""
    def fmt(ads):
        return "\n\n".join(
            f"### {a['name']} ({CATEGORY_NL.get(a['category'], a['category'])})\n{a['description']}"
            for a in ads
        )
    sections = []
    if high:
        sections.append("## TOP PRESTEERDERS\n" + fmt(high))
    if underperformers:
        sections.append("## ONDERPRESTEERDERS\n" + fmt(underperformers))
    if no_data:
        sections.append("## GEEN PRESTATIEDATA\n" + fmt(no_data))
    return "\n\n---\n\n".join(sections)


@st.cache_data(show_spinner=False, ttl=3600)
def _compare_creatives_cached(context: str, api_key: str) -> str:
    """GPT-4o strategy call — cached 1 h by analysis context hash."""
    _c = OpenAI(api_key=api_key)
    system_msg = f"Je bent een senior Meta Ads creatief strateeg. {TONE_INSTRUCTION}"
    prompt = (
        "Hieronder staan visuele beschrijvingen van advertenties uit dezelfde campagne, "
        "gegroepeerd per prestatieniveau.\n\n"
        f"{context}\n\n---\n\n"
        "Schrijf een strategische analyse met de volgende secties:\n\n"
        "## 1. Waarom de Winnaar Wint\n"
        "Benoem de specifieke creatieve keuzes die de hogere prestatie veroorzaken. "
        "Verwijs naar advertentienamen.\n\n"
        "## 2. Waarom de Verliezer Verliest\n"
        "Benoem de specifieke creatieve zwaktes. Verwijs naar advertentienamen.\n\n"
        "## 3. Het Beslissende Patroon\n"
        "Het belangrijkste creatieve verschil in één vetgedrukte, pakkende zin.\n\n"
        "## 4. Aanbevelingen\n"
        "Geef 3 concrete verbeteringen:\n\n"
        "**Aanbeveling N:** [actie]\n"
        "**Prioriteit:** Hoog | Middel | Laag\n"
        "**Verwachte Impact:** [waarom dit werkt]\n\n"
        "Vermijd generiek advies."
    )
    response = _c.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": system_msg},
            {"role": "user", "content": prompt},
        ],
        max_tokens=1400,
    )
    return response.choices[0].message.content


@st.cache_data(show_spinner=False, ttl=3600)
def _generate_concepts_cached(
    analysis_report: str, filenames_tuple: tuple, api_key: str
) -> list:
    """GPT-4o concepts call — cached 1 h by (report, filenames) hash."""
    return generate_concepts(
        OpenAI(api_key=api_key), analysis_report, list(filenames_tuple)
    )


def _strip_markdown(text: str):
    """Remove common Markdown syntax for plain-text output (e.g. PDF)."""
    text = re.sub(r"\*\*(.+?)\*\*", r"\1", text)
    text = re.sub(r"\*(.+?)\*", r"\1", text)
    text = re.sub(r"#{1,6}\s+", "", text)
    text = re.sub(r"`(.+?)`", r"\1", text)
    text = re.sub(r"_{1,2}(.+?)_{1,2}", r"\1", text)
    return text.strip()


def safe_pdf_text(text: str) -> str:
    """
    Replace common Unicode punctuation with ASCII equivalents, then strip
    any character outside the latin-1 range (emojis, special symbols) so
    FPDF core fonts never raise an encoding error.
    """
    if not text:
        return ""
    replacements = {
        "\u2019": "'", "\u2018": "'",
        "\u201c": '"', "\u201d": '"',
        "\u2013": "-", "\u2014": "--",
        "\u2026": "...", "\u20ac": "EUR",
        "\u2022": "-", "\u00b7": "-",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    # Drop anything that cannot be encoded in latin-1 (e.g. emojis)
    return text.encode("latin-1", errors="ignore").decode("latin-1")


def _pdf_header_bar(pdf: FPDF, title: str) -> None:
    pdf.set_fill_color(0, 87, 60)
    pdf.set_text_color(255, 255, 255)
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(pdf.epw, 9, safe_pdf_text(title), fill=True, ln=True)
    pdf.set_text_color(0, 0, 0)
    pdf.ln(2)


def _pdf_footer(pdf: FPDF) -> None:
    pdf.set_y(-16)
    pdf.set_font("Helvetica", "I", 8)
    pdf.set_text_color(160, 160, 160)
    pdf.cell(
        pdf.epw, 6,
        "Team up, power up  |  Online Klik  |  Authentiek onlinemarketingbureau van Zuid-Nederland",
        align="C", ln=True,
    )


# Fixed margins — 10 mm on all sides → effective page width = 190 mm
_PDF_MARGIN   = 10
_PDF_LABEL_W  = 36   # width of label column in concept fields
_PDF_STAT_W   = 100  # width of label column in summary stats


def generate_pdf(
    df: pd.DataFrame,
    metric_used: str,
    analysed: List[Dict],
    analysis_text: str,
    concepts: List[Dict],
) -> bytes:
    pdf = FPDF()
    pdf.core_fonts_encoding = "utf-8"
    pdf.set_auto_page_break(auto=True, margin=18)
    pdf.set_margins(_PDF_MARGIN, _PDF_MARGIN, _PDF_MARGIN)

    # ── Title page ────────────────────────────────────────────────────────────
    pdf.add_page()
    # Full-bleed green header band (absolute coordinates, bypasses margins)
    pdf.set_fill_color(0, 87, 60)
    pdf.rect(0, 0, 210, 48, "F")

    # Header text — positioned within the green band
    pdf.set_xy(_PDF_MARGIN, 14)
    pdf.set_font("Helvetica", "B", 20)
    pdf.set_text_color(255, 255, 255)
    pdf.cell(pdf.epw, 10, "Online Klik | Meta Ads Analyser", align="C", ln=True)
    pdf.set_x(_PDF_MARGIN)
    pdf.set_font("Helvetica", "", 10)
    pdf.cell(pdf.epw, 7, "Authentiek onlinemarketingbureau van Zuid-Nederland", align="C", ln=True)

    # Body starts below the band
    pdf.set_xy(_PDF_MARGIN, 58)
    pdf.set_text_color(0, 87, 60)
    pdf.set_font("Helvetica", "B", 16)
    pdf.cell(pdf.epw, 10, "Campagne Analyserapport", ln=True)

    pdf.set_x(_PDF_MARGIN)
    pdf.set_font("Helvetica", "", 10)
    pdf.set_text_color(80, 80, 80)
    pdf.cell(pdf.epw, 7, safe_pdf_text(f"Gegenereerd op: {datetime.now().strftime('%d-%m-%Y om %H:%M')}"), ln=True)
    pdf.set_x(_PDF_MARGIN)
    pdf.cell(pdf.epw, 7, safe_pdf_text(f"Gebruikte metric: {metric_used}"), ln=True)
    pdf.ln(6)

    # Performance summary table
    counts = df["performance_category"].value_counts()
    total = len(df)
    pdf.set_text_color(0, 0, 0)
    val_w = pdf.epw - _PDF_STAT_W
    for label, val in [
        ("Totaal advertenties", str(total)),
        ("Top Presteerders", str(int(counts.get("High Performer", 0)))),
        ("Gemiddeld", str(int(counts.get("Average", 0)))),
        ("Onderpresteerders", str(int(counts.get("Underperformer", 0)))),
        ("Geen prestatiedata", str(int(counts.get("No Data", 0)))),
    ]:
        pdf.set_x(_PDF_MARGIN)
        pdf.set_font("Helvetica", "", 10)
        pdf.cell(_PDF_STAT_W, 7, safe_pdf_text(label) + ":", ln=False)
        pdf.set_font("Helvetica", "B", 10)
        pdf.cell(val_w, 7, val, ln=True)

    _pdf_footer(pdf)

    # ── Visual analysis ───────────────────────────────────────────────────────
    if analysed:
        pdf.add_page()
        _pdf_header_bar(pdf, "Visuele Analyse per Advertentie")
        for ad in analysed:
            if pdf.get_y() > 235:
                pdf.add_page()
                _pdf_header_bar(pdf, "Visuele Analyse (vervolg)")
            cat_nl = CATEGORY_NL.get(ad["category"], ad["category"])
            pdf.set_x(_PDF_MARGIN)
            pdf.set_font("Helvetica", "B", 10)
            pdf.set_text_color(0, 87, 60)
            pdf.cell(pdf.epw, 7, safe_pdf_text(f"{ad['name']}  ({cat_nl})"), ln=True)
            pdf.set_x(_PDF_MARGIN)
            pdf.set_font("Helvetica", "", 9)
            pdf.set_text_color(40, 40, 40)
            pdf.multi_cell(pdf.epw, 4.5, safe_pdf_text(_strip_markdown(ad["description"])))
            pdf.ln(2)
            pdf.set_draw_color(210, 210, 210)
            lx = _PDF_MARGIN
            rx = 210 - _PDF_MARGIN
            pdf.line(lx, pdf.get_y(), rx, pdf.get_y())
            pdf.ln(3)
        _pdf_footer(pdf)

    # ── Strategic analysis ────────────────────────────────────────────────────
    if analysis_text:
        pdf.add_page()
        _pdf_header_bar(pdf, "Strategische Analyse")
        pdf.set_x(_PDF_MARGIN)
        pdf.set_font("Helvetica", "", 9)
        pdf.set_text_color(40, 40, 40)
        pdf.multi_cell(pdf.epw, 4.5, safe_pdf_text(_strip_markdown(analysis_text)))
        _pdf_footer(pdf)

    # ── Creative concepts ─────────────────────────────────────────────────────
    if concepts:
        pdf.add_page()
        _pdf_header_bar(pdf, "10 Creatieve Briefings")
        val_w = pdf.epw - _PDF_LABEL_W
        for c in concepts:
            if pdf.get_y() > 220:
                pdf.add_page()
                _pdf_header_bar(pdf, "Creatieve Briefings (vervolg)")
                val_w = pdf.epw - _PDF_LABEL_W

            # Concept title
            pdf.set_x(_PDF_MARGIN)
            pdf.set_font("Helvetica", "B", 11)
            pdf.set_text_color(0, 87, 60)
            pdf.cell(
                pdf.epw, 8,
                safe_pdf_text(f"Concept {c.get('nummer', '?')}: {c.get('titel', '')}"),
                ln=True,
            )
            # Priority + impact (italic subtitle)
            pdf.set_x(_PDF_MARGIN)
            pdf.set_font("Helvetica", "I", 9)
            pdf.set_text_color(100, 100, 100)
            pdf.multi_cell(
                pdf.epw, 5,
                safe_pdf_text(f"Prioriteit: {c.get('prioriteit', '-')}  |  {c.get('verwachte_impact', '-')}"),
            )
            # Field rows: label + value
            for field_label, field_key in [
                ("Hook",           "hook"),
                ("Primary Text",   "primary_text"),
                ("Headline",       "headline"),
                ("Visuele Briefing", "visuele_omschrijving"),
            ]:
                pdf.set_x(_PDF_MARGIN)
                pdf.set_font("Helvetica", "B", 9)
                pdf.set_text_color(60, 60, 60)
                # Label cell — no newline, moves cursor right
                pdf.cell(_PDF_LABEL_W, 5, safe_pdf_text(field_label) + ":", ln=False)
                pdf.set_font("Helvetica", "", 9)
                pdf.set_text_color(20, 20, 20)
                pdf.multi_cell(val_w, 5, safe_pdf_text(str(c.get(field_key, "-"))))

            # Referentie-afbeelding
            ref_afb = c.get("referentie_afbeelding", "")
            if ref_afb:
                pdf.set_x(_PDF_MARGIN)
                pdf.set_font("Helvetica", "B", 9)
                pdf.set_text_color(0, 87, 60)
                pdf.cell(_PDF_LABEL_W, 5, "Referentie-afbeelding:", ln=False)
                pdf.set_font("Helvetica", "", 9)
                pdf.set_text_color(20, 20, 20)
                ref_display = "Geen product-referentie" if ref_afb.lower() == "geen" else ref_afb
                pdf.multi_cell(val_w, 5, safe_pdf_text(ref_display))

            # Rationale
            rationale = c.get("rationale", c.get("design_strategy", ""))
            if rationale:
                pdf.set_x(_PDF_MARGIN)
                pdf.set_font("Helvetica", "B", 9)
                pdf.set_text_color(0, 87, 60)
                pdf.cell(_PDF_LABEL_W, 5, "Rationale:", ln=False)
                pdf.set_font("Helvetica", "I", 9)
                pdf.set_text_color(20, 20, 20)
                pdf.multi_cell(val_w, 5, safe_pdf_text(rationale))

            # Master Prompt (XML)
            master_prompt = c.get("master_prompt", "")
            if master_prompt:
                pdf.set_x(_PDF_MARGIN)
                pdf.set_font("Helvetica", "B", 9)
                pdf.set_text_color(0, 87, 60)
                pdf.cell(pdf.epw, 5, "Master Prompt — Nano Banana Pro / Freepik (XML):", ln=True)
                pdf.set_x(_PDF_MARGIN)
                pdf.set_fill_color(245, 250, 247)
                pdf.set_font("Courier", "", 8)
                pdf.set_text_color(20, 20, 20)
                pdf.multi_cell(pdf.epw, 4.5, safe_pdf_text(master_prompt), fill=True)

            pdf.ln(2)
            pdf.set_draw_color(210, 210, 210)
            lx = _PDF_MARGIN
            rx = 210 - _PDF_MARGIN
            pdf.line(lx, pdf.get_y(), rx, pdf.get_y())
            pdf.ln(4)
        _pdf_footer(pdf)

    return bytes(pdf.output())


def clipboard_button(text: str, key: str) -> None:
    """Renders a small branded copy-to-clipboard button via JS."""
    safe = (
        text.replace("\\", "\\\\")
            .replace("'", "\\'")
            .replace("\n", "\\n")
            .replace("\r", "")
    )
    components.html(
        f"""
        <button id="cb_{key}"
            onclick="navigator.clipboard.writeText('{safe}').then(()=>{{
                var b=document.getElementById('cb_{key}');
                b.innerText='✓ Gekopieerd!';b.style.background='#00573C';
                setTimeout(()=>{{b.innerText='📋 Kopieer';b.style.background='#33B784'}},2000);
            }})"
            style="background:#33B784;color:#fff;border:none;padding:5px 14px;
                   border-radius:6px;cursor:pointer;font-size:0.8rem;font-weight:600;
                   font-family:Rubik,sans-serif;margin-top:4px;transition:background 0.2s">
            📋 Kopieer
        </button>""",
        height=42,
    )


def parse_recommendations(analysis_text: str) -> List[Dict]:
    """Extract structured recommendations from the compare_creatives output."""
    pattern = (
        r"\*\*Aanbeveling\s+\d+:\*\*\s*(.+?)\n"
        r"\*\*Prioriteit:\*\*\s*(Hoog|Middel|Laag)\n"
        r"\*\*Verwachte Impact:\*\*\s*(.+?)(?=\n\*\*Aanbeveling|\Z)"
    )
    matches = re.findall(pattern, analysis_text, re.DOTALL)
    return [
        {"actie": m[0].strip(), "prioriteit": m[1].strip(), "impact": m[2].strip()}
        for m in matches
    ]


def match_category(filename: str, perf_map: Dict[str, str]) -> Tuple[str, str]:
    """
    Three-pass matching — most to least strict:

    Pass 1 — slug containment: "summer_sale_banner" ↔ "Summer Sale Banner"
    Pass 2 — substring containment: any significant word from the filename
             appears inside the ad name string (covers "banner_v1" → "Ad Banner v1")
    Pass 3 — word-overlap scoring: highest token overlap wins
    """
    stem = os.path.splitext(filename)[0]
    stem_lower = stem.lower()
    # Normalise: underscores/dashes → spaces
    stem_clean = re.sub(r"[_\-]+", " ", stem_lower).strip()

    # Pass 1: bi-directional slug containment
    for ad_name, cat in perf_map.items():
        ad_slug  = ad_name.lower().replace(" ", "_")
        ad_lower = ad_name.lower()
        if (ad_slug in stem_lower or stem_lower in ad_slug
                or ad_lower in stem_clean or stem_clean in ad_lower):
            return ad_name, cat

    # Pass 2: any meaningful word from the filename found as substring in ad name
    stem_words = [w for w in stem_clean.split() if len(w) > 2]
    best_score, best_name, best_cat = 0, stem, "No Data"
    for ad_name, cat in perf_map.items():
        ad_lower = ad_name.lower()
        # Exact word intersection (weight 2) + substring hits (weight 1)
        ad_tokens = set(ad_lower.split())
        exact   = sum(2 for w in stem_words if w in ad_tokens)
        partial = sum(1 for w in stem_words if any(w in tok or tok in w for tok in ad_tokens))
        score   = exact + partial
        if score > best_score:
            best_score, best_name, best_cat = score, ad_name, cat

    return best_name, best_cat


# ---------------------------------------------------------------------------
# Vision pre-processing — extract visual keywords for semantic ad-name matching
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner=False, ttl=3600)
def _extract_image_keywords_cached(b64: str, mime: str, filename: str, api_key: str) -> Dict:
    """GPT-4o vision call (detail=low) — returns visual keywords for semantic matching."""
    _c = OpenAI(api_key=api_key)
    prompt = (
        "Look at this banner ad image. Extract keywords to match it to an ad name.\n"
        "Return ONLY valid JSON with exactly these keys:\n"
        "- product (string): main product shown (e.g. 'Ring', 'Necklace', 'Earrings', or '' if unclear)\n"
        "- promo (string): any promotion/occasion shown (e.g. 'Summer Sale', 'Kingsday', '20% Off', or '' if none)\n"
        "- colors (array of strings): 1–3 dominant colors (e.g. ['Gold', 'White'])\n"
        "- keywords (array of lowercase strings): all relevant tokens for matching "
        "(product type, promo name, colors, style, season, occasion — max 12 items)"
    )
    try:
        response = _c.chat.completions.create(
            model="gpt-4o",
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url",
                     "image_url": {"url": f"data:{mime};base64,{b64}", "detail": "low"}},
                ],
            }],
            max_tokens=200,
            response_format={"type": "json_object"},
        )
        return json.loads(response.choices[0].message.content)
    except Exception:
        return {"product": "", "promo": "", "colors": [], "keywords": []}


def _semantic_score(ad_name: str, kw_data: Dict) -> float:
    """Return 0–1: fraction of significant ad-name tokens that hit at least one image keyword."""
    keywords = kw_data.get("keywords", [])
    if not keywords:
        return 0.0
    ad_tokens = {t for t in re.sub(r"[_\-]+", " ", ad_name.lower()).split() if len(t) > 2}
    if not ad_tokens:
        return 0.0
    kw_lower = [k.lower() for k in keywords]
    hits = sum(1 for t in ad_tokens if any(t in k or k in t for k in kw_lower))
    return hits / len(ad_tokens)


def match_image_with_confidence(
    filename: str,
    img_keywords: Dict,
    perf_map: Dict[str, str],
) -> Tuple[str, str, float]:
    """
    Returns (matched_ad_name, category, confidence 0–1).

    Pass 1 — filename slug containment     → confidence 1.0
    Pass 2 — filename word-overlap         → confidence 0.50–0.85
    Pass 3 — semantic keyword match        → confidence = keyword score (0–1)
    Fallback — best word-overlap or stem   → confidence 0.0
    """
    stem       = os.path.splitext(filename)[0]
    stem_lower = stem.lower()
    stem_clean = re.sub(r"[_\-]+", " ", stem_lower).strip()

    # Pass 1: exact slug / substring containment → high confidence
    for ad_name, cat in perf_map.items():
        ad_slug  = ad_name.lower().replace(" ", "_")
        ad_lower = ad_name.lower()
        if (ad_slug in stem_lower or stem_lower in ad_slug
                or ad_lower in stem_clean or stem_clean in ad_lower):
            return ad_name, cat, 1.0

    # Pass 2: filename word-overlap score
    stem_words = [w for w in stem_clean.split() if len(w) > 2]
    best_fn_score, best_fn_name, best_fn_cat = 0, stem, "No Data"
    for ad_name, cat in perf_map.items():
        ad_lower  = ad_name.lower()
        ad_tokens = set(ad_lower.split())
        exact   = sum(2 for w in stem_words if w in ad_tokens)
        partial = sum(1 for w in stem_words if any(w in tok or tok in w for tok in ad_tokens))
        score   = exact + partial
        if score > best_fn_score:
            best_fn_score, best_fn_name, best_fn_cat = score, ad_name, cat
    fn_conf = min(0.50 + best_fn_score * 0.08, 0.85) if best_fn_score > 0 else 0.0

    # Pass 3: semantic keyword match (GPT-extracted visual keywords vs ad-name tokens)
    best_kw_score, best_kw_name, best_kw_cat = 0.0, stem, "No Data"
    for ad_name, cat in perf_map.items():
        score = _semantic_score(ad_name, img_keywords)
        if score > best_kw_score:
            best_kw_score, best_kw_name, best_kw_cat = score, ad_name, cat

    # Prefer the higher-confidence strategy
    if best_kw_score > fn_conf:
        return best_kw_name, best_kw_cat, best_kw_score
    elif fn_conf > 0:
        return best_fn_name, best_fn_cat, fn_conf
    else:
        return best_fn_name, best_fn_cat, 0.0


def _find_best_image_for_ad(ad_name: str, img_names: List[str]) -> Optional[str]:
    """
    Return the filename from img_names that best matches ad_name (no API calls).

    Pass 1 — slug / substring containment  → return immediately
    Pass 2 — word-overlap scoring          → return highest scorer (or None if score=0)
    """
    if not img_names:
        return None

    ad_lower = ad_name.lower()
    ad_slug  = ad_lower.replace(" ", "_")
    ad_clean = re.sub(r"[_\-]+", " ", ad_lower).strip()

    # Pass 1
    for fn in img_names:
        stem = re.sub(r"[_\-]+", " ", os.path.splitext(fn)[0].lower()).strip()
        if (ad_slug in fn.lower() or fn.lower() in ad_slug
                or ad_clean in stem or stem in ad_clean):
            return fn

    # Pass 2
    ad_tokens = {t for t in ad_clean.split() if len(t) > 2}
    best_score, best_fn = 0, None
    for fn in img_names:
        stem_tokens = set(re.sub(r"[_\-]+", " ", os.path.splitext(fn)[0].lower()).split())
        exact   = sum(2 for t in ad_tokens if t in stem_tokens)
        partial = sum(1 for t in ad_tokens if any(t in s or s in t for s in stem_tokens))
        score   = exact + partial
        if score > best_score:
            best_score, best_fn = score, fn

    return best_fn if best_score > 0 else None


# ---------------------------------------------------------------------------
# Shared helper: named BytesIO for reconstructed uploads
# ---------------------------------------------------------------------------

class _NamedBytesIO(BytesIO):
    """BytesIO that carries a .name attribute so it behaves like an uploaded file."""
    def __init__(self, data: bytes, name: str):
        super().__init__(data)
        self.name = name


# ═══════════════════════════════════════════════════════════════════════════
# PHASE 1 — Input (uploaders + welcome screen)
# ═══════════════════════════════════════════════════════════════════════════

if st.session_state.phase == "UPLOAD":
    with _main_area.container():
        # Recovery banner — shown when a refresh wiped an in-progress LANCERING
        if st.session_state.pop("_refresh_warning", False):
            st.warning(
                "⚠️ **Analyse onderbroken.** Je pagina is ververst tijdens de analyse — "
                "de sessiedata is verloren gegaan. Upload je bestanden opnieuw om verder te gaan.",
                icon="🔄",
            )

        st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
        col_csv, col_img = st.columns(2)

        with col_csv:
            st.subheader("📁 Campagne CSV")
            csv_file = st.file_uploader(
                "Exporteer vanuit Meta Ads Manager en upload hier",
                type=["csv"],
                label_visibility="collapsed",
            )
            if csv_file:
                st.success(f"✓ {csv_file.name}")

        with col_img:
            st.subheader("🖼️ Bannerafbeeldingen")
            uploaded_images = st.file_uploader(
                "Selecteer één of meerdere banners",
                type=["jpg", "jpeg", "png", "webp"],
                accept_multiple_files=True,
                label_visibility="collapsed",
            )
            if uploaded_images:
                st.success(f"✓ {len(uploaded_images)} afbeelding(en) geladen")

        st.divider()
        start_btn = st.button(
            "🚀 Analyse Starten",
            disabled=(csv_file is None),
            type="primary",
            use_container_width=True,
        )

        if not csv_file:
            st.markdown(
                "<div style='background:linear-gradient(135deg,#f5faf7 0%,#edfaf3 100%);"
                "border:1px solid #b2dcc5;border-radius:14px;padding:32px 36px;"
                "margin:20px 0;text-align:center'>"
                "<div style='font-size:1.4rem;font-weight:800;color:#00573C;"
                "font-family:Rubik,sans-serif;margin-bottom:10px'>Welkom bij de Meta Ads Analyser</div>"
                "<div style='font-size:0.92rem;color:#444;max-width:520px;"
                "margin:0 auto 20px auto;line-height:1.65'>"
                "Upload je Meta Ads CSV-export en bannerafbeeldingen om te starten. "
                "De AI classificeert je advertenties, analyseert de visuals en genereert "
                "direct 10 creatieve briefings op basis van de winnende patronen.</div>"
                "<div style='display:flex;justify-content:center;gap:16px;flex-wrap:wrap'>"
                "<div style='background:#fff;border:1px solid #d0ead7;border-radius:10px;padding:14px 18px;min-width:130px'>"
                "<div style='font-size:1.4rem'>📁</div>"
                "<div style='font-size:0.75rem;font-weight:700;color:#00573C;text-transform:uppercase;letter-spacing:0.05em;margin:6px 0 2px'>Stap 1</div>"
                "<div style='font-size:0.82rem;color:#555'>Upload je CSV</div></div>"
                "<div style='background:#fff;border:1px solid #d0ead7;border-radius:10px;padding:14px 18px;min-width:130px'>"
                "<div style='font-size:1.4rem'>🖼️</div>"
                "<div style='font-size:0.75rem;font-weight:700;color:#00573C;text-transform:uppercase;letter-spacing:0.05em;margin:6px 0 2px'>Stap 2</div>"
                "<div style='font-size:0.82rem;color:#555'>Upload je banners</div></div>"
                "<div style='background:#fff;border:1px solid #d0ead7;border-radius:10px;padding:14px 18px;min-width:130px'>"
                "<div style='font-size:1.4rem'>🚀</div>"
                "<div style='font-size:0.75rem;font-weight:700;color:#00573C;text-transform:uppercase;letter-spacing:0.05em;margin:6px 0 2px'>Stap 3</div>"
                "<div style='font-size:0.82rem;color:#555'>Start de analyse</div></div>"
                "<div style='background:#fff;border:1px solid #d0ead7;border-radius:10px;padding:14px 18px;min-width:130px'>"
                "<div style='font-size:1.4rem'>📄</div>"
                "<div style='font-size:0.75rem;font-weight:700;color:#00573C;text-transform:uppercase;letter-spacing:0.05em;margin:6px 0 2px'>Stap 4</div>"
                "<div style='font-size:0.82rem;color:#555'>Download het rapport</div></div>"
                "</div>"
                "<div style='margin-top:20px;font-size:0.78rem;color:#888'>"
                "Ondersteunt alle Meta Ads Manager exports — komma, puntkomma of tab-gescheiden</div>"
                "</div>",
                unsafe_allow_html=True,
            )

        if start_btn and csv_file:
            csv_file.seek(0)
            st.session_state["_csv_bytes"] = csv_file.read()
            st.session_state["_csv_name"]  = csv_file.name
            st.session_state["_imgs"] = []
            for _img in (uploaded_images or []):
                _img.seek(0)
                st.session_state["_imgs"].append({"name": _img.name, "data": _img.read()})
            # Clear any stale matches / selectbox keys from a previous run
            st.session_state.pop("final_matches", None)
            st.session_state.pop("results", None)
            for _k in [k for k in st.session_state if k.startswith("match_")]:
                del st.session_state[_k]
            st.session_state.phase = "MATCHER"
            st.rerun()

# ═══════════════════════════════════════════════════════════════════════════
# PHASE 1b — Pre-Flight Matcher
# ═══════════════════════════════════════════════════════════════════════════

elif st.session_state.phase == "MATCHER":

    csv_bytes = st.session_state.get("_csv_bytes", b"")
    csv_name  = st.session_state.get("_csv_name", "upload.csv")
    imgs_data = st.session_state.get("_imgs", [])

    with _main_area.container():
        # ── Header ────────────────────────────────────────────────────────────
        st.markdown(
            "<div style='background:linear-gradient(135deg,#00573C 0%,#33B784 100%);"
            "color:#fff;padding:20px 28px 16px;border-radius:12px;margin-bottom:20px'>"
            "<div style='font-size:1.3rem;font-weight:800;font-family:Rubik,sans-serif'>"
            "🛸 Pre-Flight Matcher</div>"
            "<div style='font-size:0.85rem;opacity:.88;margin-top:4px'>"
            "Koppel je banners aan de top-advertenties uit de CSV. "
            "De AI heeft een voorzet gemaakt op basis van je bestandsnamen — "
            "pas aan waar nodig en bevestig om de analyse te starten."
            "</div></div>",
            unsafe_allow_html=True,
        )

        # ── Load + classify CSV (no API calls) ────────────────────────────────
        try:
            _m_df = load_csv_from_upload(_NamedBytesIO(csv_bytes, csv_name))
            _m_df, _m_metric = classify_ads(_m_df)
        except Exception as _m_err:
            st.error(f"CSV kon niet worden ingeladen: {_m_err}")
            st.stop()

        _m_ad_col = find_column(_m_df, AD_NAME_COLUMNS)
        if not _m_ad_col:
            st.error(
                "Geen advertentienaam-kolom gevonden in de CSV. "
                "Controleer of je een originele Meta Ads Manager export gebruikt."
            )
            st.stop()

        # Build perf_map + metric value per ad
        _m_roas_col = find_column(_m_df, ROAS_COLUMNS_KPI)
        _m_spend_col = find_column(_m_df, SPEND_COLUMNS)
        _priority = {"High Performer": 0, "Average": 1, "Underperformer": 2, "No Data": 3}
        _m_seen: Dict[str, tuple] = {}
        for _, _row in _m_df.iterrows():
            _adn  = str(_row[_m_ad_col]).strip()
            _cat  = _row.get("performance_category", "No Data")
            _roas = float(_row[_m_roas_col]) if _m_roas_col and pd.notna(_row[_m_roas_col]) else 0.0
            if _adn not in _m_seen or _roas > _m_seen[_adn][1]:
                _m_seen[_adn] = (_cat, _roas)

        _sorted_ads = sorted(
            _m_seen.items(),
            key=lambda x: (_priority.get(x[1][0], 3), -x[1][1]),
        )

        # Cap at top 10 (or number of images, whichever is larger up to 10)
        _show_n = min(10, max(len(imgs_data) if imgs_data else 5, 5))
        _top_ads = _sorted_ads[:_show_n]

        _img_names     = [img["name"] for img in imgs_data]
        _img_bytes_map = {img["name"]: img["data"] for img in imgs_data}

        # Smart defaults: filename-based matching (no API)
        _defaults: Dict[int, str] = {}
        _assigned: set = set()
        for _idx, (_adn, _) in enumerate(_top_ads):
            _remaining = [n for n in _img_names if n not in _assigned]
            _best = _find_best_image_for_ad(_adn, _remaining)
            _defaults[_idx] = _best or "(geen afbeelding)"
            if _best:
                _assigned.add(_best)

        # ── Performance badge helper ───────────────────────────────────────────
        _BADGE = {
            "High Performer": ("#d4edda", "#155724", "🏆 Top"),
            "Average":        ("#fff3cd", "#856404", "➡️ Gem"),
            "Underperformer": ("#f8d7da", "#721c24", "⚠️ Laag"),
            "No Data":        ("#e2e3e5", "#383d41", "❓ N/A"),
        }

        # ── Column headers ────────────────────────────────────────────────────
        _hc1, _hc2, _hc3 = st.columns([3, 3, 2])
        _hc1.markdown(
            "<div style='font-size:0.72rem;font-weight:700;color:#00573C;"
            "text-transform:uppercase;letter-spacing:0.06em;padding-bottom:4px'>"
            "Advertentienaam</div>",
            unsafe_allow_html=True,
        )
        _hc2.markdown(
            "<div style='font-size:0.72rem;font-weight:700;color:#00573C;"
            "text-transform:uppercase;letter-spacing:0.06em;padding-bottom:4px'>"
            "Gekoppelde afbeelding</div>",
            unsafe_allow_html=True,
        )
        _hc3.markdown(
            "<div style='font-size:0.72rem;font-weight:700;color:#00573C;"
            "text-transform:uppercase;letter-spacing:0.06em;padding-bottom:4px'>"
            "Voorbeeld</div>",
            unsafe_allow_html=True,
        )

        st.markdown(
            "<hr style='margin:0 0 8px 0;border-color:#e0ede6'>",
            unsafe_allow_html=True,
        )

        # ── Matcher rows ──────────────────────────────────────────────────────
        _selectbox_opts = ["(geen afbeelding)"] + _img_names

        for _idx, (_adn, (_cat, _roas)) in enumerate(_top_ads):
            _bg, _tc, _bl = _BADGE.get(_cat, _BADGE["No Data"])
            _default_sel  = _defaults.get(_idx, "(geen afbeelding)")
            _default_idx  = (
                _selectbox_opts.index(_default_sel)
                if _default_sel in _selectbox_opts else 0
            )

            _rc1, _rc2, _rc3 = st.columns([3, 3, 2])

            with _rc1:
                _roas_str = f"ROAS {_roas:.2f}x" if _roas > 0 else ""
                st.markdown(
                    f"<div style='padding:6px 0'>"
                    f"<span style='background:{_bg};color:{_tc};padding:2px 8px;"
                    f"border-radius:12px;font-size:0.7rem;font-weight:700;"
                    f"margin-right:6px'>{_bl}</span>"
                    f"<span style='font-size:0.88rem;font-weight:600;color:#111'>"
                    f"{_adn}</span>"
                    f"<br><span style='font-size:0.75rem;color:#888'>{_roas_str}</span>"
                    f"</div>",
                    unsafe_allow_html=True,
                )

            with _rc2:
                st.selectbox(
                    "Afbeelding",
                    options=_selectbox_opts,
                    index=_default_idx,
                    key=f"match_{_idx}",
                    label_visibility="collapsed",
                )

            with _rc3:
                # Dynamic thumbnail — reads current selectbox value from session state
                _cur_sel = st.session_state.get(f"match_{_idx}", _default_sel)
                if _cur_sel != "(geen afbeelding)" and _cur_sel in _img_bytes_map:
                    st.image(_img_bytes_map[_cur_sel], use_container_width=True)
                else:
                    st.markdown(
                        "<div style='background:#f5f5f5;border:1px dashed #ccc;"
                        "border-radius:6px;height:52px;display:flex;align-items:center;"
                        "justify-content:center;font-size:0.75rem;color:#aaa'>geen</div>",
                        unsafe_allow_html=True,
                    )

            st.markdown(
                "<hr style='margin:2px 0;border-color:#f0f0f0'>",
                unsafe_allow_html=True,
            )

        # ── Images not yet assigned (info row) ────────────────────────────────
        _all_selected = {
            st.session_state.get(f"match_{_i}", _defaults.get(_i, "(geen afbeelding)"))
            for _i in range(len(_top_ads))
        } - {"(geen afbeelding)"}
        _unassigned = [n for n in _img_names if n not in _all_selected]
        if _unassigned:
            st.markdown(
                f"<div style='background:#fff8e1;border:1px solid #f59e0b;"
                f"border-radius:8px;padding:10px 14px;font-size:0.82rem;color:#7c5a00;"
                f"margin-top:8px'>ℹ️ <strong>{len(_unassigned)} afbeelding(en) "
                f"nog niet gekoppeld:</strong> "
                f"{', '.join(_unassigned)}</div>",
                unsafe_allow_html=True,
            )

        st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)

        if not api_key:
            st.warning(
                "⚠️ Voer eerst je OpenAI API-sleutel in via **Developer Settings** "
                "in de zijbalk voor je de analyse start.",
                icon="🔑",
            )

        # ── Confirm button ─────────────────────────────────────────────────────
        _confirm_btn = st.button(
            "✅ Matches bevestigen & Analyse starten",
            type="primary",
            use_container_width=True,
            disabled=(not api_key),
        )

        if _confirm_btn:
            _final: Dict[str, str] = {}
            for _idx, (_adn, _) in enumerate(_top_ads):
                _sel = st.session_state.get(f"match_{_idx}", "(geen afbeelding)")
                if _sel != "(geen afbeelding)":
                    _final[_adn] = _sel
            st.session_state["final_matches"] = _final
            st.session_state.phase = "LANCERING"
            st.rerun()


# ═══════════════════════════════════════════════════════════════════════════
# PHASE 2 — Rocket animation + Analysis pipeline
# ═══════════════════════════════════════════════════════════════════════════

elif st.session_state.phase == "LANCERING":

    # ── JS: warn the user before they accidentally close/refresh ──────────────
    components.html(
        "<script>window.addEventListener('beforeunload',function(e){"
        "e.preventDefault();"
        "e.returnValue='Weet je zeker dat je de pagina wilt verlaten? Je analyse gaat verloren.';"
        "return e.returnValue;});</script>",
        height=0,
    )

    # ── Guard: file data gone (page was refreshed) → recovery UI ─────────────
    if not st.session_state.get("_csv_bytes"):
        with _main_area.container():
            st.markdown(_ROCKET_ANIM_HTML, unsafe_allow_html=True)
            st.warning(
                "⚠️ **Verbinding verbroken.** De pagina is ververst tijdens de analyse. "
                "Klik hieronder om opnieuw te uploaden.",
                icon="🔄",
            )
            if st.button("← Terug naar upload", type="primary", use_container_width=True):
                st.session_state.phase = "UPLOAD"
                st.rerun()
        st.stop()

    if not api_key:
        with _main_area.container():
            st.error("Voer eerst je OpenAI API-sleutel in via Developer Settings in de zijbalk.")
        st.session_state.phase = "UPLOAD"
        st.stop()

    # ── Render rocket + progress bar ──────────────────────────────────────────
    with _main_area.container():
        st.markdown(_ROCKET_ANIM_HTML, unsafe_allow_html=True)
        st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
        _prog_bar = st.progress(0, text="🚀 Campagnedata inladen...")

    try:
        results = {}

        csv_bytes = st.session_state.get("_csv_bytes", b"")
        csv_name  = st.session_state.get("_csv_name", "upload.csv")
        imgs_data = st.session_state.get("_imgs", [])

        # Stap 1: CSV classificeren
        _prog_bar.progress(6, text="📊 Campagne CSV analyseren...")
        df = load_csv_from_upload(_NamedBytesIO(csv_bytes, csv_name))
        df, metric_used = classify_ads(df)
        results["df"] = df
        results["metric_used"] = metric_used

        # Stap 2: Visuele analyse — gebruik bevestigde matches uit MATCHER
        analysed: List[Dict] = []
        ad_col = find_column(df, AD_NAME_COLUMNS)
        perf_map: Dict[str, str] = {}
        if ad_col:
            for _, row in df.iterrows():
                perf_map[str(row[ad_col]).strip()] = row.get("performance_category", "No Data")

        image_bytes_map: Dict[str, bytes]   = {}
        image_keywords_map: Dict[str, Dict] = {}
        match_confidences: Dict[str, float] = {}
        matched_count = 0

        # Invert final_matches (ad_name → filename) to filename → ad_name
        final_matches   = st.session_state.get("final_matches", {})
        filename_to_ad  = {v: k for k, v in final_matches.items()}

        if imgs_data:
            total_imgs       = len(imgs_data)
            confirmed_fnames = set(filename_to_ad.keys())
            unconfirmed      = [img for img in imgs_data if img["name"] not in confirmed_fnames]

            # Sub-stap 2a: keyword-extractie ALLEEN voor niet-bevestigde afbeeldingen
            for i, img_info in enumerate(unconfirmed):
                pct = 8 + int(18 * (i + 1) / max(len(unconfirmed), 1))
                _prog_bar.progress(
                    pct,
                    text=f"👁️ Visuele keywords extraheren {i + 1}/{len(unconfirmed)}: {img_info['name']}",
                )
                ext  = os.path.splitext(img_info["name"])[1].lower()
                mime = "image/jpeg" if ext in (".jpg", ".jpeg") else f"image/{ext.lstrip('.')}"
                b64  = base64.b64encode(img_info["data"]).decode("utf-8")
                try:
                    kw_data = _extract_image_keywords_cached(b64, mime, img_info["name"], api_key)
                except Exception:
                    kw_data = {"product": "", "promo": "", "colors": [], "keywords": []}
                image_keywords_map[img_info["name"]] = kw_data
                image_bytes_map[img_info["name"]]    = img_info["data"]

            # Sub-stap 2b: visuele beschrijving voor alle afbeeldingen
            for i, img_info in enumerate(imgs_data):
                pct = 26 + int(34 * (i + 1) / total_imgs)
                _prog_bar.progress(
                    pct,
                    text=f"🔬 Banner doorlichten {i + 1}/{total_imgs}: {img_info['name']}",
                )
                fn = img_info["name"]
                image_bytes_map[fn] = img_info["data"]

                if fn in filename_to_ad:
                    # Bevestigde koppeling uit MATCHER — altijd vertrouwbaar
                    name       = filename_to_ad[fn]
                    cat        = perf_map.get(name, "No Data")
                    confidence = 1.0
                    img_kw     = {}
                else:
                    # Niet bevestigd — gebruik semantische koppeling als fallback
                    img_kw = image_keywords_map.get(fn, {})
                    if perf_map:
                        name, cat, confidence = match_image_with_confidence(fn, img_kw, perf_map)
                    else:
                        name, cat, confidence = fn, "No Data", 0.0

                match_confidences[fn] = confidence
                if perf_map and confidence > 0 and name in perf_map:
                    matched_count += 1

                ext  = os.path.splitext(fn)[1].lower()
                mime = "image/jpeg" if ext in (".jpg", ".jpeg") else f"image/{ext.lstrip('.')}"
                b64  = base64.b64encode(img_info["data"]).decode("utf-8")
                try:
                    description = _describe_image_cached(b64, mime, name, api_key)
                except Exception as e:
                    description = f"_(Fout bij analyseren: {e})_"
                analysed.append({
                    "name":         name,
                    "filename":     fn,
                    "category":     cat,
                    "description":  description,
                    "confidence":   confidence,
                    "img_keywords": img_kw,
                })

        results["analysed"]           = analysed
        results["match_count"]        = matched_count
        results["total_images"]       = len(imgs_data)
        results["image_bytes_map"]    = image_bytes_map
        results["image_keywords_map"] = image_keywords_map
        results["match_confidences"]  = match_confidences
        results["perf_map"]           = perf_map

        # Stap 3: Strategische vergelijking — cached per context hash
        high   = [a for a in analysed if a["category"] == "High Performer"]
        avg    = [a for a in analysed if a["category"] == "Average"]
        under  = [a for a in analysed if a["category"] == "Underperformer"]
        no_dat = [a for a in analysed if a["category"] not in
                  ("High Performer", "Average", "Underperformer")]

        analysis_text = ""
        if analysed:
            _prog_bar.progress(65, text="🧠 Winnende patronen identificeren...")
            try:
                context = _build_creatives_context(high, under + avg, no_dat)
                analysis_text = _compare_creatives_cached(context, api_key)
            except Exception as e:
                analysis_text = f"_(Kon vergelijking niet genereren: {e})_"
        results["analysis_text"] = analysis_text

        report_lines = ["# Meta Ads Visueel Analyserapport", ""]
        for ad in analysed:
            report_lines += [
                f"## {ad['name']} ({CATEGORY_NL.get(ad['category'], ad['category'])})",
                ad["description"], "",
            ]
        if analysis_text:
            report_lines += ["## Strategische Analyse", analysis_text]
        full_report = "\n".join(report_lines)
        results["full_report"] = full_report

        # Stap 4: Master Prompts engineeren — cached per (report × filenames)
        concepts: List[Dict] = []
        img_filenames = [a["filename"] for a in analysed] if analysed else []
        _prog_bar.progress(78, text="✍️ Master Prompts engineeren...")
        try:
            concepts = _generate_concepts_cached(
                full_report, tuple(img_filenames), api_key
            )
        except Exception:
            pass  # surfaced as warning in Phase 3
        results["concepts"] = concepts

        concepts_md_lines = [
            "# Creatieve Briefings — Nieuwe Advertentieconcepten",
            f"_Gegenereerd: {datetime.now().strftime('%Y-%m-%d %H:%M')}_",
            "", "> Gebaseerd op de winnende patronen uit de campagneanalyse.", "",
        ]
        for c in concepts:
            concepts_md_lines += [
                "---",
                f"### Concept {c.get('nummer', '?')}: {c.get('titel', '')}",
                f"**Prioriteit:** {c.get('prioriteit', '—')}",
                "",
                f"**Hook:** {c.get('hook', '—')}",
                "",
                f"**Primary Text:** {c.get('primary_text', '—')}",
                "",
                f"**Headline:** {c.get('headline', '—')}",
                "",
                f"**Visuele Omschrijving:** {c.get('visuele_omschrijving', '—')}",
                "",
                f"**Referentie-afbeelding:** {c.get('referentie_afbeelding', '—')}",
                "",
                f"**Rationale:** {c.get('rationale', '—')}",
                "",
                "**Master Prompt (Freepik / Nano Banana Pro):**",
                "", "```", c.get("master_prompt", "—"), "```", "",
            ]
        results["concepts_md"] = "\n".join(concepts_md_lines)

        # Stap 5: PDF genereren en cachen
        _prog_bar.progress(93, text="📄 Rapport samenstellen...")
        try:
            results["pdf_bytes"] = generate_pdf(
                df=df,
                metric_used=metric_used,
                analysed=analysed,
                analysis_text=analysis_text,
                concepts=concepts,
            )
        except Exception:
            results["pdf_bytes"] = b""

        _prog_bar.progress(100, text="🌕 Geland op de maan! Dashboard laden...")
        st.session_state["results"] = results
        st.session_state.phase = "RESULTS"
        st.rerun()

    except Exception as _pipeline_err:
        with _main_area.container():
            st.error(
                "Oeps! Er is een fout opgetreden. "
                "Controleer je API-sleutel en upload een geldige Meta Ads CSV."
            )
            with st.expander("Technische details (voor de consultant)", expanded=False):
                st.exception(_pipeline_err)
        st.session_state.phase = "UPLOAD"

# ═══════════════════════════════════════════════════════════════════════════
# PHASE 3 — Results Dashboard
# ═══════════════════════════════════════════════════════════════════════════

elif st.session_state.phase == "RESULTS":
    # Warn before accidental close/refresh (user's work is only in session memory)
    components.html(
        "<script>window.addEventListener('beforeunload',function(e){"
        "e.preventDefault();"
        "e.returnValue='Weet je zeker dat je de pagina wilt verlaten? Je analyse gaat verloren.';"
        "return e.returnValue;});</script>",
        height=0,
    )

    r = st.session_state["results"]
    df: pd.DataFrame          = r["df"]
    metric_used: str          = r["metric_used"]
    analysed: List[Dict]      = r["analysed"]
    concepts: List[Dict]      = r.get("concepts", [])
    image_bytes_map: Dict[str, bytes] = r.get("image_bytes_map", {})

    with _main_area.container():
        mc = r.get("match_count", 0)
        ti = r.get("total_images", 0)
        uncertain_count = sum(
            1 for a in r.get("analysed", []) if a.get("confidence", 1.0) < 0.8
        )
        if ti > 0:
            icon = "✅" if (mc == ti and uncertain_count == 0) else ("⚠️" if mc > 0 else "❌")
            extra = (
                f" &nbsp;·&nbsp; <span style='color:#e63946;font-weight:600'>"
                f"{uncertain_count} onzeker — zie ↓ Match advertenties aan banners</span>"
                if uncertain_count > 0 else ""
            )
            st.markdown(
                f"<div style='background:#edfaf3;border:1px solid #33B784;border-radius:8px;"
                f"padding:8px 16px;margin-bottom:8px;font-size:0.9rem'>"
                f"{icon} <strong>Koppelstatus:</strong> {mc} van de {ti} afbeeldingen "
                f"gekoppeld aan de CSV-data.{extra}</div>",
                unsafe_allow_html=True,
            )

        tab1, tab2, tab3 = st.tabs([
            "📈 Performance Overzicht",
            "🔍 Visuele Analyse",
            "💡 Creatieve Briefings",
        ])

    # ---- Tab 1: Performance overzicht ---------------------------------------
    with tab1:
        st.subheader("Performance Overzicht")
        st.caption(f"Gebruikte metric: {metric_used}")

        counts = df["performance_category"].value_counts()
        total = len(df)

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Totaal advertenties", total)
        m2.metric("🏆 Top Presteerder", int(counts.get("High Performer", 0)))
        m3.metric("➡️ Gemiddeld", int(counts.get("Average", 0)))
        m4.metric("⚠️ Onderpresteerder", int(counts.get("Underperformer", 0)))

        # Extended KPI cards: Spend / ROAS / Beste Ad
        # classify_ads already cleaned ROAS/Spend/Revenue columns to floats;
        # _parse_num_col is a safe fallback for any column that may still be a string.
        def _parse_num_col(series: pd.Series) -> pd.Series:
            # If the column is already numeric, pass through directly
            if pd.api.types.is_numeric_dtype(series):
                return series
            # Otherwise apply the same context-aware Dutch/English parser
            return series.apply(clean_dutch_number)

        spend_col     = find_column(df, SPEND_COLUMNS)
        roas_col_kpi  = find_column(df, ROAS_COLUMNS_KPI)
        ad_col_kpi    = find_column(df, AD_NAME_COLUMNS)

        def _kpi_card(col, title: str, value: str) -> None:
            col.markdown(
                f"<div style='background:#f5faf7;border:1px solid #b2dcc5;border-radius:10px;"
                f"padding:14px 12px;text-align:center;height:90px;display:flex;"
                f"flex-direction:column;justify-content:center'>"
                f"<div style='font-size:0.72rem;font-weight:700;color:#00573C;"
                f"text-transform:uppercase;letter-spacing:0.05em;margin-bottom:6px'>{title}</div>"
                f"<div style='font-size:1.5rem;font-weight:800;color:#33B784;"
                f"font-family:Rubik,sans-serif;line-height:1.1'>{value}</div>"
                f"</div>",
                unsafe_allow_html=True,
            )

        k1, k2, k3 = st.columns(3)

        # Totaal Spend
        total_spend_num = None
        total_spend_str = "—"
        if spend_col:
            total_spend_num = _parse_num_col(df[spend_col]).sum()
            if pd.notna(total_spend_num):
                total_spend_str = fmt_nl(total_spend_num, prefix="EUR ")
        _kpi_card(k1, "Totaal Spend", total_spend_str)

        # Gemiddelde ROAS = Total Revenue / Total Spend (weighted, not simple mean)
        avg_roas_str = "—"
        if roas_col_kpi and spend_col and total_spend_num and total_spend_num > 0:
            roas_num  = _parse_num_col(df[roas_col_kpi])
            spend_num = _parse_num_col(df[spend_col])
            total_revenue = (roas_num * spend_num).sum()
            if pd.notna(total_revenue):
                avg_roas_str = fmt_nl(total_revenue / total_spend_num) + "x"
        elif roas_col_kpi and not spend_col:
            val = _parse_num_col(df[roas_col_kpi]).mean()
            if pd.notna(val):
                avg_roas_str = fmt_nl(val) + "x"
        _kpi_card(k2, "Gemiddelde ROAS", avg_roas_str)

        # Beste Advertentie — highest ROAS among ads that have actual spend
        best_ad_str = "—"
        if ad_col_kpi and roas_col_kpi:
            hp = df[df["performance_category"] == "High Performer"].copy()
            if not hp.empty:
                hp_roas = _parse_num_col(hp[roas_col_kpi])
                if spend_col:
                    hp_spend = _parse_num_col(hp[spend_col])
                    hp_roas = hp_roas.where(hp_spend > 0)  # ignore zero-spend rows
                best_idx = hp_roas.idxmax()
                if pd.notna(best_idx):
                    name_val = str(hp.loc[best_idx, ad_col_kpi])
                    best_ad_str = name_val[:28] + "…" if len(name_val) > 28 else name_val
        _kpi_card(k3, "Beste Advertentie", best_ad_str)

        st.divider()

        display_cols = [c for c in df.columns if c != "performance_category"] + ["performance_category"]
        st.dataframe(
            df[display_cols].style.apply(
                lambda row: [
                    colours_map.get(row["performance_category"], "") if c == "performance_category" else ""
                    for c in display_cols
                ],
                axis=1,
            ),
            use_container_width=True,
            hide_index=True,
        )

    # ---- Tab 2: Visuele analyse ---------------------------------------------
    with tab2:
        if not analysed:
            st.info("Geen afbeeldingen geüpload — upload banners om de visuele analyse te zien.")
        else:
            st.subheader("Visuele Analyse per Advertentie")

            tier_order = [
                ("🏆 Top Presteerders", "High Performer"),
                ("➡️ Gemiddeld", "Average"),
                ("⚠️ Onderpresteerders", "Underperformer"),
                ("❓ Geen prestatiedata", "No Data"),
            ]
            for tier_label, tier_key in tier_order:
                tier_ads = [a for a in analysed if a["category"] == tier_key]
                if not tier_ads:
                    continue
                st.markdown(f"### {tier_label}")
                for ad in tier_ads:
                    with st.expander(f"**{ad['name']}** — `{ad['filename']}`"):
                        st.markdown(ad["description"])

            st.divider()
            st.subheader("Strategische Vergelijking")

            analysis_txt = r.get("analysis_text", "")

            # ── Recommendation matrix ────────────────────────────────────────
            recs = parse_recommendations(analysis_txt)
            if recs:
                st.markdown("#### Aanbevelingen")
                cols_matrix = st.columns(3)
                groups = {"Hoog": [], "Middel": [], "Laag": []}
                for rec in recs:
                    groups.get(rec["prioriteit"], groups["Laag"]).append(rec)

                for col_widget, (prio_key, group_recs) in zip(
                    cols_matrix, MATRIX_GROUPS.items()
                ):
                    cfg = MATRIX_GROUPS[prio_key]
                    icon = "●" if prio_key == "Hoog" else ("◆" if prio_key == "Middel" else "▲")
                    with col_widget:
                        # Header badge
                        st.markdown(
                            f"<div style='background:{cfg['bg']};color:{cfg['color']};"
                            f"padding:8px 12px;border-radius:8px 8px 0 0;"
                            f"font-weight:700;font-size:0.82rem;letter-spacing:0.05em'>"
                            f"{icon} {cfg['label'].upper()}</div>",
                            unsafe_allow_html=True,
                        )
                        if groups[prio_key]:
                            for rec in groups[prio_key]:
                                st.markdown(
                                    f"<div style='border:2px solid {cfg['bg']};border-top:none;"
                                    f"border-radius:0 0 8px 8px;padding:10px 12px;"
                                    f"margin-bottom:6px;background:#ffffff'>"
                                    f"<div style='font-size:0.88rem;font-weight:600;"
                                    f"color:#111111;margin-bottom:5px'>{rec['actie']}</div>"
                                    f"<div style='font-size:0.80rem;color:#444444;"
                                    f"line-height:1.45'>{rec['impact']}</div>"
                                    f"</div>",
                                    unsafe_allow_html=True,
                                )
                        else:
                            st.markdown(
                                f"<div style='border:2px solid {cfg['bg']};border-top:none;"
                                f"border-radius:0 0 8px 8px;padding:10px 12px;"
                                f"background:#ffffff;font-size:0.85rem;color:#888'>—</div>",
                                unsafe_allow_html=True,
                            )
                st.divider()

            # ── Full analysis text ───────────────────────────────────────────
            with st.expander("📄 Volledige strategische analyse", expanded=not bool(recs)):
                st.markdown(analysis_txt if analysis_txt else "_(geen analyse beschikbaar)_")

            st.divider()
            st.download_button(
                "⬇️ Download analyserapport (Markdown)",
                data=r["full_report"].encode("utf-8"),
                file_name="analyserapport.md",
                mime="text/markdown",
            )

    # ---- Tab 3: Creatieve briefings -----------------------------------------
    with tab3:
        st.subheader("🚀 Strategische Nieuwe Concepten")
        st.markdown(
            "**Hoe gebruik je deze briefings?**  "
            "Elke kaart toont links de referentie-afbeelding die de AI heeft geselecteerd. "
            "Kopieer de **Master Prompt** (het XML-blok), ga naar "
            "**Nano Banana Pro** of **Freepik**, upload de referentie-afbeelding, "
            "plak de prompt en genereer je nieuwe banner. "
            "De prompt beschrijft het product visueel zodat de generator weet welk object te integreren."
        )

        if not concepts:
            st.warning("Geen concepten beschikbaar. Controleer je API-sleutel en probeer opnieuw.")
        else:
            # Priority summary metrics
            prio_counts = {}
            for c in concepts:
                p = c.get("prioriteit", "Onbekend")
                prio_counts[p] = prio_counts.get(p, 0) + 1

            p1, p2, p3 = st.columns(3)
            p1.metric("🔴 Hoge Prioriteit", prio_counts.get("Hoog", 0))
            p2.metric("🔵 Middel Prioriteit", prio_counts.get("Middel", 0))
            p3.metric("⚪ Lage Prioriteit",  prio_counts.get("Laag", 0))

            st.divider()

            for concept in concepts:
                nummer        = concept.get("nummer", "?")
                titel         = concept.get("titel", "Zonder titel")
                prioriteit    = concept.get("prioriteit", "—")
                rationale     = concept.get("rationale", concept.get("verwachte_impact", "—"))
                hook          = concept.get("hook", "—")
                body          = concept.get("primary_text", "—")
                headline      = concept.get("headline", "—")
                visual        = concept.get("visuele_omschrijving", "—")
                master_prompt = concept.get("master_prompt", "")
                ref_img       = concept.get("referentie_afbeelding", "")
                ref_is_file   = bool(ref_img and ref_img.lower() != "geen")

                prio_cfg              = PRIORITY_COLOURS.get(prioriteit, ("#6c757d", "#fff", prioriteit))
                bg_col, txt_col, badge_label = prio_cfg

                with st.container(border=True):
                    # ── Card header ────────────────────────────────────────────
                    h_left, h_right = st.columns([3, 1])
                    with h_left:
                        st.markdown(f"### 💡 {titel}")
                        st.caption(f"Concept {nummer}")
                    with h_right:
                        st.markdown(
                            f"<div style='text-align:right;padding-top:8px'>"
                            f"<span style='background:{bg_col};color:{txt_col};"
                            f"padding:5px 14px;border-radius:20px;font-size:0.78rem;"
                            f"font-weight:700;letter-spacing:0.05em;display:inline-block'>"
                            f"● {prioriteit.upper()}</span><br>"
                            f"<span style='font-size:0.72rem;color:{bg_col};"
                            f"font-weight:600;margin-top:2px;display:inline-block'>"
                            f"{badge_label}</span></div>",
                            unsafe_allow_html=True,
                        )

                    # ── Rationale ──────────────────────────────────────────────
                    st.markdown(
                        f"<div style='background:#f5faf7;border-left:4px solid #00573C;"
                        f"padding:10px 14px;border-radius:4px;margin:8px 0 14px 0;"
                        f"font-size:0.9rem;color:#1a3a2a'>"
                        f"<strong>Rationale:</strong> {rationale}</div>",
                        unsafe_allow_html=True,
                    )

                    # ── Hook ────────────────────────────────────────────────────
                    st.markdown(
                        f"<div style='background:#fff8e1;border-radius:8px;"
                        f"padding:12px 16px;margin-bottom:10px'>"
                        f"<span style='font-size:0.7rem;font-weight:700;color:#b7791f;"
                        f"letter-spacing:0.07em;text-transform:uppercase'>Hook</span><br>"
                        f"<span style='font-size:1.05rem;font-weight:600;color:#1a1a1a'>{hook}</span></div>",
                        unsafe_allow_html=True,
                    )

                    # ── Primary Text ────────────────────────────────────────────
                    st.markdown(
                        f"<div style='background:#f8f9fa;border-radius:8px;"
                        f"padding:12px 16px;margin-bottom:4px'>"
                        f"<span style='font-size:0.7rem;font-weight:700;color:#495057;"
                        f"letter-spacing:0.07em;text-transform:uppercase'>Primary Text</span><br>"
                        f"<span style='font-size:0.92rem;color:#333;line-height:1.55'>{body}</span></div>",
                        unsafe_allow_html=True,
                    )
                    clipboard_button(body, key=f"pt_{nummer}")
                    st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)

                    # ── Headline ────────────────────────────────────────────────
                    st.markdown(
                        f"<div style='background:#e8f4fd;border-radius:8px;"
                        f"padding:12px 16px;margin-bottom:4px'>"
                        f"<span style='font-size:0.7rem;font-weight:700;color:#1a6ca8;"
                        f"letter-spacing:0.07em;text-transform:uppercase'>Headline</span><br>"
                        f"<span style='font-size:1.05rem;font-weight:700;color:#1a1a1a'>{headline}</span></div>",
                        unsafe_allow_html=True,
                    )
                    clipboard_button(headline, key=f"hl_{nummer}")
                    st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)

                    # ── Art Direction (collapsible) ─────────────────────────────
                    with st.expander("🎨 Art Direction Briefing"):
                        st.markdown(visual)

                    st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)

                    # ── Footer: Master Prompt (left) + Reference popover (right) ─
                    col_prompt, col_ref = st.columns([2, 1])

                    with col_prompt:
                        st.markdown(
                            "<div style='font-size:0.7rem;font-weight:700;color:#00573C;"
                            "letter-spacing:0.07em;text-transform:uppercase;margin-bottom:4px'>"
                            "🎨 Master Prompt — Nano Banana Pro / Freepik</div>",
                            unsafe_allow_html=True,
                        )
                        if master_prompt:
                            st.code(master_prompt, language="xml")
                        st.caption(
                            "Copy-paste in Nano Banana Pro of Freepik. "
                            "Upload daarna de referentie-afbeelding →"
                        )

                    with col_ref:
                        st.markdown(
                            "<div style='font-size:0.7rem;font-weight:700;color:#00573C;"
                            "letter-spacing:0.07em;text-transform:uppercase;margin-bottom:8px'>"
                            "📌 Referentie</div>",
                            unsafe_allow_html=True,
                        )
                        if ref_is_file:
                            thumb_bytes = image_bytes_map.get(ref_img)
                            with st.popover("🖼️ Zie referentieafbeelding", use_container_width=True):
                                if thumb_bytes:
                                    st.image(thumb_bytes, use_container_width=True)
                                st.caption(f"Upload dit bestand: **{ref_img}**")
                        else:
                            st.markdown(
                                "<div style='background:#f8f9fa;border:1px dashed #ced4da;"
                                "border-radius:8px;padding:18px 10px;font-size:0.8rem;"
                                "color:#6c757d;text-align:center'>"
                                "🖼️<br><em>Geen referentie</em></div>",
                                unsafe_allow_html=True,
                            )

                st.divider()

    # ── Manual Override: handmatig koppelen van onzekere banners ────────────────
    _perf_map  = r.get("perf_map", {})
    _uncertain = [a for a in analysed if a.get("confidence", 1.0) < 0.8]

    if _uncertain and _perf_map:
        st.markdown(
            "<div style='height:8px'></div>",
            unsafe_allow_html=True,
        )
        with st.expander("🔗 Match advertenties aan banners", expanded=True):
            st.markdown(
                "<div style='font-size:0.9rem;color:#444;margin-bottom:14px'>"
                "De AI was onzeker over onderstaande koppelingen (onder 80% zekerheid). "
                "Selecteer de juiste advertentienaam per banner om de creatieve briefings te verbeteren."
                "</div>",
                unsafe_allow_html=True,
            )
            ad_name_opts = list(_perf_map.keys())

            for _ad_item in _uncertain:
                _fname = _ad_item["filename"]
                _cur   = _ad_item["name"]
                _conf  = int(_ad_item.get("confidence", 0) * 100)
                _kw    = _ad_item.get("img_keywords", {})
                kw_tags = ", ".join(_kw.get("keywords", [])[:8])

                oc1, oc2, oc3 = st.columns([1, 2, 3])
                with oc1:
                    _thumb = image_bytes_map.get(_fname)
                    if _thumb:
                        st.image(_thumb, width=90)
                    short_name = (_fname[:28] + "…") if len(_fname) > 28 else _fname
                    st.caption(short_name)
                with oc2:
                    st.markdown(
                        f"<div style='font-size:0.78rem;color:#6c757d;margin-bottom:6px'>"
                        f"Visueel: <em>{kw_tags or '—'}</em></div>"
                        f"<div style='font-size:0.82rem;color:#555'>"
                        f"Huidig: <strong>{_cur}</strong><br>"
                        f"<span style='color:#e63946'>{_conf}% zekerheid</span></div>",
                        unsafe_allow_html=True,
                    )
                with oc3:
                    default_idx = (
                        ad_name_opts.index(_cur) + 1
                        if _cur in ad_name_opts else 0
                    )
                    st.selectbox(
                        "Advertentienaam",
                        options=["(ongewijzigd)"] + ad_name_opts,
                        index=default_idx,
                        key=f"override_{_fname}",
                        label_visibility="collapsed",
                    )

            st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)

            if st.button(
                "✅ Toepassen & Briefings Opnieuw Genereren",
                type="primary",
                use_container_width=True,
            ):
                _updated = False
                for _ad_item in _uncertain:
                    _fname    = _ad_item["filename"]
                    _new_name = st.session_state.get(f"override_{_fname}", "(ongewijzigd)")
                    if _new_name != "(ongewijzigd)" and _new_name in _perf_map:
                        for _i, _a in enumerate(analysed):
                            if _a["filename"] == _fname:
                                analysed[_i]["name"]       = _new_name
                                analysed[_i]["category"]   = _perf_map[_new_name]
                                analysed[_i]["confidence"] = 1.0
                                _updated = True
                                break

                if _updated:
                    # Rebuild full report with corrected ad-name mappings
                    _rpt = ["# Meta Ads Visueel Analyserapport", ""]
                    for _a in analysed:
                        _rpt += [
                            f"## {_a['name']} ({CATEGORY_NL.get(_a['category'], _a['category'])})",
                            _a["description"], "",
                        ]
                    if r.get("analysis_text"):
                        _rpt += ["## Strategische Analyse", r["analysis_text"]]
                    _new_report    = "\n".join(_rpt)
                    _new_filenames = tuple(_a["filename"] for _a in analysed)
                    try:
                        _new_concepts = _generate_concepts_cached(
                            _new_report, _new_filenames, api_key
                        )
                        r["analysed"]    = analysed
                        r["concepts"]    = _new_concepts
                        r["full_report"] = _new_report
                        st.session_state["results"] = r
                        st.success("✅ Koppelingen bijgewerkt — briefings zijn opnieuw gegenereerd!")
                        st.rerun()
                    except Exception as _ov_err:
                        st.error(f"Fout bij regenereren: {_ov_err}")
                else:
                    st.info("Geen wijzigingen — selecteer een andere advertentienaam per banner.")

    st.markdown(
        "<div class='ok-footer'>"
        "<strong>Team up, power up</strong> &nbsp;·&nbsp; "
        "Authentiek onlinemarketingbureau van Zuid-Nederland"
        "</div>",
        unsafe_allow_html=True,
    )
