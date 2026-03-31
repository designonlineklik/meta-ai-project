import os
import re
import json
import base64
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
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Zijbalk
# ---------------------------------------------------------------------------

with st.sidebar:
    st.markdown(
        "<div style='padding:8px 0 4px 0'>"
        "<span style='font-size:1.1rem;font-weight:800;color:#00573C;font-family:Rubik,sans-serif'>"
        "Online Klik</span>"
        "<span style='font-size:0.75rem;color:#6c757d;display:block;margin-top:2px'>"
        "Meta Ads Analyser</span></div>",
        unsafe_allow_html=True,
    )
    st.divider()
    st.markdown(
        "**Hoe werkt het?**\n\n"
        "1. Upload je Meta Ads CSV-export\n"
        "2. Upload je bannerafbeeldingen\n"
        "3. Klik op **Analyse Starten**\n\n"
        "De AI classificeert je advertenties, analyseert de visuals "
        "met GPT-4o en genereert 10 creatieve briefings op basis van "
        "de winnende patronen."
    )
    st.divider()
    with st.expander("💡 Hoe te gebruiken", expanded=False):
        st.markdown(
            "**Master Prompt gebruiken:**\n\n"
            "1. Ga naar **Tab 3 → Creatieve Briefings**\n"
            "2. Kijk bij elk concept welke **referentie-afbeelding** de AI aanbeveelt\n"
            "3. Kopieer de **Master Prompt** met één klik\n"
            "4. Ga naar **[Nano Banana Pro](https://gemini.google.com/app)** of **[Freepik](https://freepik.com)**\n"
            "5. Upload de aangegeven referentie-afbeelding\n"
            "6. Plak de prompt en genereer je visual\n\n"
            "_De prompt zorgt er automatisch voor dat het product "
            "geïntegreerd wordt in de nieuwe banner._"
        )
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
# Koptekst
# ---------------------------------------------------------------------------

st.markdown(
    "<div class='ok-header'>"
    "<h1>Online Klik | Meta Ads Analyser</h1>"
    "<p>Upload je campagnedata en banners — de AI analyseert wat werkt, waarom het werkt, "
    "en genereert direct nieuwe creatieve briefings op basis van de winnende patronen.</p>"
    "</div>",
    unsafe_allow_html=True,
)
st.divider()

# ---------------------------------------------------------------------------
# Uploadgedeelte
# ---------------------------------------------------------------------------

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

# ---------------------------------------------------------------------------
# Welcome screen — only shown when no files are uploaded yet
# ---------------------------------------------------------------------------

if not csv_file and "results" not in st.session_state:
    st.markdown(
        "<div style='background:linear-gradient(135deg,#f5faf7 0%,#edfaf3 100%);"
        "border:1px solid #b2dcc5;border-radius:14px;padding:32px 36px;"
        "margin:20px 0;text-align:center'>"

        "<div style='font-size:1.4rem;font-weight:800;color:#00573C;"
        "font-family:Rubik,sans-serif;margin-bottom:10px'>"
        "Welkom bij de Meta Ads Analyser</div>"

        "<div style='font-size:0.92rem;color:#444;max-width:520px;"
        "margin:0 auto 20px auto;line-height:1.65'>"
        "Upload je Meta Ads CSV-export en bannerafbeeldingen om te starten. "
        "De AI classificeert je advertenties, analyseert de visuals en genereert "
        "direct 10 creatieve briefings op basis van de winnende patronen."
        "</div>"

        "<div style='display:flex;justify-content:center;gap:16px;flex-wrap:wrap'>"

        "<div style='background:#fff;border:1px solid #d0ead7;border-radius:10px;"
        "padding:14px 18px;min-width:130px'>"
        "<div style='font-size:1.4rem'>📁</div>"
        "<div style='font-size:0.75rem;font-weight:700;color:#00573C;"
        "text-transform:uppercase;letter-spacing:0.05em;margin:6px 0 2px'>Stap 1</div>"
        "<div style='font-size:0.82rem;color:#555'>Upload je CSV</div>"
        "</div>"

        "<div style='background:#fff;border:1px solid #d0ead7;border-radius:10px;"
        "padding:14px 18px;min-width:130px'>"
        "<div style='font-size:1.4rem'>🖼️</div>"
        "<div style='font-size:0.75rem;font-weight:700;color:#00573C;"
        "text-transform:uppercase;letter-spacing:0.05em;margin:6px 0 2px'>Stap 2</div>"
        "<div style='font-size:0.82rem;color:#555'>Upload je banners</div>"
        "</div>"

        "<div style='background:#fff;border:1px solid #d0ead7;border-radius:10px;"
        "padding:14px 18px;min-width:130px'>"
        "<div style='font-size:1.4rem'>🚀</div>"
        "<div style='font-size:0.75rem;font-weight:700;color:#00573C;"
        "text-transform:uppercase;letter-spacing:0.05em;margin:6px 0 2px'>Stap 3</div>"
        "<div style='font-size:0.82rem;color:#555'>Start de analyse</div>"
        "</div>"

        "<div style='background:#fff;border:1px solid #d0ead7;border-radius:10px;"
        "padding:14px 18px;min-width:130px'>"
        "<div style='font-size:1.4rem'>📄</div>"
        "<div style='font-size:0.75rem;font-weight:700;color:#00573C;"
        "text-transform:uppercase;letter-spacing:0.05em;margin:6px 0 2px'>Stap 4</div>"
        "<div style='font-size:0.82rem;color:#555'>Download het rapport</div>"
        "</div>"

        "</div>"

        "<div style='margin-top:20px;font-size:0.78rem;color:#888'>"
        "Ondersteunt alle Meta Ads Manager exports — komma, puntkomma of tab-gescheiden"
        "</div>"
        "</div>",
        unsafe_allow_html=True,
    )

# ---------------------------------------------------------------------------
# Hulpfuncties
# ---------------------------------------------------------------------------

def load_csv_from_upload(file) -> pd.DataFrame:
    """
    Reads an uploaded CSV regardless of delimiter (comma, semicolon, tab)
    or encoding (UTF-8, latin-1, cp1252). Uses sep=None + Python engine for
    auto-detection so Meta Ads exports from any locale are handled correctly.
    """
    from io import BytesIO
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
    nummer, titel, prioriteit, verwachte_impact,
    hook, primary_text, headline, visuele_omschrijving,
    referentie_afbeelding, master_prompt, design_strategy
    """
    image_filenames = image_filenames or []
    filenames_block = (
        "Beschikbare referentie-afbeeldingen (exacte bestandsnamen):\n"
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
        "- verwachte_impact (string): één zin — waarom dit concept scoort op basis van de winnende patronen\n"
        "- hook (string): thumb-stopper, max. 10 woorden\n"
        "- primary_text (string): 2–4 zinnen, menselijk en on-brand, inclusief het aanbod\n"
        "- headline (string): klik-trigger, max. 8 woorden\n"
        "- visuele_omschrijving (string): volledige art direction briefing voor de ontwerper — "
        "hero-afbeelding, model (uitdrukking, houding, demografisch profiel), productplaatsing, "
        "kleurenpalet, typografiestijl, sfeer, grafische elementen, beeldverhouding\n"
        "- referentie_afbeelding (string): kies de EXACTE bestandsnaam (uit de lijst hierboven) die het beste "
        "past als product-referentie voor dit concept. "
        "Als het concept product-gericht is, kies de meest passende afbeelding. "
        "Als het concept service- of sfeer-gericht is zonder duidelijk product, gebruik de waarde 'geen'.\n"
        "- master_prompt (string): een gedetailleerde, gebruiksklare prompt voor Freepik of Nano Banana Pro. "
        "De prompt is in het Engels en dekt verplicht: "
        "(1) Lighting — omschrijf het type licht (natural window light, soft studio diffused, golden hour, etc.), "
        "(2) Camera angle — benoem het type opname (close-up, hero shot, lifestyle wide, flat lay, over-the-shoulder), "
        "(3) Composition — geef de beeldopbouw aan (rule of thirds, centered product, negative space, layered depth), "
        "(4) Online Klik aesthetic — clean, professional, conversion-focused, minimalist luxury. "
        "VERPLICHT: als referentie_afbeelding een bestandsnaam is (niet 'geen'), bevat de prompt: "
        "(a) de tag '[Product vanuit {BESTANDSNAAM}]' vroeg in de prompt, waarbij {BESTANDSNAAM} de exacte waarde "
        "van referentie_afbeelding is — dit vertelt de AI-generator welk object te isoleren; "
        "(b) de zin 'Integrated with the product and style from {BESTANDSNAAM}'. "
        "Als de afbeelding meerdere producten kan bevatten, voeg dan toe: "
        "'Focus op: [het meest prominente product] uit de afbeelding' — beschrijf het product specifiek "
        "(kleur, materiaal, vorm) zodat er geen ambiguïteit is. "
        "Als referentie_afbeelding 'geen' is, schrijf: "
        "'No specific product reference needed — focus on the vibe and human interaction described in the briefing.' "
        "Als het concept product-gericht is: benadruk productdetails, textuur en materiaal. "
        "Als het concept service- of sfeer-gericht is: focus op menselijke interactie en emotie. "
        "Sluit af met: aspect ratio 1:1, no text overlays, no watermarks.\n"
        "- design_strategy (string): 1-2 zinnen in het Nederlands die uitleggen waarom juist deze visuele richting "
        "gekozen is op basis van de top-presterende advertenties in de analyse.\n\n"
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


def _strip_markdown(text: str) -> str:
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

            # Design Strategy
            design_strategy = c.get("design_strategy", "")
            if design_strategy:
                pdf.set_x(_PDF_MARGIN)
                pdf.set_font("Helvetica", "B", 9)
                pdf.set_text_color(0, 87, 60)
                pdf.cell(_PDF_LABEL_W, 5, "Design Strategy:", ln=False)
                pdf.set_font("Helvetica", "I", 9)
                pdf.set_text_color(20, 20, 20)
                pdf.multi_cell(val_w, 5, safe_pdf_text(design_strategy))

            # Master Prompt
            master_prompt = c.get("master_prompt", "")
            if master_prompt:
                pdf.set_x(_PDF_MARGIN)
                pdf.set_font("Helvetica", "B", 9)
                pdf.set_text_color(0, 87, 60)
                pdf.cell(pdf.epw, 5, "Master Prompt (Freepik / Nano Banana Pro):", ln=True)
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
# Analyse-pipeline
# ---------------------------------------------------------------------------

if start_btn:
    st.session_state.pop("results", None)

    if not api_key:
        st.error("Voer eerst je OpenAI API-sleutel in via Developer Settings in de zijbalk.")
        st.stop()

    try:
        client = OpenAI(api_key=api_key)
        results = {}

        # Stap 1: CSV classificeren
        with st.spinner("Krachten bundelen... Wij vertalen jouw data naar groei-kansen."):
            df = load_csv_from_upload(csv_file)
            df, metric_used = classify_ads(df)
            results["df"] = df
            results["metric_used"] = metric_used

        # Stap 2: Visuele analyse
        analysed = []
        ad_col = find_column(df, AD_NAME_COLUMNS)
        perf_map = {}
        if ad_col:
            for _, row in df.iterrows():
                perf_map[str(row[ad_col]).strip()] = row.get("performance_category", "No Data")

        matched_count = 0
        if uploaded_images:
            progress = st.progress(0, text="Visuele analyse bezig...")
            for i, img_file in enumerate(uploaded_images):
                progress.progress(
                    (i + 1) / len(uploaded_images),
                    text=f"GPT-4o analyseert: {img_file.name}",
                )
                name, cat = match_category(img_file.name, perf_map) if perf_map else (img_file.name, "No Data")
                if perf_map and name in perf_map:
                    matched_count += 1
                try:
                    description = describe_image(client, img_file, name)
                except Exception as e:
                    description = f"_(Fout bij analyseren: {e})_"
                analysed.append({"name": name, "filename": img_file.name, "category": cat, "description": description})
            progress.empty()

        results["analysed"] = analysed
        results["match_count"] = matched_count
        results["total_images"] = len(uploaded_images) if uploaded_images else 0

        # Build a filename → bytes map so Tab 3 can render thumbnails
        image_bytes_map: Dict[str, bytes] = {}
        if uploaded_images:
            for img_file in uploaded_images:
                img_file.seek(0)
                image_bytes_map[img_file.name] = img_file.read()
        results["image_bytes_map"] = image_bytes_map

        # Stap 3: Strategische vergelijking
        high   = [a for a in analysed if a["category"] == "High Performer"]
        avg    = [a for a in analysed if a["category"] == "Average"]
        under  = [a for a in analysed if a["category"] == "Underperformer"]
        no_dat = [a for a in analysed if a["category"] not in ("High Performer", "Average", "Underperformer")]

        analysis_text = ""
        if analysed:
            with st.spinner("Krachten bundelen... Wij vertalen jouw data naar groei-kansen."):
                try:
                    analysis_text = compare_creatives(client, high, under + avg, no_dat)
                except Exception as e:
                    analysis_text = f"_(Kon vergelijking niet genereren: {e})_"
        results["analysis_text"] = analysis_text

        report_lines = ["# Meta Ads Visueel Analyserapport", ""]
        for ad in analysed:
            report_lines += [f"## {ad['name']} ({CATEGORY_NL.get(ad['category'], ad['category'])})", ad["description"], ""]
        if analysis_text:
            report_lines += ["## Strategische Analyse", analysis_text]
        full_report = "\n".join(report_lines)
        results["full_report"] = full_report

        # Stap 4: Concepten genereren
        concepts: List[Dict] = []
        img_filenames = [a["filename"] for a in analysed] if analysed else []
        with st.spinner("Krachten bundelen... Wij vertalen jouw data naar groei-kansen."):
            try:
                concepts = generate_concepts(client, full_report, img_filenames)
            except Exception as e:
                st.warning(f"Kon concepten niet genereren: {e}")
        results["concepts"] = concepts

        concepts_md_lines = [
            "# Creatieve Briefings — Nieuwe Advertentieconcepten",
            f"_Gegenereerd: {datetime.now().strftime('%Y-%m-%d %H:%M')}_",
            "",
            "> Gebaseerd op de winnende patronen uit de campagneanalyse.",
            "",
        ]
        for c in concepts:
            concepts_md_lines += [
                "---",
                f"### Concept {c.get('nummer', '?')}: {c.get('titel', '')}",
                f"**Prioriteit:** {c.get('prioriteit', '—')}  |  **Verwachte Impact:** {c.get('verwachte_impact', '—')}",
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
                f"**Design Strategy:** {c.get('design_strategy', '—')}",
                "",
                "**Master Prompt (Freepik / Nano Banana Pro):**",
                "",
                f"```",
                c.get('master_prompt', '—'),
                f"```",
                "",
            ]
        results["concepts_md"] = "\n".join(concepts_md_lines)
        st.session_state["results"] = results

    except Exception as _pipeline_err:
        st.error(
            "Oeps! Dit bestandstype wordt niet herkend. "
            "Gebruik a.u.b. een originele Meta Ads Manager export."
        )
        with st.expander("Technische details (voor de consultant)", expanded=False):
            st.exception(_pipeline_err)

# ---------------------------------------------------------------------------
# Resultaten weergeven
# ---------------------------------------------------------------------------

if "results" in st.session_state:
    r = st.session_state["results"]
    df: pd.DataFrame = r["df"]
    metric_used: str = r["metric_used"]
    analysed: List[Dict] = r["analysed"]
    concepts: List[Dict] = r.get("concepts", [])
    image_bytes_map: Dict[str, bytes] = r.get("image_bytes_map", {})

    st.success("✅ Analyse voltooid!")

    mc = r.get("match_count", 0)
    ti = r.get("total_images", 0)
    if ti > 0:
        icon = "✅" if mc == ti else ("⚠️" if mc > 0 else "❌")
        st.markdown(
            f"<div style='background:#edfaf3;border:1px solid #33B784;border-radius:8px;"
            f"padding:8px 16px;margin-bottom:8px;font-size:0.9rem'>"
            f"{icon} <strong>Koppelstatus:</strong> {mc} van de {ti} afbeeldingen "
            f"succesvol gekoppeld aan de CSV-data.</div>",
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

        st.download_button(
            "⬇️ Download geclassificeerde CSV",
            data=df.to_csv(index=False).encode("utf-8"),
            file_name="geclassificeerde_advertenties.csv",
            mime="text/csv",
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
            dl1, dl2 = st.columns(2)
            with dl1:
                st.download_button(
                    "⬇️ Download analyserapport (Markdown)",
                    data=r["full_report"].encode("utf-8"),
                    file_name="analyserapport.md",
                    mime="text/markdown",
                    use_container_width=True,
                )
            with dl2:
                with st.spinner("PDF voorbereiden..."):
                    pdf_bytes = generate_pdf(
                        df=df,
                        metric_used=metric_used,
                        analysed=analysed,
                        analysis_text=analysis_txt,
                        concepts=concepts,
                    )
                st.download_button(
                    "📄 Download Rapport (PDF)",
                    data=pdf_bytes,
                    file_name="online_klik_analyserapport.pdf",
                    mime="application/pdf",
                    use_container_width=True,
                )

    # ---- Tab 3: Creatieve briefings -----------------------------------------
    with tab3:
        st.subheader("10 Creatieve Briefings")
        st.caption(
            "Gegenereerd op basis van de winnende creatieve patronen. "
            "Elke briefing is direct bruikbaar voor een ontwerper of copywriter."
        )

        if not concepts:
            st.warning("Geen concepten beschikbaar. Controleer je API-sleutel en probeer opnieuw.")
        else:
            # Summary row: count per priority
            prio_counts = {}
            for c in concepts:
                p = c.get("prioriteit", "Onbekend")
                prio_counts[p] = prio_counts.get(p, 0) + 1

            p1, p2, p3 = st.columns(3)
            p1.metric("🔴 Hoge Prioriteit", prio_counts.get("Hoog", 0))
            p2.metric("🟠 Middel Prioriteit", prio_counts.get("Middel", 0))
            p3.metric("⚪ Lage Prioriteit", prio_counts.get("Laag", 0))

            st.divider()

            for concept in concepts:
                nummer         = concept.get("nummer", "?")
                titel          = concept.get("titel", "Zonder titel")
                prioriteit     = concept.get("prioriteit", "—")
                impact         = concept.get("verwachte_impact", "—")
                hook           = concept.get("hook", "—")
                body           = concept.get("primary_text", "—")
                headline       = concept.get("headline", "—")
                visual         = concept.get("visuele_omschrijving", "—")
                master_prompt  = concept.get("master_prompt", "")
                design_strategy = concept.get("design_strategy", "")
                ref_img        = concept.get("referentie_afbeelding", "")
                ref_is_file    = bool(ref_img and ref_img.lower() != "geen")

                prio_cfg       = PRIORITY_COLOURS.get(prioriteit, ("#6c757d", "#fff", prioriteit))
                bg_col, txt_col, badge_label = prio_cfg

                with st.container(border=True):
                    # ── Header: concept name + priority badge ──────────────────
                    head_left, head_right = st.columns([3, 1])
                    with head_left:
                        st.markdown(f"#### Concept {nummer}: {titel}")
                    with head_right:
                        st.markdown(
                            f"<div style='text-align:right;margin-top:6px'>"
                            f"<span style='background:{bg_col};color:{txt_col};"
                            f"padding:5px 14px;border-radius:20px;font-size:0.78rem;"
                            f"font-weight:700;letter-spacing:0.05em;display:inline-block'>"
                            f"● {prioriteit.upper()}</span><br>"
                            f"<span style='font-size:0.72rem;color:{bg_col};font-weight:600;"
                            f"letter-spacing:0.04em'>{badge_label}</span>"
                            f"</div>",
                            unsafe_allow_html=True,
                        )

                    # ── Strategic ratio banner ─────────────────────────────────
                    st.markdown(
                        f"<div style='background:#edfaf3;border-left:4px solid #33B784;"
                        f"padding:8px 14px;border-radius:4px;margin:4px 0 14px 0;"
                        f"font-size:0.9rem'>💡 <strong>Strategische Ratio:</strong> {impact}</div>",
                        unsafe_allow_html=True,
                    )

                    # ── Main 2-column layout: LEFT thumbnail | RIGHT content ───
                    col_left, col_right = st.columns([1, 2])

                    with col_left:
                        # Thumbnail of the reference image
                        if ref_is_file:
                            thumb_bytes = image_bytes_map.get(ref_img)
                            if thumb_bytes:
                                st.image(thumb_bytes, use_container_width=True)
                            st.markdown(
                                f"<div style='background:#fff3cd;border:1px solid #ffc107;"
                                f"border-radius:6px;padding:6px 10px;margin-top:6px;"
                                f"font-size:0.78rem;word-break:break-all'>"
                                f"🖼️ <strong>Referentie:</strong><br><code>{ref_img}</code></div>",
                                unsafe_allow_html=True,
                            )
                        else:
                            st.markdown(
                                "<div style='background:#f8f9fa;border:1px solid #dee2e6;"
                                "border-radius:8px;padding:14px;font-size:0.82rem;"
                                "color:#6c757d;text-align:center;min-height:120px;"
                                "display:flex;align-items:center;justify-content:center'>"
                                "🖼️<br><em>Geen product-referentie — focus op de vibe</em></div>",
                                unsafe_allow_html=True,
                            )

                    with col_right:
                        # Concept naam + design strategy
                        if design_strategy:
                            st.markdown(
                                f"<div style='background:#f5faf7;border-left:3px solid #00573C;"
                                f"padding:8px 12px;border-radius:4px;margin-bottom:10px;"
                                f"font-size:0.85rem;color:#1a3a2a'>"
                                f"<strong>Design Strategy:</strong> {design_strategy}</div>",
                                unsafe_allow_html=True,
                            )

                        # Master Prompt box
                        st.markdown(
                            "<div style='font-size:0.78rem;font-weight:700;color:#00573C;"
                            "letter-spacing:0.05em;text-transform:uppercase;margin-bottom:4px'>"
                            "🎨 Master Prompt</div>",
                            unsafe_allow_html=True,
                        )
                        if master_prompt:
                            st.code(master_prompt, language=None)
                        st.caption(
                            "Copy-paste this into Nano Banana Pro / Freepik — "
                            "upload the reference image on the left first."
                        )

                    # ── Copy text fields (full width, collapsible) ─────────────
                    with st.expander("✏️ Advertentieteksten (Hook · Primary Text · Headline)"):
                        st.markdown(
                            f"<div style='background:#fff8e1;border-radius:6px;"
                            f"padding:10px 14px;margin-bottom:8px'>"
                            f"<span style='font-size:0.72rem;font-weight:700;color:#b7791f;"
                            f"letter-spacing:0.06em;text-transform:uppercase'>Hook</span><br>"
                            f"<span style='font-size:1rem'>{hook}</span></div>",
                            unsafe_allow_html=True,
                        )
                        st.markdown(
                            f"<div style='background:#f8f9fa;border-radius:6px;"
                            f"padding:10px 14px;margin-bottom:4px'>"
                            f"<span style='font-size:0.72rem;font-weight:700;color:#495057;"
                            f"letter-spacing:0.06em;text-transform:uppercase'>Primary Text</span><br>"
                            f"<span style='font-size:0.9rem'>{body}</span></div>",
                            unsafe_allow_html=True,
                        )
                        clipboard_button(body, key=f"pt_{nummer}")
                        st.markdown("<div style='margin-bottom:6px'></div>", unsafe_allow_html=True)
                        st.markdown(
                            f"<div style='background:#e8f4fd;border-radius:6px;"
                            f"padding:10px 14px;margin-bottom:4px'>"
                            f"<span style='font-size:0.72rem;font-weight:700;color:#1a6ca8;"
                            f"letter-spacing:0.06em;text-transform:uppercase'>Headline</span><br>"
                            f"<span style='font-size:1rem;font-weight:600'>{headline}</span></div>",
                            unsafe_allow_html=True,
                        )
                        clipboard_button(headline, key=f"hl_{nummer}")

                    # ── Art direction (full width, collapsible) ────────────────
                    with st.expander("🎨 Art Direction Briefing"):
                        st.markdown(visual)

                st.divider()

            st.divider()
            st.download_button(
                "⬇️ Download alle briefings (Markdown)",
                data=r.get("concepts_md", "").encode("utf-8"),
                file_name="creatieve_briefings.md",
                mime="text/markdown",
                use_container_width=True,
            )

    # --- Afsluitend bericht ---------------------------------------------------
    st.divider()
    st.markdown(
        "<div style='text-align:center;padding:24px 0 8px 0'>"
        "<div style='font-size:2rem'>🎉</div>"
        "<div style='font-size:1.2rem;font-weight:700;margin:8px 0 4px 0;font-family:Rubik,sans-serif;color:#00573C'>"
        "Analyse voltooid! Je nieuwe creatives staan klaar onder de tab <em>Creatieve Briefings</em>.</div>"
        "<div style='font-size:0.9rem;color:#6c757d'>"
        "Kopieer een briefing, geef hem aan je ontwerper, en test de winnende formule opnieuw.</div>"
        "</div>"
        "<div class='ok-footer'>"
        "<strong>Team up, power up</strong><br>"
        "Authentiek onlinemarketingbureau van Zuid-Nederland"
        "</div>",
        unsafe_allow_html=True,
    )
