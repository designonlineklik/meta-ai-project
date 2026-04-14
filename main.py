import os
import glob
import base64
from datetime import datetime
from typing import Optional, Tuple, List, Dict
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# --- Column name mappings for common Meta Ads export formats ---
ROAS_COLUMNS = [
    # Dutch (NL)
    "ROAS (rendement op advertentie-uitgaven) voor aankoop",
    # English
    "Purchase ROAS (return on ad spend)",
    "ROAS (return on ad spend)",
    "purchase_roas",
    "roas",
    "ROAS",
]

CTR_COLUMNS = [
    # Dutch (NL)
    "CTR (doorklikratio voor klikken op link)",
    # English
    "CTR (all)",
    "CTR (link click-through rate)",
    "Link CTR",
    "ctr_all",
    "ctr",
    "CTR",
]

CPC_COLUMNS = [
    "CPC (kosten per klik op link)",
    "CPC (cost per link click)",
    "CPC (all)",
    "Cost per link click",
    "CPC",
]

AD_NAME_COLUMNS = ["Advertentienaam", "Ad name", "ad_name", "Ad Name", "name"]
CAMPAIGN_COLUMNS = ["Campagnenaam", "Campaign name", "campaign_name", "Campaign Name"]
ADSET_COLUMNS = ["Naam advertentieset", "Ad set name", "adset_name", "Ad Set Name"]


def find_column(df: pd.DataFrame, candidates: list) -> Optional[str]:
    """Return the first candidate column name that exists in the DataFrame."""
    for col in candidates:
        if col in df.columns:
            return col
    return None


# ---------------------------------------------------------------------------
# NOTE: The functions below (load_csv, find_image_for_ad, run_visual_analysis,
# generate_ad_concepts, main) use local folders (data/, input_images/, output/).
# They are kept here for CLI use only and are NOT called by the Streamlit app.
# The Streamlit app (app.py) handles all file I/O via st.file_uploader.
# ---------------------------------------------------------------------------

def load_csv() -> pd.DataFrame:
    """CLI only — Load the first CSV file found in the 'data' folder."""
    csv_files = glob.glob(os.path.join("data", "*.csv"))
    if not csv_files:
        raise FileNotFoundError("No CSV files found in the 'data' folder.")
    if len(csv_files) > 1:
        print(f"Multiple CSV files found. Using: {csv_files[0]}")
    df = pd.read_csv(csv_files[0])
    print(f"Loaded: {csv_files[0]}  ({len(df)} rows, {len(df.columns)} columns)\n")
    return df


def classify_by_roas(value: float, hi: float = 3.0, lo: float = 1.5) -> str:
    if value >= hi:
        return "High Performer"
    elif value >= lo:
        return "Average"
    return "Underperformer"


def classify_by_ctr(value: float, hi: float = 2.0, lo: float = 1.0) -> str:
    if value >= hi:
        return "High Performer"
    elif value >= lo:
        return "Average"
    return "Underperformer"


def classify_by_cpc(value: float, lo: float = 0.3, hi: float = 0.8) -> str:
    """Lower CPC is better. lo/hi are the thresholds for High/Underperformer."""
    if value <= lo:
        return "High Performer"
    elif value <= hi:
        return "Average"
    return "Underperformer"


def clean_dutch_number(value) -> float:
    """
    Convert a single value to float, correctly handling both Dutch and English
    number formats.

    Decision table:
      dot + comma present  → dot is thousands sep, comma is decimal
                             '1.234,56' → 1234.56
      comma only           → comma is decimal sep
                             '2,67'    → 2.67
      dot only, ≤2 decimal digits after dot
                           → dot is decimal separator (English)
                             '2.67'    → 2.67   |   '2.6' → 2.6
      dot only, exactly 3 digits after dot (or multiple dots)
                           → dot is thousands separator (Dutch)
                             '1.000'   → 1000   |   '1.000.000' → 1000000
      '€ 1.234'           → strip €, then '1.234' → 1000  (3-digit rule)

    This fixes the previous bug where '2.67' became 267 (all dots stripped).
    """
    s = str(value).strip()

    # Strip currency/percentage symbols
    for sym in ("€", "$", "£", "%"):
        s = s.replace(sym, "")
    s = s.strip()

    if not s or s.lower() in ("nan", "none", "-", ""):
        return float("nan")

    if "." in s and "," in s:
        # Both separators → Dutch thousands format
        s = s.replace(".", "").replace(",", ".")

    elif "," in s:
        # Comma only → European decimal
        s = s.replace(",", ".")

    elif "." in s:
        # Dot only — disambiguate by inspecting decimal-part length
        parts = s.split(".")
        if len(parts) > 2:
            # Multiple dots (e.g. '1.000.000') → all are thousands separators
            s = s.replace(".", "")
        elif len(parts) == 2 and len(parts[1]) == 3:
            # Exactly 3 digits after dot → Dutch thousands separator
            # e.g. '1.000' → 1000, '1.234' → 1234
            s = s.replace(".", "")
        # else: 1 or 2 decimal digits → English decimal point, leave as-is

    try:
        return float(s)
    except ValueError:
        return float("nan")


def parse_dutch_numeric(series: pd.Series) -> pd.Series:
    """Apply clean_dutch_number to every value in a Series, returning floats."""
    return series.apply(clean_dutch_number)


# Columns that may contain spend / budget amounts (cleaned alongside ROAS/CTR)
SPEND_COLUMNS = [
    "Besteed bedrag (EUR)", "Amount spent (EUR)", "Besteed bedrag",
    "Amount spent", "Kosten", "Spend",
]

# Columns that may contain purchase revenue (used for weighted ROAS)
REVENUE_COLUMNS = [
    "Conversiewaarde van aankopen", "Purchase conversion value",
    "Opbrengst", "Revenue", "Omzet",
]


def _dynamic_thresholds(series: pd.Series, multiplier_hi: float = 1.2,
                        multiplier_lo: float = 0.8,
                        fallback_hi: float = 3.0,
                        fallback_lo: float = 1.5) -> Tuple[float, float, Optional[float]]:
    """
    Compute relative thresholds from a numeric series.
    Returns (hi_threshold, lo_threshold, avg_or_None).
    Falls back to the provided hardcoded values when fewer than 2 valid rows exist.
    """
    valid = series.dropna()
    valid = valid[valid > 0]
    if len(valid) >= 2:
        avg = valid.mean()
        return avg * multiplier_hi, avg * multiplier_lo, avg
    return fallback_hi, fallback_lo, None


def classify_ads(df: pd.DataFrame) -> Tuple[pd.DataFrame, str]:
    """
    Add a 'performance_category' column to df using dynamic, data-relative thresholds.

    Classification priority per row:
      1. ROAS  — dynamic thresholds: avg*1.2 (High) / avg*0.8 (Average/Under)
      2. CTR   — fallback when ROAS is missing/zero for that row
      3. CPC   — fallback when both ROAS and CTR are missing/zero (lower = better)
      4. 'No Data' — all three metrics absent

    All numeric columns are cleaned from Dutch locale format (e.g. '3.597,12' → 3597.12)
    so downstream KPI cards get plain Python floats without further parsing.
    """
    roas_col = find_column(df, ROAS_COLUMNS)
    ctr_col  = find_column(df, CTR_COLUMNS)
    cpc_col  = find_column(df, CPC_COLUMNS)

    # ── Step 1: Parse all numeric columns from Dutch/English locale ──────────
    if roas_col:
        df[roas_col] = parse_dutch_numeric(df[roas_col])
    if ctr_col:
        # Strip trailing % before parsing
        df[ctr_col] = parse_dutch_numeric(
            df[ctr_col].astype(str).str.replace("%", "", regex=False)
        )
    if cpc_col:
        df[cpc_col] = parse_dutch_numeric(df[cpc_col])
    for _col in [find_column(df, SPEND_COLUMNS), find_column(df, REVENUE_COLUMNS)]:
        if _col and _col in df.columns:
            df[_col] = parse_dutch_numeric(df[_col])

    # ── Step 2: Compute dynamic thresholds ───────────────────────────────────
    if not roas_col and not ctr_col:
        raise ValueError(
            "Geen ROAS- of CTR-kolom gevonden in de CSV.\n"
            f"Verwacht een van: {ROAS_COLUMNS + CTR_COLUMNS}\n"
            f"Aanwezige kolommen: {list(df.columns)}"
        )

    # ROAS thresholds (relative to dataset average)
    roas_hi = roas_lo = roas_avg = None
    if roas_col:
        roas_hi, roas_lo, roas_avg = _dynamic_thresholds(
            df[roas_col], fallback_hi=3.0, fallback_lo=1.5
        )

    # CTR thresholds (relative to dataset average; also used for row-level fallback)
    ctr_hi = ctr_lo = None
    if ctr_col:
        ctr_hi, ctr_lo, _ = _dynamic_thresholds(
            df[ctr_col], fallback_hi=2.0, fallback_lo=1.0
        )

    # CPC thresholds (relative, inverted: lower CPC = better)
    cpc_lo_thresh = cpc_hi_thresh = None
    if cpc_col:
        # For CPC: "high" threshold = avg*0.8 (cheaper = High Performer)
        #          "low"  threshold = avg*1.2 (expensive = Underperformer)
        _cpc_hi, _cpc_lo, _cpc_avg = _dynamic_thresholds(
            df[cpc_col], multiplier_hi=0.8, multiplier_lo=1.2,
            fallback_hi=0.3, fallback_lo=0.8
        )
        cpc_lo_thresh = _cpc_hi   # below this  → High Performer
        cpc_hi_thresh = _cpc_lo   # above this  → Underperformer

    # ── Step 3: Row-level classification with fallback chain ─────────────────
    def _classify_row(row) -> str:
        # Primary: ROAS
        if roas_col and roas_hi is not None:
            v = row[roas_col]
            if pd.notna(v) and v > 0:
                return classify_by_roas(v, roas_hi, roas_lo)

        # Secondary: CTR (when ROAS is absent/zero for this row)
        if ctr_col and ctr_hi is not None:
            v = row[ctr_col]
            if pd.notna(v) and v > 0:
                return classify_by_ctr(v, ctr_hi, ctr_lo)

        # Tertiary: CPC (lower is better; when ROAS and CTR both absent)
        if cpc_col and cpc_lo_thresh is not None:
            v = row[cpc_col]
            if pd.notna(v) and v > 0:
                return classify_by_cpc(v, cpc_lo_thresh, cpc_hi_thresh)

        return "No Data"

    if roas_col:
        df["performance_category"] = df.apply(_classify_row, axis=1)
        if roas_avg is not None:
            thresh_str = (
                f"gem {roas_avg:.2f}x → Hoog ≥ {roas_hi:.2f}, "
                f"Gem ≥ {roas_lo:.2f}, Laag < {roas_lo:.2f}"
            )
        else:
            thresh_str = f"Hoog ≥ {roas_hi}, Gem ≥ {roas_lo}"
        metric_used = f"ROAS (dynamisch — {thresh_str})"

    else:
        # Pure CTR mode (no ROAS column in this CSV at all)
        df["performance_category"] = df[ctr_col].apply(
            lambda v: classify_by_ctr(v, ctr_hi, ctr_lo) if pd.notna(v) and v > 0
            else "No Data"
        )
        _, _, ctr_avg = _dynamic_thresholds(df[ctr_col], fallback_hi=2.0, fallback_lo=1.0)
        if ctr_avg is not None:
            thresh_str = (
                f"gem {ctr_avg:.2f}% → Hoog ≥ {ctr_hi:.2f}%, "
                f"Gem ≥ {ctr_lo:.2f}%"
            )
        else:
            thresh_str = f"Hoog ≥ {ctr_hi}%, Gem ≥ {ctr_lo}%"
        metric_used = f"CTR (dynamisch — {thresh_str})"

    return df, metric_used


def print_summary(df: pd.DataFrame, metric_used: str) -> None:
    """Print a formatted performance summary to the terminal."""
    ad_col = find_column(df, AD_NAME_COLUMNS)
    campaign_col = find_column(df, CAMPAIGN_COLUMNS)

    counts = df["performance_category"].value_counts()
    total = len(df)

    print("\n" + "=" * 52)
    print("   META ADS PERFORMANCE SUMMARY")
    print("=" * 52)
    print(f"   Total ads analysed : {total}")
    print(f"   Metric used        : {metric_used}")
    print("-" * 52)

    for category in ["High Performer", "Average", "Underperformer", "No Data"]:
        count = counts.get(category, 0)
        pct = (count / total * 100) if total > 0 else 0
        bar = "#" * int(pct / 5)
        print(f"   {category:<18} {count:>3}  ({pct:5.1f}%)  {bar}")

    print("=" * 52)

    # Show top 5 high performers
    high = df[df["performance_category"] == "High Performer"]
    if not high.empty and ad_col:
        print("\n   Top High Performers:")
        for _, row in high.head(5).iterrows():
            name = row.get(ad_col, "-")
            campaign = f"  [{row[campaign_col]}]" if campaign_col else ""
            print(f"     * {name}{campaign}")

    # Show underperformers
    under = df[df["performance_category"] == "Underperformer"]
    if not under.empty and ad_col:
        print("\n   Underperformers:")
        for _, row in under.head(5).iterrows():
            name = row.get(ad_col, "-")
            campaign = f"  [{row[campaign_col]}]" if campaign_col else ""
            print(f"     * {name}{campaign}")

    print()



# ---------------------------------------------------------------------------
# Visual Analysis with GPT-4o
# ---------------------------------------------------------------------------

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".gif", ".webp"}


def encode_image(path: str) -> Tuple[str, str]:
    """Return (base64_string, mime_type) for a local image file."""
    ext = os.path.splitext(path)[1].lower()
    mime = "image/jpeg" if ext in (".jpg", ".jpeg") else f"image/{ext.lstrip('.')}"
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8"), mime


def find_image_for_ad(ad_name: str, image_dir: str = "input_images") -> Optional[str]:
    """
    Try to match an ad name to an image file.
    Checks for exact filename match, then a fuzzy substring match.
    Returns the file path or None.
    """
    if not ad_name:
        return None
    ad_slug = ad_name.lower().replace(" ", "_")
    for ext in IMAGE_EXTENSIONS:
        # Exact slug match
        candidate = os.path.join(image_dir, f"{ad_slug}{ext}")
        if os.path.exists(candidate):
            return candidate
    # Fuzzy: ad name words appear in filename
    words = [w.lower() for w in ad_name.split() if len(w) > 2]
    for f in glob.glob(os.path.join(image_dir, "*")):
        fname = os.path.basename(f).lower()
        if any(w in fname for w in words):
            return f
    return None


TONE_INSTRUCTION = (
    "BELANGRIJK: Begin je antwoord ALTIJD direct met de gevraagde inhoud. "
    "Gebruik NOOIT inleidende zinnen zoals 'Zeker!', 'Hier is', 'Natuurlijk', 'Goed idee', "
    "'Natuurlijk, hier is', 'Met plezier', of enige andere conversationele opening. "
    "Eerste woord van je antwoord is altijd inhoud, nooit een begroeting of bevestiging. "
    "Schrijf uitsluitend in professioneel, direct Nederlands."
)


def describe_image(client: OpenAI, image_path: str, ad_name: str) -> str:
    """Send one image to GPT-4o and get a structured visual description."""
    b64, mime = encode_image(image_path)
    prompt = (
        f"Je bent een senior Meta Ads creatief strateeg. {TONE_INSTRUCTION}\n\n"
        f"Analyseer de banneradvertentie '{ad_name}' en beschrijf de volgende elementen:\n\n"
        "1. **Hook / Headline**: Wat ziet de kijker als eerste? Is het voordeel-gedreven, nieuwsgierigheidsgericht of aanbod-gebaseerd?\n"
        "2. **Visuele Stijl**: Algemene uitstraling — minimalistisch, gedurfd, lifestyle, productgericht, enz.\n"
        "3. **Kleurenpalet**: Dominante kleuren en de emotie/energie die ze uitstralen.\n"
        "4. **Productpresentatie**: Hoe wordt het product getoond? Close-up, in context, flatlay?\n"
        "5. **Persoon / Model**: Is er een persoon? Zo ja, beschrijf uitdrukking, houding en doelgroepkenmerken.\n"
        "6. **Call to Action (CTA)**: Wat zegt de CTA en hoe prominent is deze?\n"
        "7. **Algemene Eerste Indruk**: In één zin — welk gevoel of verlangen wekt deze advertentie op?\n\n"
        "Gebruik opsommingstekens onder elke kop. Wees specifiek en beknopt."
    )
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "system",
                "content": f"Je bent een senior Meta Ads creatief strateeg. {TONE_INSTRUCTION}",
            },
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
        max_tokens=800,
    )
    return response.choices[0].message.content


def compare_creatives(
    client: OpenAI,
    high_performers: List[Dict],
    underperformers: List[Dict],
    no_data: List[Dict],
) -> str:
    """Ask GPT-4o to reason about WHY winners win and losers lose."""

    def format_block(ads: List[Dict]) -> str:
        return "\n\n".join(
            f"### {a['name']} ({a['category']})\n{a['description']}" for a in ads
        )

    sections = []
    if high_performers:
        sections.append("## HIGH PERFORMERS\n" + format_block(high_performers))
    if underperformers:
        sections.append("## UNDERPERFORMERS\n" + format_block(underperformers))
    if no_data:
        sections.append("## NO PERFORMANCE DATA (included for reference)\n" + format_block(no_data))

    context = "\n\n---\n\n".join(sections)

    system_msg = f"Je bent een senior Meta Ads creatief strateeg. {TONE_INSTRUCTION}"
    prompt = (
        "Hieronder staan visuele beschrijvingen van advertenties uit dezelfde campagne, "
        "gegroepeerd per prestatieniveau.\n\n"
        f"{context}\n\n"
        "---\n\n"
        "Schrijf op basis van uitsluitend de visuele en creatieve elementen een strategische analyse "
        "met de volgende secties:\n\n"
        "## 1. Waarom de Winnaar Wint\n"
        "Benoem de specifieke creatieve keuzes die de hogere prestatie veroorzaken. Verwijs naar advertentienamen.\n\n"
        "## 2. Waarom de Verliezer Verliest\n"
        "Benoem de specifieke creatieve zwaktes. Wat ontbreekt of slaagt er niet in de scroll te stoppen? "
        "Verwijs naar advertentienamen.\n\n"
        "## 3. Het Beslissende Patroon\n"
        "Het belangrijkste creatieve verschil tussen top en onderkant in één vetgedrukte, pakkende zin.\n\n"
        "## 4. Aanbevelingen\n"
        "Geef 3 concrete, specifieke verbeteringen voor de onderpresterende advertenties. "
        "Gebruik voor elke aanbeveling dit formaat:\n\n"
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
        max_tokens=1200,
    )
    return response.choices[0].message.content


def run_visual_analysis(df: pd.DataFrame) -> None:
    """
    Main orchestrator for the visual analysis phase.
    Reads images from input_images/, calls GPT-4o, writes output/analysis_report.md.
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Skipping visual analysis: OPENAI_API_KEY not set.")
        return

    images = glob.glob(os.path.join("input_images", "*"))
    images = [p for p in images if os.path.splitext(p)[1].lower() in IMAGE_EXTENSIONS]

    if not images:
        print("No images found in input_images/ — skipping visual analysis.")
        return

    ad_col = find_column(df, AD_NAME_COLUMNS)
    client = OpenAI(api_key=api_key)

    print(f"\nFound {len(images)} image(s) in input_images/. Starting GPT-4o analysis...\n")

    # Build a lookup: ad_name -> performance_category
    perf_map: Dict[str, str] = {}
    if ad_col:
        for _, row in df.iterrows():
            perf_map[str(row[ad_col]).strip()] = row.get("performance_category", "No Data")

    analysed: List[Dict] = []

    for img_path in sorted(images):
        filename = os.path.basename(img_path)
        name_stem = os.path.splitext(filename)[0]

        # Try to find a matching ad name in the dataframe
        matched_ad_name = name_stem
        category = "No Data"
        if perf_map:
            for ad_name, cat in perf_map.items():
                slug = ad_name.lower().replace(" ", "_")
                if slug in name_stem.lower() or name_stem.lower() in slug:
                    matched_ad_name = ad_name
                    category = cat
                    break
            else:
                # Fallback: word-overlap match
                stem_words = set(name_stem.lower().replace("_", " ").split())
                best_score, best_name = 0, None
                for ad_name in perf_map:
                    ad_words = set(ad_name.lower().split())
                    score = len(stem_words & ad_words)
                    if score > best_score:
                        best_score, best_name = score, ad_name
                if best_name and best_score > 0:
                    matched_ad_name = best_name
                    category = perf_map[best_name]

        print(f"  Analysing: {filename}  [{category}]")
        try:
            description = describe_image(client, img_path, matched_ad_name)
        except Exception as e:
            description = f"(Error analysing image: {e})"
            print(f"    ERROR: {e}")

        analysed.append({
            "name": matched_ad_name,
            "filename": filename,
            "category": category,
            "description": description,
        })

    # Sort into tiers
    high = [a for a in analysed if a["category"] == "High Performer"]
    under = [a for a in analysed if a["category"] == "Underperformer"]
    average = [a for a in analysed if a["category"] == "Average"]
    no_data = [a for a in analysed if a["category"] not in ("High Performer", "Underperformer", "Average")]

    print("\nGenerating strategic comparison via GPT-4o...")
    try:
        comparison = compare_creatives(client, high, under + average, no_data)
    except Exception as e:
        comparison = f"_(Strategic comparison could not be generated: {e})_"
        print(f"  ERROR during comparison: {e}")

    # --- Build the Markdown report ---
    report_lines = [
        "# Meta Ads Visual Analysis Report",
        f"_Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}_",
        "",
        "---",
        "",
        "## Individual Ad Descriptions",
        "",
    ]

    for tier_label, tier_ads in [
        ("High Performers", high),
        ("Average", average),
        ("Underperformers", under),
        ("No Performance Data", no_data),
    ]:
        if not tier_ads:
            continue
        report_lines.append(f"### {tier_label}")
        for ad in tier_ads:
            report_lines += [
                "",
                f"#### {ad['name']}",
                f"_File: `{ad['filename']}`_",
                "",
                ad["description"],
                "",
            ]

    report_lines += [
        "---",
        "",
        "## Strategic Analysis",
        "",
        comparison,
        "",
    ]

    report_path = os.path.join("output", "analysis_report.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))

    print(f"\nAnalysis report saved to: {report_path}")



# ---------------------------------------------------------------------------
# Step 6: Generate new ad concepts from the analysis report
# ---------------------------------------------------------------------------

def generate_ad_concepts() -> None:
    """
    Reads output/analysis_report.md, sends it to GPT-4o, and writes
    10 new ad concepts to output/new_ad_concepts.md.
    """
    report_path = os.path.join("output", "analysis_report.md")
    if not os.path.exists(report_path):
        print("analysis_report.md not found — run visual analysis first.")
        return

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Skipping concept generation: OPENAI_API_KEY not set.")
        return

    with open(report_path, "r", encoding="utf-8") as f:
        report_content = f.read()

    client = OpenAI(api_key=api_key)

    print("\nGenerating 10 new ad concepts based on the analysis report...")

    system_msg = f"Je bent een senior Meta Ads creatief directeur gespecialiseerd in sieraden en lifestyle-merken. {TONE_INSTRUCTION}"
    prompt = f"""Hieronder staat een gedetailleerd visueel analyserapport van een 'Summer Sale'-campagne.

---
{report_content}
---

Genereer op basis van de winnende patronen 10 nieuwe Meta-advertentieconcepten.

Elk concept volgt dit exacte formaat:

---
### Concept [nummer]: [korte creatieve naam]

**Hook:** [Max. 10 woorden. Stopt de scroll. Voordeel-gedreven, nieuwsgierigheidsgericht of urgentie-gebaseerd.]

**Primary Text:** [2–4 zinnen advertentietekst. Conversationeel, on-brand, inclusief het aanbod.]

**Headline:** [Max. 8 woorden. Actiegericht.]

**Visuele Omschrijving:** [Gedetailleerde art direction briefing: hero-afbeelding, model (uitdrukking, pose, doelgroep), productplaatsing, kleurenpalet, typografiestijl, sfeer en grafische elementen. Specifiek genoeg dat een ontwerper het direct kan uitvoeren.]
---

Regels:
- Bouw voort op de winnende patronen: dubbel aanbod, elegante close-ups, aspirationeel model, zachte natuurtinten.
- Varieer de hooks: nieuwsgierigheid, social proof, urgentie, storytelling, voordeel-gedreven.
- Merktoon: elegant, verfijnd, toegankelijke luxe.
- Elk concept moet uniek aanvoelen.
- Schrijf alle tekst in het Nederlands."""

    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": prompt},
            ],
            max_tokens=4000,
        )
        concepts = response.choices[0].message.content
    except Exception as e:
        print(f"  ERROR generating concepts: {e}")
        return

    output_lines = [
        "# New Ad Concepts — Based on High Performer Analysis",
        f"_Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}_",
        "",
        "> These concepts are derived from the patterns that drove the best-performing ad in the Summer Sale campaign.",
        "",
        "---",
        "",
        concepts,
    ]

    output_path = os.path.join("output", "new_ad_concepts.md")
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(output_lines))

    print(f"10 new ad concepts saved to: {output_path}")


def main():
    df = load_csv()
    df, metric_used = classify_ads(df)
    print_summary(df, metric_used)

    output_path = os.path.join("output", "classified_ads.csv")
    df.to_csv(output_path, index=False)
    print(f"Classified data saved to: {output_path}")

    run_visual_analysis(df)
    generate_ad_concepts()


if __name__ == "__main__":
    main()
