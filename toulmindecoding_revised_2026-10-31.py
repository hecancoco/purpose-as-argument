# -*- coding: utf-8 -*-
"""ToulminDecoding — Purpose as Argument pipeline (Essay 2), 31 October 2026

  Step 1  SPLIT AND CATEGORISE  letter -> argument units, each with ONE of nine categories
                                 (purpose, strategy, performance, product, governance, risk,
                                 shareholder_value, sustainability_culture, other)
  Step 2  DECODE                 each PURPOSE unit -> content, substantiation and commitment
                                 components, every explicit component anchored to the source
                                 text with character offsets (inferences stored separately)
  Step 3  SCORE                  the four dimensions at the unit level and the letter / firm-year
                                 level; measures are kept separate, no composite purpose score

Lines beginning with  NOTE:  explain a design decision next to the code it concerns.

===========================================================================================
HOW TO RUN
===========================================================================================
Requirements
  * A VALID API KEY for the provider named in PROVIDER (OpenAI by default). Steps 1 and 2
    send every letter and every purpose unit to the model through the prompts below and
    cannot decode anything without it. Store the key in Colab Secrets (key icon in the
    left sidebar) as OPENAI_API_KEY, ANTHROPIC_API_KEY or GEMINI_API_KEY, or export it as
    an environment variable before running. Never paste a key into this file.
  * Packages: openai (or anthropic / google-genai), PyMuPDF, python-docx, pandas;
    pytesseract and pillow only if some PDFs are scanned.
  * In Colab, Google Drive mounted; otherwise change the paths in section 1 to local folders.

Steps
  0. Set the paths in section 1. LETTERS_DIR is the folder of original shareholder letters,
     one letter per file (.pdf or .docx). The file stem becomes letter_id
     (ACME_2021.pdf -> ACME_2021), so name the files so that firm and year can be parsed.
  1. run_step1()  — split and categorise.
     For each letter: build the canonical text and save it to LettersCanonical/<letter_id>.txt;
     send that text to the model with STEP1_SYSTEM_PROMPT; validate every unit against the
     canonical text; write Arguments/<provider>/JSON/<letter_id>.json (all units, with
     category, verbatim text, character offsets, provenance) and then
     Arguments/<provider>/CSV/<letter_id>.csv (one row per unit) for each letter.
     Letters whose JSON and CSV already exist are skipped, so the step can be re-run after
     an interruption without re-paying for finished letters.
  2. run_step2()  — decode the purpose units.
     For each letter's Step 1 JSON: send every unit with category "purpose" to the model
     with STEP2_SYSTEM_PROMPT; validate statuses and anchor every explicit span; write
     Decode/<provider>/JSON/<letter_id>.json (the decoded components) and then
     Decode/<provider>/CSV/<letter_id>.csv (one row per component) for each letter.
     Set DECODE_ALL_UNITS = True to decode the non-purpose units as well; this is needed
     only for the within-letter comparison in Step 3 and multiplies the API cost.
  3. run_step3()  — score the four dimensions. No API call.
     Reads the Step 1 and Step 2 JSON files and writes Scores/<provider>/claim_level.csv
     (one row per decoded purpose unit, nested in its firm-year), Scores/<provider>/
     firm_year.csv (one row per letter, i.e. per firm-year) and firm_panel.csv (the
     firm-year rows reduced to the main measures, for within-firm change over time).
     firm_id and year are parsed from letter_id with LETTER_ID_PATTERN or taken from
     letter_index.csv; letters that match neither keep their rows with empty firm_id / year.
  Running the file as a script (the last cell in Colab, or `python <file>.py`) performs
  the three steps in order; each step can also be called on its own.

Outputs at a glance
  LettersCanonical/<id>.txt                      frozen text; every char_start / char_end refers to it
  Arguments/<provider>/JSON|CSV/<id>.json|.csv   Step 1: units and categories
  Decode/<provider>/JSON|CSV/<id>.json|.csv      Step 2: components of the purpose units
  Scores/<provider>/claim_level.csv, firm_year.csv, firm_panel.csv   Step 3: the four dimensions
                                                 (claim within firm-year; firm-year; firm by year)

===========================================================================================
NOTES ON THE DESIGN
===========================================================================================
N1  Categories. Splitting and categorising happen in one pass. A unit is `purpose` only
    when both inclusion tests are met in the words of its claim sentence (why the firm
    exists or what it is for, and a beneficiary or outcome beyond shareholders); ESG metrics
    and targets, values lists and culture statements without that framing are
    `sustainability_culture`; returns to shareholders are `shareholder_value`. The category
    follows the claim sentence, so attached evidence or commitments never change it. A
    secondary category is recorded only with a verbatim quotation as evidence.
N2  Verbatim anchoring. Text cleaning is done once in Python, never by the model, and the
    cleaned text is frozen as the canonical letter. Every span the model returns is looked
    up in that text; found spans get character offsets, missing ones are flagged
    `not_verbatim` and are not treated as evidence. Inferred components carry an
    `inference` field and never a text span, so inferences stay separate from evidence.
N3  Step 2 schema follows Table 1 of the proposal: content (claim, end, beneficiaries),
    substantiation (grounds, warrant, backing, qualifier, rebuttal, each explicit |
    inferred | absent), commitment (aspiration | future_pledge | reported_past_action, with
    action, quantitative target and target date for pledges). Warrant, backing and rebuttal
    are coded once per unit; the warrant's appeal profile (logos / ethos / pathos) is
    recorded but never scored; speech acts use Searle's five labels.
N4  Step 3 measures. Prominence = purpose units / all units in the letter (share of words
    as a secondary line). Content = the extracted ends and beneficiaries; the domain
    taxonomy is built inductively from the `claim.end` values in a later pass (see
    classify_domains_stub) and then mapped to Rajan, Ramella and Zingales (2023).
    Substantiation = the status of each of the five components per unit, with letter
    maxima and means. Commitment = statement-type counts and a 0-3 specificity index per
    pledge, computed in Python from the recorded attributes, never by the model. Nothing is
    summed into a composite purpose score.
N5  Provenance. Every record carries run_id, provider, model, temperature, prompt_hash and
    codebook_version, so any figure can be traced to the prompt and model that produced it.
    gpt-4.1 snapshots retire on 14 January 2027: archive the outputs and pin an open-weight
    model for the reportable run.
N6  Grounds specificity counts (numerals, dates) are computed in Python; named-entity
    counts need spaCy and are left as a TODO.
===========================================================================================
"""

# ------------------------------------------------------------------------------------------
# 0. Packages
# ------------------------------------------------------------------------------------------
# In Colab run once:
#   !pip install -q openai anthropic google-genai PyMuPDF python-docx pytesseract pillow
# NOTE: provider SDKs and PDF libraries are imported lazily inside the functions that use
# them, so Step 3 (pure pandas) can run on a machine without them.

import os
import re
import json
import time
import hashlib
import datetime as dt
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

# ------------------------------------------------------------------------------------------
# 1. Configuration
# ------------------------------------------------------------------------------------------

def get_secret(name: str) -> Optional[str]:
    """Read a key from Colab Secrets (the key icon in the left sidebar) or the environment.
    NOTE: Steps 1 and 2 need a valid key for the chosen provider; without one every call
    returns None and nothing is decoded. Never paste a key into the notebook."""
    try:
        from google.colab import userdata  # type: ignore
        value = userdata.get(name)
        if value:
            return value
    except Exception:
        pass
    return os.environ.get(name)


PROVIDER = "openai"                 # "openai" | "anthropic" | "gemini"
MODEL = "gpt-4.1-2025-04-14"        # NOTE: this snapshot retires 14 Jan 2027; pin an open-weight model for the reportable run.
TEMPERATURE = 0.0
CODEBOOK_VERSION = "1.2"            # the frozen category / component definitions behind this run
RUN_ID = dt.datetime.utcnow().strftime("run_%Y%m%d_%H%M")

DECODE_ALL_UNITS = False            # NOTE: True -> decode non-purpose units too (within-letter comparison)
MAX_ATTEMPTS = 5

BASE = Path("/content/drive/MyDrive/FemaleLeaders_Purpose/Data")
LETTERS_DIR = BASE / "Letters"                  # INPUT: the original shareholder letters, one .pdf or .docx each
CANON_DIR = BASE / "LettersCanonical"           # Step 1 output: frozen canonical text per letter
UNITS_DIR = BASE / "Arguments" / PROVIDER       # Step 1 output: JSON then CSV per letter
DECODE_DIR = BASE / "Decode" / PROVIDER         # Step 2 output: JSON then CSV per letter
SCORES_DIR = BASE / "Scores" / PROVIDER         # Step 3 output: claim_level.csv, firm_year.csv

for d in (CANON_DIR, UNITS_DIR / "JSON", UNITS_DIR / "CSV", DECODE_DIR / "JSON", DECODE_DIR / "CSV", SCORES_DIR):
    try:
        d.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass  # running outside Colab (e.g. to test Step 3 helpers)

# How to get firm_id and year out of a letter_id (the file stem). Default: "<firm>_<year>",
# e.g. ACME_2021.pdf -> firm_id "ACME", year 2021. If your filenames follow another pattern,
# change the regex (it must define the named groups firm_id and year) or put a lookup table
# at LETTER_INDEX_CSV with the columns letter_id, firm_id, year; the table wins when present.
LETTER_ID_PATTERN = r"^(?P<firm_id>.+?)[_\- ](?P<year>(?:19|20)\d{2})$"
LETTER_INDEX_CSV = BASE / "letter_index.csv"

CATEGORIES = ["purpose", "strategy", "performance", "product", "governance", "risk",
              "shareholder_value", "sustainability_culture", "other"]
COMPONENTS = ["grounds", "warrant", "backing", "qualifier", "rebuttal"]   # the substantiation dimension


def mount_drive() -> None:
    try:
        from google.colab import drive  # type: ignore
        drive.mount("/content/drive")
    except Exception:
        print("Not in Colab; using local paths.")


# ------------------------------------------------------------------------------------------
# 2. Helpers: LLM call, JSON parsing, text canonicalisation, span anchoring
# ------------------------------------------------------------------------------------------

def prompt_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]


def llm_call(system_prompt: str, user_prompt: str, provider: str = PROVIDER, model: str = MODEL,
             temperature: float = TEMPERATURE, max_attempts: int = MAX_ATTEMPTS) -> Optional[str]:
    """One dispatcher for all providers. Returns the raw text of the reply or None.
    Requires a valid key for `provider` (see get_secret). JSON mode is requested where the
    provider offers it; both prompts ask for a JSON OBJECT."""
    attempt = 0
    while attempt < max_attempts:
        try:
            if provider == "openai":
                from openai import OpenAI
                client = OpenAI(api_key=get_secret("OPENAI_API_KEY"))
                resp = client.chat.completions.create(
                    model=model, temperature=temperature,
                    response_format={"type": "json_object"},
                    messages=[{"role": "system", "content": system_prompt},
                              {"role": "user", "content": user_prompt}])
                return resp.choices[0].message.content
            if provider == "anthropic":
                from anthropic import Anthropic
                client = Anthropic(api_key=get_secret("ANTHROPIC_API_KEY"))
                resp = client.messages.create(
                    model=model, temperature=temperature, max_tokens=8192, system=system_prompt,
                    messages=[{"role": "user", "content": user_prompt}])
                return "".join(block.text for block in resp.content if getattr(block, "type", "") == "text")
            if provider == "gemini":
                from google import genai
                from google.genai.types import GenerateContentConfig
                client = genai.Client(api_key=get_secret("GEMINI_API_KEY"))
                resp = client.models.generate_content(
                    model=model, contents=f"{system_prompt}\n\n{user_prompt}",
                    config=GenerateContentConfig(temperature=temperature, response_mime_type="application/json"))
                return resp.text
            raise ValueError(f"unknown provider {provider}")
        except Exception as e:  # network errors, rate limits, malformed replies
            attempt += 1
            print(f"[llm_call] attempt {attempt} failed: {e}")
            if attempt >= max_attempts:
                return None
            time.sleep(min(2 ** attempt, 60))
    return None


def fix_json_string(data: str) -> Tuple[bool, Any, str]:
    """Last-resort repair of a malformed reply (quoting keys, trailing commas, unbalanced brackets)."""
    try:
        return True, json.loads(data), "valid"
    except json.JSONDecodeError:
        content = data
        content = re.sub(r"([{,]\s*)(\w+)(:)", r'\1"\2"\3', content)
        content = re.sub(r",(\s*[}\]])", r"\1", content)
        stack, balanced = [], ""
        for ch in content:
            if ch in "[{":
                stack.append(ch); balanced += ch
            elif ch in "]}":
                if stack and ((ch == "]" and stack[-1] == "[") or (ch == "}" and stack[-1] == "{")):
                    stack.pop(); balanced += ch
            else:
                balanced += ch
        while stack:
            balanced += "}" if stack.pop() == "{" else "]"
        try:
            return True, json.loads(balanced), "repaired"
        except json.JSONDecodeError as e:
            return False, None, f"could not repair: {e}"


def parse_json_response(raw: Optional[str]) -> Optional[Dict[str, Any]]:
    """Strip code fences, parse, repair if needed. Never alters whitespace inside values."""
    if not raw:
        return None
    text = raw.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    start = text.find("{")
    if start > 0:
        text = text[start:]
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        ok, parsed, msg = fix_json_string(text)
        if not ok:
            print(f"[parse_json_response] {msg}")
            return None
        return parsed


def read_docx(path: Path) -> str:
    from docx import Document
    return "\n".join(p.text for p in Document(str(path)).paragraphs)


def read_pdf(path: Path) -> str:
    """Text extraction with OCR fallback for scanned pages."""
    import fitz  # PyMuPDF
    out = []
    with fitz.open(str(path)) as pdf:
        for page in pdf:
            t = page.get_text("text")
            if t.strip():
                out.append(t)
            else:
                try:
                    import io
                    import pytesseract
                    from PIL import Image
                    for img in page.get_images(full=True):
                        base = pdf.extract_image(img[0])
                        out.append(pytesseract.image_to_string(Image.open(io.BytesIO(base["image"]))))
                except Exception as e:
                    print(f"[read_pdf] OCR failed on {path.name}: {e}")
    return "\n".join(out)


def canonicalize(raw: str) -> str:
    """Deterministic cleaning, done ONCE in Python. The result is the frozen text that every
    offset refers to. NOTE: cleaning is never delegated to the model, otherwise its 'verbatim'
    output could not be traced back to the source letter."""
    t = raw.replace("\r", "\n")
    t = re.sub(r"(\w)-\n\s*(\w)", r"\1\2", t)           # re-join words hyphenated at line ends
    t = re.sub(r"\[\d{1,3}\]|\(\d{1,3}\)", "", t)         # citation markers [1] (2)
    t = re.sub(r"\n\s*\d{1,3}\s*\n", "\n", t)             # bare page numbers on their own line
    t = re.sub(r"[ \t]+", " ", t)
    t = re.sub(r" *\n *", "\n", t)
    t = re.sub(r"\n{3,}", "\n\n", t)
    return t.strip()


def load_canonical_text(letter_id: str) -> str:
    """Read the frozen canonical text, building it from the source file on first use."""
    canon = CANON_DIR / f"{letter_id}.txt"
    if canon.exists():
        return canon.read_text(encoding="utf-8")
    src_pdf, src_docx = LETTERS_DIR / f"{letter_id}.pdf", LETTERS_DIR / f"{letter_id}.docx"
    raw = read_pdf(src_pdf) if src_pdf.exists() else read_docx(src_docx)
    text = canonicalize(raw)
    canon.write_text(text, encoding="utf-8")
    return text


def find_span(span: Optional[str], text: str, start_at: int = 0) -> Optional[Tuple[int, int]]:
    """Locate a model-returned span in the canonical text, tolerating whitespace and quote
    differences. Returns (char_start, char_end) with end exclusive, or None.
    This is the extractive validator: a span that cannot be found is flagged, never trusted."""
    if not span or not span.strip():
        return None
    tokens = [re.escape(tok) for tok in span.split()]
    pattern = r"\s+".join(tokens)
    pattern = pattern.replace(re.escape("'"), "['’]").replace(re.escape('"'), '["“”]')
    m = re.compile(pattern).search(text, start_at) or re.compile(pattern, re.IGNORECASE).search(text, start_at)
    return (m.start(), m.end()) if m else None


def anchor(obj: Dict[str, Any], text: str, base_offset: int = 0, start_at: int = 0) -> Dict[str, Any]:
    """Add char_start / char_end / anchor_status to a component dict that has a 'text' field."""
    span = obj.get("text")
    if not span:
        obj["char_start"] = obj["char_end"] = None
        obj["anchor_status"] = "no_text"
        return obj
    hit = find_span(span, text, start_at)
    if hit:
        obj["char_start"], obj["char_end"] = base_offset + hit[0], base_offset + hit[1]
        obj["anchor_status"] = "verbatim"
    else:
        obj["char_start"] = obj["char_end"] = None
        obj["anchor_status"] = "not_verbatim"     # NOTE: re-prompt or review; do not score as evidence
    return obj


def atomic_write_json(path: Path, data: Any) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    tmp.replace(path)


def provenance(prompt: str) -> Dict[str, Any]:
    return {"run_id": RUN_ID, "provider": PROVIDER, "model": MODEL, "temperature": TEMPERATURE,
            "prompt_hash": prompt_hash(prompt), "codebook_version": CODEBOOK_VERSION}


_letter_index_cache: Optional[Dict[str, Dict[str, Any]]] = None


def parse_letter_id(letter_id: str) -> Dict[str, Any]:
    """Return {"firm_id", "year"} for a letter, from the lookup table if present, else from
    LETTER_ID_PATTERN. Both None when neither applies (the row is kept; fix the filename)."""
    global _letter_index_cache
    if _letter_index_cache is None:
        _letter_index_cache = {}
        try:
            if LETTER_INDEX_CSV.exists():
                idx = pd.read_csv(LETTER_INDEX_CSV, dtype=str)
                for _, r in idx.iterrows():
                    _letter_index_cache[str(r["letter_id"])] = {"firm_id": r.get("firm_id"), "year": int(r["year"]) if str(r.get("year", "")).isdigit() else None}
        except Exception as e:
            print(f"[parse_letter_id] could not read {LETTER_INDEX_CSV}: {e}")
    if letter_id in _letter_index_cache:
        return _letter_index_cache[letter_id]
    m = re.match(LETTER_ID_PATTERN, letter_id)
    return {"firm_id": m.group("firm_id"), "year": int(m.group("year"))} if m else {"firm_id": None, "year": None}


def count_numerals(text: str) -> int:
    return len(re.findall(r"(?<![\w.])\d[\d,]*(?:\.\d+)?%?", text or ""))


def count_dates(text: str) -> int:
    years = re.findall(r"\b(?:19|20)\d{2}\b", text or "")
    months = re.findall(r"\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\b", text or "")
    return len(years) + len(months)


# ------------------------------------------------------------------------------------------
# 3. STEP 1 — split and categorise
# ------------------------------------------------------------------------------------------
# NOTE: one pass; nine categories; the category follows the CLAIM sentence's frame.
# NOTE: the model receives already-cleaned text and must copy units verbatim (see N2).

STEP1_SYSTEM_PROMPT = """
You are an expert discourse analyst coding a shareholder letter for a research project on corporate purpose.

# Task
Split the letter into ARGUMENT UNITS and give each unit ONE category. Do both in a single pass.

# What an argument unit is
- One claim sentence (the main point) plus the sentences attached to it: its support (evidence, reasons, examples), its exceptions or caveats, and any commitment coordinated with it (for example "and we will fund 20 more by 2026").
- A unit must stand alone. Never output a claim without the support that follows it; if the next 1 to 3 sentences support or elaborate the same claim, they belong to the unit.
- Start a new unit when the text moves to a different main claim or topic, when a new piece of evidence supports a different claim, or at a section break or bullet that introduces a distinct point.
- Do not split a claim from a following sentence that carries numbers, dates, product or programme names, causal connectors (because, due to, driven by, which led to, resulting in) or regulatory actions; these almost always belong to the same unit.
- Greetings, closings, headings, signatures and other non-argument text are NOT units. Leave them out.
- Copy text VERBATIM from the input. Do not clean, paraphrase, re-punctuate or merge words; the text is already cleaned.

# Categories: assign exactly ONE primary category, decided by the main frame of the CLAIM sentence, not by its supporting sentences or attached commitments
- purpose: the claim states why the firm exists or what it is for, beyond financial return, AND names a beneficiary or an outcome beyond shareholders. Both tests must be met in the words of the claim sentence. Never infer purpose from the topic alone. Attached evidence or commitments do not change the category.
- strategy: how the firm competes: positioning, growth plans, strategic priorities, vision and strategic intent, forecasts about its direction.
- performance: reported financial or operating results of the past period (revenue, profit, EPS, margins, volumes), financial targets, and how the company runs its operations (supply chain, cost control, capital allocation).
- product: products, services, R&D pipeline, scientific data, technology, innovation.
- governance: board, leadership changes, executive pay, transparency, compliance.
- risk: challenges, headwinds, setbacks, controversies, uncertainties.
- shareholder_value: returns to shareholders: dividends, buybacks, value creation framed for shareholders.
- sustainability_culture: environmental, social, employee, culture and values content WITHOUT purpose framing: ESG metrics and dated ESG targets, diversity and workforce figures, lists of values, culture statements. The same content framed as why the firm exists is purpose.
- other: an argument that fits none of the above.

# Secondary category
- category_secondary is null unless a SECOND frame is explicit inside the claim sentence itself. If so, name it and quote its exact words in secondary_evidence. Thematic relatedness is never enough.

# Output
Return ONLY a JSON object with this exact structure, no prose before or after:
{
  "units": [
    {
      "argument_id": 1,
      "category": "purpose",
      "category_secondary": null,
      "secondary_evidence": null,
      "claim_sentence": "verbatim claim sentence",
      "text": "verbatim text of the whole unit",
      "justification": "one sentence on why this category"
    }
  ]
}
"""


def validate_units(units: List[Dict[str, Any]], letter_text: str) -> List[Dict[str, Any]]:
    """Anchor each unit and its claim sentence to the canonical text; normalise the category."""
    cursor = 0
    for u in units:
        cat = str(u.get("category", "other")).strip().lower().replace(" ", "_")
        u["category"] = cat if cat in CATEGORIES else "other"
        sec = u.get("category_secondary")
        if sec:
            sec = str(sec).strip().lower().replace(" ", "_")
            u["category_secondary"] = sec if (sec in CATEGORIES and u.get("secondary_evidence")) else None
            if u["category_secondary"] is None:
                u["secondary_evidence"] = None      # secondary without quoted evidence is dropped
        anchor(u, letter_text, start_at=cursor)
        if u["anchor_status"] == "verbatim":
            cursor = u["char_start"]                # units come in reading order; keep searching forward
        claim = {"text": u.get("claim_sentence")}
        anchor(claim, letter_text, start_at=u["char_start"] or 0)
        u["claim_char_start"], u["claim_char_end"] = claim["char_start"], claim["char_end"]
        u["claim_anchor_status"] = claim["anchor_status"]
        u["n_words"] = len((u.get("text") or "").split())
    return units


def run_step1(letter_ids: Optional[List[str]] = None) -> None:
    letter_ids = letter_ids or sorted({p.stem for p in LETTERS_DIR.iterdir() if p.suffix.lower() in (".pdf", ".docx")})
    skipped = processed = failed = 0
    for lid in letter_ids:
        json_path, csv_path = UNITS_DIR / "JSON" / f"{lid}.json", UNITS_DIR / "CSV" / f"{lid}.csv"
        if json_path.exists() and csv_path.exists():
            skipped += 1
            continue
        try:
            text = load_canonical_text(lid)
            # The canonical TEXT of the letter is sent to the model (requires a valid API key).
            raw = llm_call(STEP1_SYSTEM_PROMPT, f"Shareholder letter:\n\n{text}")
            parsed = parse_json_response(raw)
            if not parsed or "units" not in parsed:
                raise ValueError("no 'units' in model reply")
            units = validate_units(parsed["units"], text)
            n_bad = sum(u["anchor_status"] != "verbatim" for u in units)
            record = {"letter_id": lid, "n_words_letter": len(text.split()), "n_units": len(units),
                      "n_units_not_verbatim": n_bad, **provenance(STEP1_SYSTEM_PROMPT), "units": units}
            atomic_write_json(json_path, record)
            pd.DataFrame(units).assign(letter_id=lid).to_csv(csv_path, index=False)
            print(f"[OK] {lid}: {len(units)} units, {n_bad} not verbatim")
            processed += 1
        except Exception as e:
            print(f"[ERROR] {lid}: {e}")
            failed += 1
    print(f"Step 1 summary: processed={processed}, skipped={skipped}, failed={failed}")


# ------------------------------------------------------------------------------------------
# 4. STEP 2 — decode purpose units (content, substantiation, commitment)
# ------------------------------------------------------------------------------------------
# NOTE: schema follows Table 1 (see N3). Three statuses per component; inferred components
# carry an `inference`, never a `text` span. Warrant / backing / rebuttal are coded per unit.

STEP2_SYSTEM_PROMPT = """
You are an expert discourse analyst. You receive ONE argument unit from a shareholder letter. Decode it into the components below and return ONE JSON object.

# Principles
- Verbatim spans: every "text" field is an exact substring of the input unit. Never paraphrase inside "text".
- Status of each component: "explicit" (stated in the text: fill "text", leave "inference" null), "inferred" (not stated, but reconstructable: fill "inference" in your own words, leave "text" null), "absent" (neither: both null). The claim, beneficiaries, grounds and commitments can only be explicit or absent; never infer them.
- When in doubt between explicit and inferred, choose inferred. When in doubt between inferred and absent, choose absent. Do not invent facts.

# Components
A. CONTENT
- claim: the sentence stating the end the firm says it exists to advance (or, in a non-purpose unit, its main point). "end": that end in at most 12 words, in the text's own terms, no new facts. "speech_act": assertive | directive | commissive | expressive | declaration.
- beneficiaries: each stakeholder or group the unit says the purpose is for. List only those named in the text, each with its verbatim span and a short label (for example "patients", "rural households", "employees").

B. SUBSTANTIATION (Toulmin)
- grounds: each distinct piece of support offered for the claim: reported actions, outcomes, figures, examples, external facts. One object per distinct item, in text order. evidence_type: own_action | own_outcome | external_fact | example | other.
- warrant: the reasoning that connects the firm's activities or grounds to the end. Often inferred. appeal: ONLY for an explicit warrant, three true/false fields: logos (reason, cause and effect), ethos (the firm's character, identity or credibility), pathos (emotion, aspiration); all three null when the warrant is not explicit.
- backing: support offered for the warrant itself (research, expert consensus, assurance, track record). source: internal | audited_or_assured | third_party | null.
- qualifier: words that limit the strength or scope of the claim ("we believe", "in most markets", "where possible"). List each hedge as a verbatim span; status is explicit if any, else absent.
- rebuttal: an acknowledged exception or condition under which the claim may not hold.

C. COMMITMENT
- commitments: every clause in the unit that says what the firm has done or will do. type: aspiration (hoped-for state with no specified action) | future_pledge (commits to a future action) | reported_past_action (an action or outcome that has happened). For a future_pledge also give specifies_action (true/false), quantitative_target (the quantity as written, else null) and target_date (as written, else null); for the other types set these three to null.

# Output: ONLY this JSON object, exact keys
{
  "claim": {"text": "...", "end": "...", "speech_act": "assertive"},
  "beneficiaries": [{"text": "...", "label": "..."}],
  "grounds": [{"ground_id": "G1", "text": "...", "evidence_type": "own_outcome"}],
  "warrant": {"status": "explicit|inferred|absent", "text": null, "inference": null, "appeal": {"logos": null, "ethos": null, "pathos": null}},
  "backing": {"status": "explicit|inferred|absent", "text": null, "inference": null, "source": null},
  "qualifier": {"status": "explicit|absent", "hedges": [{"text": "..."}]},
  "rebuttal": {"status": "explicit|inferred|absent", "text": null, "inference": null},
  "commitments": [{"text": "...", "type": "future_pledge", "specifies_action": true, "quantitative_target": "5,000 clinics", "target_date": "end of 2027"}]
}
"""


def _norm_status(obj: Optional[Dict[str, Any]], allow_inferred: bool = True) -> Dict[str, Any]:
    obj = dict(obj or {})
    status = str(obj.get("status", "absent")).lower()
    if status not in ("explicit", "inferred", "absent"):
        status = "absent"
    if status == "explicit" and not obj.get("text"):
        status = "inferred" if (allow_inferred and obj.get("inference")) else "absent"
    if status == "inferred" and not allow_inferred:
        status = "absent"
    if status != "explicit":
        obj["text"] = None
    if status != "inferred":
        obj["inference"] = None
    obj["status"] = status
    return obj


def validate_decoded(dec: Dict[str, Any], unit_text: str, letter_text: str, unit_offset: Optional[int]) -> Dict[str, Any]:
    """Normalise statuses, anchor every explicit span to the letter (offsets are letter-level
    when the unit itself was anchored), compute deterministic specificity counts."""
    base = unit_offset or 0
    haystack = letter_text if unit_offset is not None else unit_text

    def anc(o: Dict[str, Any]) -> Dict[str, Any]:
        return anchor(o, haystack, base_offset=0, start_at=base if unit_offset is not None else 0)

    claim = dec.get("claim") or {}
    claim.setdefault("text", None); claim.setdefault("end", None)
    claim["speech_act"] = str(claim.get("speech_act", "assertive")).lower()
    dec["claim"] = anc(claim)
    dec["beneficiaries"] = [anc(dict(b)) for b in (dec.get("beneficiaries") or []) if b.get("text")]
    grounds = []
    for i, g in enumerate(dec.get("grounds") or [], 1):
        if not g.get("text"):
            continue
        g = anc(dict(g)); g["ground_id"] = g.get("ground_id") or f"G{i}"
        g["n_numerals"], g["n_dates"] = count_numerals(g["text"]), count_dates(g["text"])   # counted in Python, not by the model (N6)
        grounds.append(g)
    dec["grounds"] = grounds
    w = _norm_status(dec.get("warrant")); w = anc(w) if w["status"] == "explicit" else w
    appeal = w.get("appeal") or {}
    w["appeal"] = {k: (appeal.get(k) if w["status"] == "explicit" else None) for k in ("logos", "ethos", "pathos")}
    dec["warrant"] = w
    b = _norm_status(dec.get("backing")); b = anc(b) if b["status"] == "explicit" else b
    b["source"] = b.get("source") if b["status"] == "explicit" else None
    dec["backing"] = b
    hedges = [anc(dict(h)) for h in ((dec.get("qualifier") or {}).get("hedges") or []) if h.get("text")]
    dec["qualifier"] = {"status": "explicit" if hedges else "absent", "hedges": hedges}
    r = _norm_status(dec.get("rebuttal")); r = anc(r) if r["status"] == "explicit" else r
    dec["rebuttal"] = r
    commitments = []
    for c in dec.get("commitments") or []:
        if not c.get("text"):
            continue
        c = anc(dict(c))
        c["type"] = c.get("type") if c.get("type") in ("aspiration", "future_pledge", "reported_past_action") else "aspiration"
        if c["type"] != "future_pledge":
            c["specifies_action"] = c["quantitative_target"] = c["target_date"] = None
        else:
            c["specifies_action"] = bool(c.get("specifies_action"))
            c["quantitative_target"] = c.get("quantitative_target") or None
            c["target_date"] = c.get("target_date") or None
        commitments.append(c)
    dec["commitments"] = commitments
    return dec


def decoded_to_rows(letter_id: str, unit: Dict[str, Any], dec: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Flatten one decoded unit into one row per component, for the per-letter CSV."""
    rows: List[Dict[str, Any]] = []

    def add(role: str, status: str, text: Optional[str], obj: Dict[str, Any], **extra: Any) -> None:
        rows.append({"letter_id": letter_id, "argument_id": unit["argument_id"], "category": unit["category"],
                     "role": role, "status": status, "text": text,
                     "char_start": obj.get("char_start"), "char_end": obj.get("char_end"),
                     "anchor_status": obj.get("anchor_status"), "inference": obj.get("inference"), **extra})

    c = dec["claim"]; add("claim", "explicit" if c.get("text") else "absent", c.get("text"), c, end=c.get("end"), speech_act=c.get("speech_act"))
    for b in dec["beneficiaries"]:
        add("beneficiary", "explicit", b["text"], b, label=b.get("label"))
    for g in dec["grounds"]:
        add("grounds", "explicit", g["text"], g, ground_id=g["ground_id"], evidence_type=g.get("evidence_type"), n_numerals=g["n_numerals"], n_dates=g["n_dates"])
    if not dec["grounds"]:
        add("grounds", "absent", None, {})
    w = dec["warrant"]; add("warrant", w["status"], w.get("text"), w, logos=w["appeal"]["logos"], ethos=w["appeal"]["ethos"], pathos=w["appeal"]["pathos"])
    bk = dec["backing"]; add("backing", bk["status"], bk.get("text"), bk, source=bk.get("source"))
    for h in dec["qualifier"]["hedges"]:
        add("qualifier", "explicit", h["text"], h)
    if not dec["qualifier"]["hedges"]:
        add("qualifier", "absent", None, {})
    r = dec["rebuttal"]; add("rebuttal", r["status"], r.get("text"), r)
    for cm in dec["commitments"]:
        add("commitment", "explicit", cm["text"], cm, commitment_type=cm["type"], specifies_action=cm.get("specifies_action"),
            quantitative_target=cm.get("quantitative_target"), target_date=cm.get("target_date"))
    return rows


def run_step2(letter_ids: Optional[List[str]] = None) -> None:
    letter_ids = letter_ids or sorted(p.stem for p in (UNITS_DIR / "JSON").glob("*.json"))
    skipped = processed = failed = 0
    for lid in letter_ids:
        out_json, out_csv = DECODE_DIR / "JSON" / f"{lid}.json", DECODE_DIR / "CSV" / f"{lid}.csv"
        if out_json.exists() and out_csv.exists():
            skipped += 1
            continue
        try:
            step1 = json.loads((UNITS_DIR / "JSON" / f"{lid}.json").read_text(encoding="utf-8"))
            letter_text = load_canonical_text(lid)
            decoded, rows = [], []
            for u in step1["units"]:
                # Purpose units only, unless DECODE_ALL_UNITS is set for the within-letter comparison.
                if u["category"] != "purpose" and not DECODE_ALL_UNITS:
                    continue
                raw = llm_call(STEP2_SYSTEM_PROMPT, f"Category: {u['category']}\nArgument unit:\n{u['text']}")
                parsed = parse_json_response(raw)
                if not parsed:
                    print(f"[WARN] {lid} u{u['argument_id']}: unparseable reply; skipped")
                    continue
                dec = validate_decoded(parsed, u["text"], letter_text,
                                       u["char_start"] if u.get("anchor_status") == "verbatim" else None)
                decoded.append({"argument_id": u["argument_id"], "category": u["category"], "text": u["text"],
                                "char_start": u.get("char_start"), "char_end": u.get("char_end"),
                                "n_words": u.get("n_words"), "decoding": dec})
                rows.extend(decoded_to_rows(lid, u, dec))
            record = {"letter_id": lid, **provenance(STEP2_SYSTEM_PROMPT), "decode_all_units": DECODE_ALL_UNITS, "units": decoded}
            atomic_write_json(out_json, record)
            pd.DataFrame(rows).to_csv(out_csv, index=False)
            print(f"[OK] {lid}: {len(decoded)} units decoded")
            processed += 1
        except Exception as e:
            print(f"[ERROR] {lid}: {e}")
            failed += 1
    print(f"Step 2 summary: processed={processed}, skipped={skipped}, failed={failed}")


# ------------------------------------------------------------------------------------------
# 5. STEP 3 — score the four dimensions (new)
# ------------------------------------------------------------------------------------------
# NOTE: measures stay separate (see N4). Nothing here is summed into one purpose score.
# This step makes no API call; it only reads the JSON files written by Steps 1 and 2.

def unit_scores(dec: Dict[str, Any]) -> Dict[str, Any]:
    """Claim-level measures for one decoded unit."""
    status = {
        "grounds": "explicit" if dec["grounds"] else "absent",
        "warrant": dec["warrant"]["status"],
        "backing": dec["backing"]["status"],
        "qualifier": dec["qualifier"]["status"],
        "rebuttal": dec["rebuttal"]["status"],
    }
    pledges = [c for c in dec["commitments"] if c["type"] == "future_pledge"]
    # candidate specificity index: 0-3 = action + quantitative target + target date, computed here, never by the model
    spec = [int(bool(p.get("specifies_action"))) + int(p.get("quantitative_target") is not None) + int(p.get("target_date") is not None) for p in pledges]
    return {
        # content
        "claim_end": dec["claim"].get("end"),
        "beneficiary_status": "explicit" if dec["beneficiaries"] else "absent",
        "beneficiary_labels": "; ".join(b.get("label") or "" for b in dec["beneficiaries"]) or None,
        # substantiation: one column per component, plus a count of explicit components
        **{f"{k}_status": v for k, v in status.items()},
        "n_components_explicit": sum(v == "explicit" for v in status.values()),
        "n_components_inferred": sum(v == "inferred" for v in status.values()),
        "n_grounds": len(dec["grounds"]),
        "grounds_numerals": sum(g["n_numerals"] for g in dec["grounds"]),
        "grounds_dates": sum(g["n_dates"] for g in dec["grounds"]),
        "warrant_logos": dec["warrant"]["appeal"]["logos"], "warrant_ethos": dec["warrant"]["appeal"]["ethos"], "warrant_pathos": dec["warrant"]["appeal"]["pathos"],
        # commitment
        "n_aspirations": sum(c["type"] == "aspiration" for c in dec["commitments"]),
        "n_pledges": len(pledges),
        "n_reported_actions": sum(c["type"] == "reported_past_action" for c in dec["commitments"]),
        "specificity_max": max(spec) if spec else None,
        "has_dated_pledge": any(p.get("target_date") for p in pledges),
    }


def letter_scores(step1: Dict[str, Any], step2: Dict[str, Any]) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """Letter-level (firm-year) measures plus the claim-level rows.
    Levels: a letter is one firm-year; a claim-level row is one purpose unit inside that
    firm-year, so claim rows nest within firm-years, and firm-years within firms."""
    ids = parse_letter_id(step1["letter_id"])
    units = step1["units"]
    n_units = len(units)
    purpose_units = [u for u in units if u["category"] == "purpose"]
    n_words = step1.get("n_words_letter") or sum(u.get("n_words", 0) for u in units)
    purpose_words = sum(u.get("n_words", 0) for u in purpose_units)
    claim_rows = []
    for d in step2["units"]:
        claim_rows.append({"letter_id": step1["letter_id"], "firm_id": ids["firm_id"], "year": ids["year"],
                           "argument_id": d["argument_id"], "category": d["category"],
                           "char_start": d.get("char_start"), "char_end": d.get("char_end"), **unit_scores(d["decoding"])})
    p_rows = [r for r in claim_rows if r["category"] == "purpose"]
    o_rows = [r for r in claim_rows if r["category"] != "purpose"]
    mean = lambda xs: (sum(xs) / len(xs)) if xs else None
    specs = [r["specificity_max"] for r in p_rows if r["specificity_max"] is not None]
    letter = {
        "letter_id": step1["letter_id"], "firm_id": ids["firm_id"], "year": ids["year"],
        # 1 prominence: purpose claims / all claims (one claim per argument unit); share of words as a secondary line
        "n_units": n_units, "n_purpose_units": len(purpose_units),
        "prominence": (len(purpose_units) / n_units) if n_units else None,
        "purpose_share_words": (purpose_words / n_words) if n_words else None,
        # 2 content
        "claim_ends": " | ".join(r["claim_end"] or "" for r in p_rows) or None,
        "beneficiary_labels": " | ".join(r["beneficiary_labels"] or "" for r in p_rows) or None,
        "share_beneficiary_explicit": mean([r["beneficiary_status"] == "explicit" for r in p_rows]),
        # 3 substantiation
        "subst_explicit_max": max([r["n_components_explicit"] for r in p_rows], default=None),
        "subst_explicit_mean": mean([r["n_components_explicit"] for r in p_rows]),
        **{f"share_{c}_explicit": mean([r[f"{c}_status"] == "explicit" for r in p_rows]) for c in COMPONENTS},
        # 4 commitment
        "n_pledges": sum(r["n_pledges"] for r in p_rows),
        "n_dated_pledges": sum(r["has_dated_pledge"] for r in p_rows),
        "n_aspirations": sum(r["n_aspirations"] for r in p_rows),
        "specificity_max": max(specs) if specs else None,
        "specificity_mean": mean(specs),
        # within-letter comparison (only when non-purpose units were decoded with DECODE_ALL_UNITS)
        "baseline_other_units_decoded": len(o_rows),
        "baseline_subst_purpose_minus_other": (mean([r["n_components_explicit"] for r in p_rows]) - mean([r["n_components_explicit"] for r in o_rows])) if (p_rows and o_rows) else None,
        "codebook_version": step1.get("codebook_version"), "run_id_step1": step1.get("run_id"), "run_id_step2": step2.get("run_id"),
    }
    return letter, claim_rows


def run_step3() -> None:
    letters, claims = [], []
    for p in sorted((DECODE_DIR / "JSON").glob("*.json")):
        step1_path = UNITS_DIR / "JSON" / p.name
        if not step1_path.exists():
            continue
        step1 = json.loads(step1_path.read_text(encoding="utf-8"))
        step2 = json.loads(p.read_text(encoding="utf-8"))
        L, rows = letter_scores(step1, step2)
        letters.append(L); claims.extend(rows)
    claims_df = pd.DataFrame(claims).sort_values(["firm_id", "year", "letter_id", "argument_id"], na_position="last")
    letters_df = pd.DataFrame(letters).sort_values(["firm_id", "year", "letter_id"], na_position="last")
    claims_df.to_csv(SCORES_DIR / "claim_level.csv", index=False)     # one row per purpose unit within a firm-year
    letters_df.to_csv(SCORES_DIR / "firm_year.csv", index=False)      # one row per letter = firm-year
    # Firm-level panel for within-firm change over time: one row per firm-year, pivoted by year as needed.
    if letters_df["firm_id"].notna().any():
        panel_cols = ["firm_id", "year", "n_units", "n_purpose_units", "prominence", "purpose_share_words",
                      "subst_explicit_max", "subst_explicit_mean", "specificity_max", "n_dated_pledges"]
        letters_df[panel_cols].to_csv(SCORES_DIR / "firm_panel.csv", index=False)
    n_unparsed = int(letters_df["firm_id"].isna().sum())
    if n_unparsed:
        print(f"[WARN] {n_unparsed} letter_id(s) did not match LETTER_ID_PATTERN and have no firm_id / year.")
    print(f"Step 3: {len(letters)} letters, {len(claims)} decoded units -> {SCORES_DIR}")


def classify_domains_stub() -> None:
    """Content dimension, second pass (TODO).
    1. Collect all `claim_end` strings from claim_level.csv.
    2. Build the domain taxonomy inductively (cluster / card-sort / LLM-assisted grouping, then review).
    3. Map each inductive domain to Rajan, Ramella and Zingales (2023) and document convergences.
    4. Code beneficiaries into categories separately from the end; keep explicit / inferred status.
    Only after the taxonomy is frozen should a classification prompt assign domains (multi-label)."""
    raise NotImplementedError


# ------------------------------------------------------------------------------------------
# 6. Run
# ------------------------------------------------------------------------------------------
if __name__ == "__main__":
    mount_drive()
    run_step1()
    run_step2()
    run_step3()
