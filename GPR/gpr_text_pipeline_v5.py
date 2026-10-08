"""
gpr_text_pipeline_v4.py

Corrected pipeline for constructing firm-level Geopolitical Risk (GPR) perception indices
from earnings call transcripts.

Key changes from v3:
1. GPR sentences = sentences that DIRECTLY CONTAIN GPR keywords (not ±K context window)
2. Country dictionary supports variants/demonyms (e.g., "Chinese" -> "china")
3. Added year filtering for 2015-2017 analysis
4. Clearer separation of baseline vs modifier-enhanced measures

Outputs:
- sentence_level.csv: All sentences with GPR flags
- is_gpr_sentences_country.csv: GPR sentences linked to countries
- firm_country_year.csv: I_GPR_ict, Frac_GPR_ict, Word_GPR_ict
- firm_year.csv: I_GPR_it, Frac_GPR_it
- firm_year_2015_2017.csv: Same as firm_year but filtered to 2015-2017
"""

import os
import re
import glob
import pandas as pd
import xml.etree.ElementTree as ET
from datetime import datetime
from typing import Dict, Set, Tuple, List, Optional

# ---------------------------------------------------
# CONFIGURATION (EDIT THESE PATHS)
# ---------------------------------------------------

XML_DIR = "/path/to/xml_transcripts"          # folder with XML transcript files

# Dictionary files
GPR_DICT_FILE = "gpr_dictionary.csv"          # columns: term, category
COUNTRY_DICT_FILE = "country_variants.csv"    # columns: country, variant

# Output directory
OUTPUT_DIR = "output"

# Column names in dictionary files
GPR_TERM_COL = "term"
GPR_CAT_COL = "category"
COUNTRY_CODE_COL = "country"
COUNTRY_VARIANT_COL = "variant"

# Year range for filtered analysis
YEAR_FILTER_START = 2015
YEAR_FILTER_END = 2017

# ---------------------------------------------------
# TEXT UTILITIES
# ---------------------------------------------------

def clean_text(text: str) -> str:
    """Remove extra whitespace and normalize text."""
    if text is None:
        return ""
    text = text.replace("\r", " ").replace("\n", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def normalize_lower(text: str) -> str:
    """Clean and lowercase text."""
    return clean_text(text).lower()


def split_into_sentences(text: str) -> List[str]:
    """
    Split text into sentences.
    Uses simple regex; consider nltk.sent_tokenize for better accuracy.
    """
    text = text.strip()
    if not text:
        return []
    # Split on sentence-ending punctuation followed by space
    sentence_end_re = re.compile(r"(?<=[.?!])\s+")
    sents = sentence_end_re.split(text)
    return [s.strip() for s in sents if s.strip()]


def tokenize_words(sentence: str) -> List[str]:
    """Tokenize sentence into words."""
    cleaned = re.sub(r"[^a-zA-Z0-9']+", " ", sentence)
    return [t for t in cleaned.split() if t]


def count_words(sentence: str) -> int:
    """Count words in a sentence."""
    return len(tokenize_words(sentence))


def compile_term_regex(terms: Set[str]) -> re.Pattern:
    """
    Create a compiled regex pattern for matching any term in the set.
    Supports multi-word terms and ensures whole-word matching.
    """
    if not terms:
        return re.compile(r"(?!x)x")  # matches nothing
    # Sort by length (longest first) to match multi-word terms first
    escaped = sorted((re.escape(t) for t in terms), key=len, reverse=True)
    pattern = r"\b(?:" + "|".join(escaped) + r")\b"
    return re.compile(pattern, re.IGNORECASE)


# ---------------------------------------------------
# LOAD DICTIONARIES
# ---------------------------------------------------

def load_gpr_dictionary(path: str) -> Tuple[Set[str], Set[str], Dict[str, str]]:
    """
    Load GPR keyword dictionary.
    
    Returns:
        all_terms: Set of all GPR terms (lowercased)
        modifier_terms: Set of terms in 'Modifier' category
        term_to_category: Dict mapping term -> category
    """
    if path.lower().endswith((".xlsx", ".xls")):
        df = pd.read_excel(path)
    else:
        df = pd.read_csv(path, sep=None, engine='python')  # auto-detect delimiter

    df = df[[GPR_TERM_COL, GPR_CAT_COL]].dropna()
    df[GPR_TERM_COL] = df[GPR_TERM_COL].astype(str).str.strip().str.lower()
    df[GPR_CAT_COL] = df[GPR_CAT_COL].astype(str).str.strip()

    all_terms = set(df[GPR_TERM_COL].unique())
    
    modifier_mask = df[GPR_CAT_COL].str.lower().str.contains("modifier", na=False)
    modifier_terms = set(df.loc[modifier_mask, GPR_TERM_COL].unique())
    
    term_to_category = dict(zip(df[GPR_TERM_COL], df[GPR_CAT_COL]))
    
    print(f"  Loaded {len(all_terms)} GPR terms ({len(modifier_terms)} modifiers)")
    return all_terms, modifier_terms, term_to_category


def load_country_lexicon(path: str) -> Tuple[Dict[str, str], Set[str]]:
    """
    Load country variant dictionary.
    
    Returns:
        variant_to_country: Dict mapping variant (lowercased) -> country code
        all_variants: Set of all variant strings for regex matching
    """
    if path.lower().endswith((".xlsx", ".xls")):
        df = pd.read_excel(path)
    else:
        df = pd.read_csv(path, sep=None, engine='python')

    df = df[[COUNTRY_CODE_COL, COUNTRY_VARIANT_COL]].dropna()
    df[COUNTRY_CODE_COL] = df[COUNTRY_CODE_COL].astype(str).str.strip().str.lower()
    df[COUNTRY_VARIANT_COL] = df[COUNTRY_VARIANT_COL].astype(str).str.strip().str.lower()

    variant_to_country: Dict[str, str] = {}
    for _, row in df.iterrows():
        variant_to_country[row[COUNTRY_VARIANT_COL]] = row[COUNTRY_CODE_COL]
    
    all_variants = set(variant_to_country.keys())
    
    countries = set(variant_to_country.values())
    print(f"  Loaded {len(all_variants)} country variants for {len(countries)} countries")
    return variant_to_country, all_variants


# ---------------------------------------------------
# XML PARSING (Thomson StreetEvents Format)
# ---------------------------------------------------

def extract_raw_body_from_xml(xml_path: str) -> str:
    """Extract raw body text from XML transcript."""
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        body_elem = root.find(".//Body")
        if body_elem is not None and body_elem.text:
            return body_elem.text
    except ET.ParseError as e:
        print(f"  Warning: XML parse error in {xml_path}: {e}")
    return ""


def extract_management_text_from_xml(xml_path: str) -> str:
    """
    Extract management presentation section from XML transcript.
    Returns text between 'Presentation' and 'Questions and Answers' markers.
    """
    raw_body = extract_raw_body_from_xml(xml_path)
    if not raw_body:
        return ""
    
    body_lower = raw_body.lower()
    
    # Find presentation section start
    idx_pres = body_lower.find("presentation")
    start_idx = idx_pres if idx_pres != -1 else 0
    
    # Find Q&A section start (end of management presentation)
    idx_qna = body_lower.find("questions and answers")
    end_idx = idx_qna if idx_qna != -1 else len(raw_body)
    
    mgmt_raw = raw_body[start_idx:end_idx]
    return normalize_lower(mgmt_raw)


def extract_call_date_from_xml(xml_path: str) -> Optional[str]:
    """
    Extract call date from XML transcript.
    Parses <startDate> like '4-Nov-15 3:00pm GMT' -> 'YYYY-MM-DD'
    """
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        start_elem = root.find(".//startDate")
        if start_elem is not None and start_elem.text:
            raw = start_elem.text.strip()
            # Remove timezone info
            raw = re.sub(r"\s*(GMT|EST|PST|UTC|ET|PT).*$", "", raw, flags=re.IGNORECASE)
            # Try multiple date formats
            for fmt in ["%d-%b-%y %I:%M%p", "%d-%b-%Y %I:%M%p", "%Y-%m-%d", "%d-%b-%y"]:
                try:
                    dt = datetime.strptime(raw.strip(), fmt)
                    return dt.strftime("%Y-%m-%d")
                except ValueError:
                    continue
    except Exception:
        pass
    return None


def extract_firm_id_from_xml(xml_path: str) -> str:
    """
    Extract firm identifier from XML transcript.
    Uses <companyTicker>, falls back to Event Id or filename.
    """
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        
        # Try company ticker first
        ticker = root.find(".//companyTicker")
        if ticker is not None and ticker.text:
            return ticker.text.strip()
        
        # Try Event Id attribute
        event_id = root.attrib.get("Id")
        if event_id:
            return event_id
    except Exception:
        pass
    
    # Fallback to filename
    return os.path.basename(xml_path).replace(".xml", "")


def extract_year_from_date(date_str: Optional[str]) -> Optional[int]:
    """Extract year as integer from date string."""
    if date_str and len(date_str) >= 4:
        try:
            return int(date_str[:4])
        except ValueError:
            pass
    return None


# ---------------------------------------------------
# CORE ANALYSIS FUNCTIONS
# ---------------------------------------------------

def sentence_contains_gpr(sentence: str, gpr_regex: re.Pattern) -> bool:
    """Check if sentence contains any GPR keyword."""
    if not sentence:
        return False
    return bool(gpr_regex.search(sentence))


def sentence_contains_modifier(sentence: str, modifier_regex: re.Pattern) -> bool:
    """Check if sentence contains a modifier term."""
    if not sentence:
        return False
    return bool(modifier_regex.search(sentence))


def find_gpr_terms_in_sentence(sentence: str, gpr_regex: re.Pattern) -> List[str]:
    """Return list of GPR terms found in sentence."""
    if not sentence:
        return []
    matches = gpr_regex.findall(sentence)
    return [m.lower() for m in matches]


def find_country_mentions(sentence: str, 
                          variant_to_country: Dict[str, str],
                          country_regex: re.Pattern) -> Set[str]:
    """
    Find all country references in a sentence.
    Returns set of country codes.
    """
    found = set()
    if not sentence:
        return found
    
    sentence_lower = sentence.lower()
    
    # Use regex to find all variant matches
    matches = country_regex.findall(sentence_lower)
    for match in matches:
        country = variant_to_country.get(match.lower())
        if country:
            found.add(country)
    
    return found


# ---------------------------------------------------
# BUILD SENTENCE-LEVEL DATASET
# ---------------------------------------------------

def build_sentence_level_dataset(xml_dir: str,
                                 gpr_regex: re.Pattern,
                                 modifier_regex: re.Pattern,
                                 variant_to_country: Dict[str, str],
                                 country_regex: re.Pattern,
                                 term_to_category: Dict[str, List[str]],
                                 context_window: int = 3) -> pd.DataFrame:
    """
    Build sentence-level dataset for all transcripts in `xml_dir` (management presentation only).

    Implements the RA instruction to:
      1) Flag seed sentences that contain any GPR keyword.
      2) Expand each seed sentence into a ±3 sentence context window.
      3) Mark the UNION of all windows as GPR sentences (deduplicating overlaps).
      4) Search for country mentions ONLY inside GPR-window sentences.

    Adds `sentence_index` (0-based) for traceability.
    """
    all_rows = []

    xml_files = sorted(glob.glob(os.path.join(xml_dir, "*_T.xml")))
    for file_path in tqdm(xml_files, desc="Building sentence-level dataset"):
        file_id = os.path.basename(file_path).replace("_T.xml", "")

        # Parse XML
        try:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                soup = BeautifulSoup(f.read(), "lxml-xml")
        except Exception:
            continue

        # Metadata
        firm_id = extract_firm_id(soup)
        call_date = extract_call_date(soup)
        year = extract_year(call_date)

        # Text (management presentation only)
        full_text = extract_transcript_text(soup)
        mgmt_text = extract_management_presentation(full_text)
        mgmt_text = clean_text(mgmt_text)

        sentences = split_into_sentences(mgmt_text)
        if not sentences:
            continue

        # Pass 1: compute seed flags and per-sentence matches
        seed_flags = [0] * len(sentences)
        per_sentence = []

        for idx, sent in enumerate(sentences):
            sent_clean = sent.strip()
            terms_found = find_gpr_terms_in_sentence(sent_clean, gpr_regex)
            is_seed = 1 if terms_found else 0
            seed_flags[idx] = is_seed

            # Categories from term_to_category mapping (if available)
            cats = set()
            for t in terms_found:
                for c in term_to_category.get(t.lower(), []):
                    cats.add(c)

            per_sentence.append({
                "sentence_index": idx,
                "sentence_text": sent_clean,
                "word_count": count_words(sent_clean),
                "is_gpr_seed": is_seed,
                "gpr_terms": "|".join(sorted(set(terms_found))) if terms_found else "",
                "gpr_categories": "|".join(sorted(cats)) if cats else "",
                "has_modifier": 1 if has_gpr_modifier(sent_clean, modifier_regex) else 0,
            })

        # Pass 2: expand seed sentences to ±context_window and union
        gpr_window_flags = [0] * len(sentences)
        for idx, is_seed in enumerate(seed_flags):
            if is_seed == 1:
                start_idx = max(0, idx - context_window)
                end_idx = min(len(sentences) - 1, idx + context_window)
                for j in range(start_idx, end_idx + 1):
                    gpr_window_flags[j] = 1

        # Pass 3: country mentions only in GPR-window sentences
        for idx in range(len(per_sentence)):
            is_gpr_sentence = gpr_window_flags[idx]

            if is_gpr_sentence == 1:
                countries = find_country_mentions(
                    per_sentence[idx]["sentence_text"],
                    variant_to_country,
                    country_regex
                )
            else:
                countries = set()

            all_rows.append({
                "file_id": file_id,
                "firm_id": firm_id,
                "call_date": call_date,
                "year": year,
                "sentence_id": f"{file_id}_S{str(idx+1).zfill(5)}",
                "sentence_index": per_sentence[idx]["sentence_index"],
                "sentence_text": per_sentence[idx]["sentence_text"],
                "word_count": per_sentence[idx]["word_count"],
                "is_gpr_seed": per_sentence[idx]["is_gpr_seed"],
                "is_gpr_sentence": is_gpr_sentence,
                "gpr_terms": per_sentence[idx]["gpr_terms"],
                "gpr_categories": per_sentence[idx]["gpr_categories"],
                "has_modifier": per_sentence[idx]["has_modifier"],
                "countries_mentioned": "|".join(sorted(countries)) if countries else "",
                "country_count": len(countries),
            })

    return pd.DataFrame(all_rows)



def build_is_gpr_sentence_country_level(df_sent: pd.DataFrame) -> pd.DataFrame:
    """
    Create dataset of GPR sentences with country mentions.
    One row per sentence-country pair.
    """
    rows = []
    
    # Filter to GPR sentences with country mentions
    gpr_with_country = df_sent[
        (df_sent["is_gpr_sentence"] == 1) & 
        (df_sent["country_count"] > 0)
    ].copy()
    
    for _, row in gpr_with_country.iterrows():
        countries = row["countries_mentioned"].split("|")
        for country in countries:
            if country:
                rows.append({
                    "firm_id": row["firm_id"],
                    "transcript_id": row["transcript_id"],
                    "call_date": row["call_date"],
                    "year": row["year"],
                    "sentence_id": row["sentence_id"],
                    "country": country,
                    "sentence_text": row["sentence_text"],
                    "word_count": row["word_count"],
                    "gpr_terms": row["gpr_terms"],
                    "gpr_categories": row["gpr_categories"],
                    "has_modifier": row["has_modifier"]
                })
    
    return pd.DataFrame(rows)


def build_firm_country_year_indices(df_sent: pd.DataFrame,
                                     df_gpr_country: pd.DataFrame) -> pd.DataFrame:
    """
    Build firm-country-year level GPR indices.
    
    Indices:
    - I_GPR_ict: Indicator = 1 if any GPR mention of country c
    - Frac_GPR_ict: # GPR sentences mentioning country c / total sentences
    - Word_GPR_ict: # words in GPR sentences mentioning country c / total words
    """
    # Calculate presentation totals by firm-year
    totals = df_sent.groupby(["firm_id", "year"], dropna=False).agg(
        total_sentences=("sentence_id", "count"),
        total_words=("word_count", "sum")
    ).reset_index()
    
    if df_gpr_country.empty:
        return pd.DataFrame(columns=[
            "firm_id", "year", "country",
            "is_gpr_sentence_count", "gpr_word_count",
            "total_sentences", "total_words",
            "I_GPR_ict", "Frac_GPR_ict", "Word_GPR_ict"
        ])
    
    # Aggregate by firm-country-year
    # Use nunique for sentence_id to avoid double-counting sentences mentioning multiple terms
    country_agg = df_gpr_country.groupby(
        ["firm_id", "year", "country"], dropna=False
    ).agg(
        is_gpr_sentence_count=("sentence_id", "nunique"),
        gpr_word_count=("word_count", "sum")
    ).reset_index()
    
    # Merge with totals
    df_fcy = country_agg.merge(totals, on=["firm_id", "year"], how="left")
    
    # Calculate indices
    df_fcy["I_GPR_ict"] = (df_fcy["is_gpr_sentence_count"] > 0).astype(int)
    df_fcy["Frac_GPR_ict"] = df_fcy["is_gpr_sentence_count"] / df_fcy["total_sentences"]
    df_fcy["Word_GPR_ict"] = df_fcy["gpr_word_count"] / df_fcy["total_words"]
    
    return df_fcy


def build_firm_year_indices_simple(df_sent: pd.DataFrame,
                                    year_filter: Tuple[int, int] = None) -> pd.DataFrame:
    """
    Build firm-year level GPR indices.
    
    Indices:
    - I_GPR_it: Indicator = 1 if any GPR mention
    - Frac_GPR_it: # GPR sentences / total sentences
    - Word_GPR_it: # words in GPR sentences / total words
    
    Args:
        df_sent: Sentence-level dataframe
        year_filter: Optional (start_year, end_year) to filter results
    """
    df = df_sent.copy()
    
    # Apply year filter if specified
    if year_filter:
        start_year, end_year = year_filter
        df = df[df["year"].between(start_year, end_year)]
    
    if df.empty:
        return pd.DataFrame(columns=[
            "firm_id", "year",
            "total_sentences", "total_words",
            "is_gpr_sentence_count", "gpr_word_count",
            "I_GPR_it", "Frac_GPR_it", "Word_GPR_it"
        ])
    
    # Calculate totals
    totals = df.groupby(["firm_id", "year"], dropna=False).agg(
        total_sentences=("sentence_id", "count"),
        total_words=("word_count", "sum")
    ).reset_index()
    
    # Calculate GPR counts (only from GPR sentences)
    gpr_df = df[df["is_gpr_sentence"] == 1]
    gpr_counts = gpr_df.groupby(["firm_id", "year"], dropna=False).agg(
        is_gpr_sentence_count=("sentence_id", "count"),
        gpr_word_count=("word_count", "sum")
    ).reset_index()
    
    # Merge
    result = totals.merge(gpr_counts, on=["firm_id", "year"], how="left")
    result["is_gpr_sentence_count"] = result["is_gpr_sentence_count"].fillna(0).astype(int)
    result["gpr_word_count"] = result["gpr_word_count"].fillna(0).astype(int)
    
    # Calculate indices
    result["I_GPR_it"] = (result["is_gpr_sentence_count"] > 0).astype(int)
    result["Frac_GPR_it"] = result["is_gpr_sentence_count"] / result["total_sentences"]
    result["Word_GPR_it"] = result["gpr_word_count"] / result["total_words"]
    
    return result


# ---------------------------------------------------
# MAIN PIPELINE
# ---------------------------------------------------

def main():
    print("=" * 60)
    print("GPR Text Analysis Pipeline v4")
    print("=" * 60)
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Load dictionaries
    print("\n[1/5] Loading dictionaries...")
    all_terms, modifier_terms, term_to_category = load_gpr_dictionary(GPR_DICT_FILE)
    variant_to_country, all_variants = load_country_lexicon(COUNTRY_DICT_FILE)
    
    # Compile regex patterns
    print("\n[2/5] Compiling regex patterns...")
    gpr_regex = compile_term_regex(all_terms)
    modifier_regex = compile_term_regex(modifier_terms)
    country_regex = compile_term_regex(all_variants)
    print(f"  GPR pattern: {len(all_terms)} terms")
    print(f"  Country pattern: {len(all_variants)} variants")
    
    # Build sentence-level dataset
    print("\n[3/5] Building sentence-level dataset...")
    df_sent = build_sentence_level_dataset(
        xml_dir=XML_DIR,
        gpr_regex=gpr_regex,
        modifier_regex=modifier_regex,
        variant_to_country=variant_to_country,
        country_regex=country_regex,
        term_to_category=term_to_category
    )
    
    n_transcripts = df_sent["transcript_id"].nunique()
    n_sentences = len(df_sent)
    n_is_gpr_sentences = df_sent["is_gpr_sentence"].sum()
    print(f"  Transcripts: {n_transcripts}")
    print(f"  Total sentences: {n_sentences}")
    print(f"  GPR sentences: {n_is_gpr_sentences} ({100*n_is_gpr_sentences/n_sentences:.2f}%)")
    
    # Build GPR sentence-country dataset
    print("\n[4/5] Building aggregated datasets...")
    df_gpr_country = build_is_gpr_sentence_country_level(df_sent)
    print(f"  GPR sentence-country pairs: {len(df_gpr_country)}")
    
    # Build firm-country-year indices
    df_fcy = build_firm_country_year_indices(df_sent, df_gpr_country)
    print(f"  Firm-country-year observations: {len(df_fcy)}")
    
    # Build firm-year indices (all years)
    df_fy = build_firm_year_indices_simple(df_sent)
    print(f"  Firm-year observations: {len(df_fy)}")
    
    # Build firm-year indices (2015-2017 only)
    df_fy_filtered = build_firm_year_indices_simple(
        df_sent, 
        year_filter=(YEAR_FILTER_START, YEAR_FILTER_END)
    )
    print(f"  Firm-year observations ({YEAR_FILTER_START}-{YEAR_FILTER_END}): {len(df_fy_filtered)}")
    
    # Save outputs
    print("\n[5/5] Saving outputs...")
    
    df_sent.to_csv(os.path.join(OUTPUT_DIR, "sentence_level.csv"), index=False)
    print(f"  Saved: sentence_level.csv")
    
    df_gpr_country.to_csv(os.path.join(OUTPUT_DIR, "is_gpr_sentences_country.csv"), index=False)
    print(f"  Saved: is_gpr_sentences_country.csv")
    
    df_fcy.to_csv(os.path.join(OUTPUT_DIR, "firm_country_year.csv"), index=False)
    print(f"  Saved: firm_country_year.csv")
    
    df_fy.to_csv(os.path.join(OUTPUT_DIR, "firm_year.csv"), index=False)
    print(f"  Saved: firm_year.csv")
    
    df_fy_filtered.to_csv(
        os.path.join(OUTPUT_DIR, f"firm_year_{YEAR_FILTER_START}_{YEAR_FILTER_END}.csv"), 
        index=False
    )
    print(f"  Saved: firm_year_{YEAR_FILTER_START}_{YEAR_FILTER_END}.csv")
    
    # Print summary statistics
    print("\n" + "=" * 60)
    print("SUMMARY STATISTICS")
    print("=" * 60)
    
    if not df_fy.empty:
        print(f"\nFirm-Year Level (all years):")
        print(f"  Firms with any GPR mention: {df_fy['I_GPR_it'].sum()} / {len(df_fy)}")
        print(f"  Mean Frac_GPR_it: {df_fy['Frac_GPR_it'].mean():.4f}")
        print(f"  Median Frac_GPR_it: {df_fy['Frac_GPR_it'].median():.4f}")
    
    if not df_fy_filtered.empty:
        print(f"\nFirm-Year Level ({YEAR_FILTER_START}-{YEAR_FILTER_END}):")
        print(f"  Firms with any GPR mention: {df_fy_filtered['I_GPR_it'].sum()} / {len(df_fy_filtered)}")
        print(f"  Mean Frac_GPR_it: {df_fy_filtered['Frac_GPR_it'].mean():.4f}")
    
    if not df_fcy.empty:
        print(f"\nFirm-Country-Year Level:")
        print(f"  Unique countries mentioned: {df_fcy['country'].nunique()}")
        top_countries = df_fcy.groupby("country")["is_gpr_sentence_count"].sum().nlargest(10)
        print(f"  Top 10 countries by GPR sentence count:")
        for country, count in top_countries.items():
            print(f"    {country}: {count}")
    
    print("\n" + "=" * 60)
    print("Pipeline completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
