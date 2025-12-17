# GPR Text Pipeline (v5)

## What this does
Constructs text-based measures of CEO geopolitical risk (GPR) perceptions from Thomson StreetEvents earnings-call transcripts, using only the **management presentation** section (excluding Q&A).

## Required inputs
- Folder of extracted transcripts: `*_T.xml`
- `GPR dictionary.csv` (updated GPR keyword dictionary from Prof Ayyagari)
- `country_variants.csv` (country list with variants)

## Core methodology
1. Parse each `*_T.xml` transcript, extract management presentation text, clean, and split into sentences.
2. Identify **GPR seed sentences** using the GPR keyword dictionary.
3. Expand each seed into a **±3 sentence context window**, and mark the **union** of all windows as **GPR sentences** (overlaps are automatically deduplicated).
4. Search for country mentions **only within GPR sentences**, using the country-variants lexicon.
5. Aggregate sentence and word counts to produce firm–country–year and firm–year measures.

## Outputs (CSV)
- `sentence_level.csv`  
  Every management-presentation sentence with GPR flags, matched terms, and countries mentioned (countries populated only for GPR-window sentences).
- `gpr_sentences_country.csv`  
  GPR sentence × country pairs.
- `firm_country_year.csv`  
  `I_GPR_ict`, `Frac_GPR_ict`, `Word_GPR_ict`.
- `firm_year.csv`  
  `I_GPR_it`, `Frac_GPR_it`, `Word_GPR_it` (also produces a filtered `firm_year_2015_2017.csv` if the constants are set that way).

## Traceability fields
- `sentence_index` is **0-based** within each transcript.
- `sentence_id` is `{file_id}_S00001`, `{file_id}_S00002`, ...

## How to run in Colab
1. Unzip transcripts into a folder, e.g. `/content/2016/`
2. Put `GPR dictionary.csv` and `country_variants.csv` in `/content/`
3. Open `gpr_text_pipeline_v5.py` and set the top-of-file constants:
   - `XML_DIR` to your transcript folder (e.g. `/content/2016/`)
   - `GPR_DICT_FILE` to `/content/GPR dictionary.csv`
   - `COUNTRY_DICT_FILE` to `/content/country_variants.csv`
   - `OUTPUT_DIR` to where you want the outputs saved
4. Run:

```python
!python /content/gpr_text_pipeline_v5.py
```

## Notes
- This version implements the **±3 sentence window** rule exactly as specified in the RA instructions.
- Sentiment is intentionally not implemented yet.
