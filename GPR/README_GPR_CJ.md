# README  
## Project: CEO Perceptions of Geopolitical Risk (GPR)  
### Textual Analysis of Earnings Call Management Presentations

---

## 1. Objective
The goal of this project is to construct text-based measures of CEOs’ perceived geopolitical risk (GPR) from earnings call transcripts. The analysis focuses exclusively on **management presentations** (CEO, CFO, COO, top executives), excluding Q&A sections and analyst language. The outputs include sentence-level, firm–country–year, and firm–year GPR measures for the period 2016–2019.

---

## 2. Input Files
The pipeline uses the following inputs:

- Earnings call transcripts (`*_T.xml`)
- GPR keyword dictionary (updated version provided by Prof. Ayyagari)
- Country list with name variants (e.g., “U.S.”, “United States”, “America”)

No regressions are run at this stage.

---

## 3. Text Pre-Processing
For each transcript:

1. Parse the XML file and extract the **management presentation text only**.
2. Clean text (remove markup, normalize spacing).
3. Split the presentation into sentences.
4. Compute transcript-level totals:
   - Total number of sentences  
   - Total number of words  

These totals are used to normalize GPR measures.

---

## 4. Identifying GPR Discussion Blocks (±3 Sentences)
1. **Anchor sentences** are identified as sentences containing at least one GPR keyword from the dictionary.
2. For each anchor sentence, a **±3 sentence context window** is constructed.
3. All sentences within this 7-sentence window are labeled as **GPR discussion-block sentences**.
4. Overlapping windows are deduplicated so that each sentence appears only once.

Two indicators are retained:

- `is_gpr_anchor`: sentence contains a GPR keyword  
- `is_gpr_block`: sentence falls within a ±3 GPR discussion block  

---

## 5. Identifying Countries in GPR Blocks
Country mentions are identified using a country-lexicon approach:

- Country names and variants are searched **only within GPR discussion-block sentences**
- A sentence may reference multiple countries
- Countries do **not** need to appear in the same sentence as the GPR keyword; appearing anywhere in the ±3 sentence block qualifies

This produces a **sentence–country** dataset.

---

## 6. Constructing Firm–Country–Year Measures
For each firm *i*, country *c*, and year *t*, the following measures are constructed:

- **I(GPR_ict)**  
  Indicator equal to 1 if any GPR discussion block references country *c*.

- **Frac(GPR_ict)**  
(# GPR sentences referencing country c) / (total sentences in presentation)

scss

- **Word(GPR_ict)**  

(# words in GPR sentences referencing country c) / (total words in presentation)

yaml

---

## 7. Constructing Firm–Year Measures (Non-Country-Specific)
Using all GPR discussion-block sentences (regardless of country mentions):

- **I(GPR_it)**  
Indicator equal to 1 if the firm mentions any GPR in year *t*.

- **Frac(GPR_it)**  
(# GPR sentences) / (total sentences in presentation)

scss

- **Word(GPR_it)**  
(# words in GPR sentences) / (total words in presentation)

yaml

---

## 8. Outputs
The pipeline produces four datasets:

| File | Description |
|------|------------|
| `sentence_level.csv` | Every sentence with GPR anchor/block flags, matched terms, and country mentions |
| `gpr_sentences_country.csv` | GPR sentences × country pairs |
| `firm_country_year.csv` | Country-specific GPR measures |
| `firm_year.csv` | Overall firm-year GPR measures |

---

## 9. Notes
- Sentiment analysis is **not implemented yet** and will be added in a later stage.
- The same pipeline is applied uniformly across years (2016–2019).
- The code is designed to be directly reusable for the larger transcript universe or restricted to the smaller sample firms.

