[README.md](https://github.com/user-attachments/files/34221706/README.md)
# Purpose as Argument — measurement pipeline

An auditable, LLM-assisted pipeline for measuring how firms justify and commit to their stated purpose in shareholder letters. A stated purpose is read as an argument addressed to stakeholders: a claim about the end the firm pursues, supported (or not) by reasoning and evidence, and tied (or not) to commitments. Every purpose argument is measured on four dimensions, kept separate rather than summed into one score:

| Dimension | What it captures | Main measures |
|---|---|---|
| 1. Prominence | How much of the letter is about purpose | purpose claims ÷ all claims in the letter; share of words |
| 2. Content | What end, and for whom | the claimed end; beneficiaries, named or inferred; purpose domain (taxonomy to be built inductively, mapped to Rajan, Ramella and Zingales 2023) |
| 3. Substantiation | What reasoning and evidence support the claim | grounds, warrant, backing, qualifier, rebuttal (Toulmin 1958), each explicit / inferred / absent |
| 4. Commitment specificity | What the firm says it has done or will do | aspiration / future pledge / reported past action; for pledges, whether an action, a quantitative target and a target date are specified (0–3) |

This repository accompanies the dissertation proposal *Purpose as Argument: An Auditable LLM Infrastructure for Measuring How Firms Justify and Commit to Their Stated Purpose* (HEC Paris). It is a work in progress; see **Status** below.

## Contents

| File | What it is |
|---|---|
| `toulmindecoding_revised_2026-10-31.py` | The pipeline: Step 1 split and categorise, Step 2 decode purpose units, Step 3 score the four dimensions. The header of the file contains the full "how to run" instructions and design notes. |
| `Allianz 2014.csv` … `Allianz 2019.csv` | Example outputs of the decoding step for six Allianz shareholder letters (development run, not results; see **Example files**). |
| `figures/pipeline_outputs.png` | Map of the files the pipeline writes and the keys that link them (see **Outputs and keys**). |
| `README.md` | This file. |

The shareholder letters themselves are not included; they are public documents available from the companies' investor-relations pages.

## How the pipeline reads a letter

1. **Split and categorise.** The letter is split into argument units (a claim sentence plus the support, exception and commitment clauses attached to it) and each unit receives one of nine categories: `purpose`, `strategy`, `performance`, `product`, `governance`, `risk`, `shareholder_value`, `sustainability_culture`, `other`. A unit is `purpose` only if its claim sentence states why the firm exists or what it is for, beyond financial return, and names a beneficiary or outcome beyond shareholders. Non-purpose units give the denominator for prominence.
2. **Decode.** Each purpose unit is coded for its components: the claim and its end, beneficiaries, grounds, warrant, backing, qualifier, rebuttal, and commitments with their attributes. Every explicit component is stored as a verbatim span with character offsets into the frozen canonical text of the letter; inferred components are stored separately and never counted as evidence.
3. **Score.** The four dimensions are computed per purpose unit and per letter (firm-year), and assembled into a firm-year panel.

## How to run

Requirements: Python 3.10+, `pandas`, `openai` (or `anthropic` / `google-genai`), `PyMuPDF`, `python-docx`; `pytesseract` and `pillow` only for scanned PDFs. The script was written for Google Colab with Google Drive mounted, but runs anywhere once the paths in section 1 of the file are changed.

Steps 1 and 2 call a language model and **require a valid API key** for the provider set in `PROVIDER` (OpenAI by default). Store the key in Colab Secrets or as an environment variable (`OPENAI_API_KEY`, `ANTHROPIC_API_KEY` or `GEMINI_API_KEY`); never write it into the script.

1. Put the shareholder letters in `LETTERS_DIR`, one `.pdf` or `.docx` per letter. The file stem becomes `letter_id`; name files `<firm>_<year>` (for example `Allianz_2019.pdf`) so that `firm_id` and `year` can be parsed, or supply a `letter_index.csv` with columns `letter_id, firm_id, year`.
2. `run_step1()` builds the canonical text of each letter and writes one JSON and one CSV of categorised units per letter. Letters already processed are skipped, so the step can be resumed.
3. `run_step2()` decodes the purpose units of each letter and writes one JSON and one CSV of components per letter. Set `DECODE_ALL_UNITS = True` to decode non-purpose units as well (needed only for the within-letter comparison; multiplies the API cost).
4. `run_step3()` reads the JSON files and writes `claim_level.csv`, `firm_year.csv` and `firm_panel.csv`. No API call.

Running the file as a script performs the three steps in order. Every record carries `run_id`, provider, model, temperature, `prompt_hash` and `codebook_version`, so any figure can be traced to the prompt and model that produced it.

## Outputs and keys

![Files written by the pipeline and the keys that join them](figures/pipeline_outputs.png)

| Output | Level | Key |
|---|---|---|
| `LettersCanonical/<letter_id>.txt` | letter | `letter_id`; every `char_start` / `char_end` refers to this text |
| `Arguments/<provider>/JSON\|CSV/<letter_id>` | unit within letter | `letter_id`, `argument_id` |
| `Decode/<provider>/JSON\|CSV/<letter_id>` | component within unit | `letter_id`, `argument_id`, `role` |
| `Scores/<provider>/claim_level.csv` | purpose unit within firm-year | `letter_id`, `argument_id`, plus `firm_id`, `year` |
| `Scores/<provider>/firm_year.csv` | letter (firm-year) | `letter_id`, `firm_id`, `year` |
| `Scores/<provider>/firm_panel.csv` | firm by year | `firm_id`, `year` |

Human reference coding for validation joins the component files on `letter_id`, `argument_id` and `role`; external firm-level data (for example Compustat or follow-through indicators) joins the panel on `firm_id` and `year`.

## Example files

`AstraZeneca 2016.csv` to `AstraZeneca 2020.csv` are example outputs of the categorization step for 5 AstraZeneca shareholder letters, one file per letter, with the component's verbatim text and its status. `decode_AstraZeneca 2016.csv` to `decode_AstraZeneca 2020.csv` are example outputs of the decoding step, one file per letter and one row per argument component, with the component's verbatim text and its status.
They were generated in a development run with `gpt-4.1-2025-04-14` at temperature 0 using the September 2025 version of the decoding prompt; the current script writes the schema described above, so column names differ in places. The files illustrate the output format only: they have not been validated against human coding and are not results.

## Status

- Implemented: Step 1 (split and categorise) and Step 2 (decode) prompts and loops; Step 3 scoring; verbatim anchoring with offsets; provenance in every record.
- To do: the inductive purpose-domain taxonomy and its mapping to Rajan, Ramella and Zingales (2023) (`classify_domains_stub` in the script); named-entity counts in grounds; the human-coded reference set and reliability tables; a pinned open-weight model for the reportable run (the gpt-4.1 snapshot used in development retires on 14 January 2027).

