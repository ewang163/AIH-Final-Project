"""
ewang163_ptsd_notes_extract.py
==============================
Extracts section-filtered discharge notes for all cohort patients.

discharge_detail.csv in MIMIC-IV note v2.2 only contains 'author' rows,
so we parse sections from the full-text discharge.csv instead.

Included sections (narrative, lower-leakage):
    History of Present Illness, Social History,
    Past Medical History, Brief Hospital Course

Excluded (label-leaking) sections are simply not extracted.

Note selection logic:
    PTSD+ primary (2,492):   pre-diagnosis admission notes only
    PTSD+ fallback (3,219):  index-admission note (section-filtered)
    Proxy:                   index-admission note only
    Unlabeled:               index-admission note only

Inputs:
    ewang163_ptsd_adm_extract.parquet

Outputs:
    ewang163_ptsd_notes_raw.parquet
        columns: subject_id, hadm_id, group, is_prediagnosis, note_text

RUN:
    source ptsd_env/bin/activate
    python ewang163_ptsd_notes_extract.py
"""

import csv
import sys
import re
import numpy as np
import pandas as pd
from collections import defaultdict

csv.field_size_limit(sys.maxsize)

# ── Paths ─────────────────────────────────────────────────────────────────
MIMIC = '/oscar/data/shared/ursa/mimic-iv'
OUT   = '/oscar/data/class/biol1595_2595/students/ewang163'

DISCHARGE_F  = f'{MIMIC}/note/2.2/discharge.csv'
ADM_PARQUET  = f'{OUT}/ewang163_ptsd_adm_extract.parquet'
OUT_PARQUET  = f'{OUT}/ewang163_ptsd_notes_raw.parquet'

# ── Section filtering ─────────────────────────────────────────────────────
# Keep these narrative sections (match case-insensitively against parsed headers)
INCLUDE_SECTIONS = {
    'history of present illness',
    'social history',
    'past medical history',
    'brief hospital course',
}

# Canonical ordering for concatenation
SECTION_ORDER = [
    'history of present illness',
    'social history',
    'past medical history',
    'brief hospital course',
]

# Regex to detect section headers: a line like "History of Present Illness:\n"
# Matches a line starting with a letter, containing words/spaces/punctuation, ending with ":"
SECTION_HEADER_RE = re.compile(
    r'^([A-Z][A-Za-z /&\-]+):[ ]*$',
    re.MULTILINE,
)


def parse_sections(text):
    """
    Parse a discharge note into {section_name_lower: section_text} dict.
    Only returns sections whose lowercased header is in INCLUDE_SECTIONS.
    """
    if not text:
        return {}

    # Find all section header positions
    headers = []
    for m in SECTION_HEADER_RE.finditer(text):
        headers.append((m.start(), m.end(), m.group(1).strip().lower()))

    result = {}
    for i, (start, end, name) in enumerate(headers):
        if name not in INCLUDE_SECTIONS:
            continue
        # Section body runs from end of header line to start of next header
        body_start = end
        body_end = headers[i + 1][0] if i + 1 < len(headers) else len(text)
        body = text[body_start:body_end].strip()
        if body:
            # If multiple sections with the same name, keep first
            if name not in result:
                result[name] = body

    return result


def concatenate_sections(sections_dict):
    """Join section texts in canonical order with header labels."""
    parts = []
    for section_name in SECTION_ORDER:
        if section_name in sections_dict:
            parts.append(sections_dict[section_name])
    return '\n\n'.join(parts)


def main():
    print('=' * 65)
    print('PTSD NLP Project — Discharge Notes Extraction')
    print('=' * 65)

    # ── Step 1: Load admission extract to determine allowed hadm_ids ──────
    print('\n[1/4] Loading admission extract ...')
    adm = pd.read_parquet(ADM_PARQUET)
    print(f'  {len(adm):,} admission rows loaded')

    # Build lookup: hadm_id → (subject_id, group)
    hadm_info = {}
    for _, row in adm.iterrows():
        hadm_info[row['hadm_id']] = (row['subject_id'], row['group'])

    # For PTSD+ patients: separate pre-diagnosis and index hadm_ids
    ptsd = adm[adm['group'] == 'ptsd_pos']

    # Pre-diagnosis hadm_ids: admissions strictly before index
    ptsd_predx_hadms = set(
        ptsd[ptsd['admittime'] < ptsd['index_admittime']]['hadm_id']
    )
    # Index hadm_ids for PTSD+ patients
    ptsd_index_hadms = set(
        ptsd[ptsd['is_index_admission']]['hadm_id']
    )
    # All allowed PTSD+ hadm_ids (we'll decide primary vs fallback in post-processing)
    ptsd_allowed_hadms = ptsd_predx_hadms | ptsd_index_hadms

    # For proxy: index admission only
    proxy_index_hadms = set(
        adm[(adm['group'] == 'proxy') & adm['is_index_admission']]['hadm_id']
    )

    # For unlabeled: index admission only
    unlab_index_hadms = set(
        adm[(adm['group'] == 'unlabeled') & adm['is_index_admission']]['hadm_id']
    )

    all_allowed_hadms = ptsd_allowed_hadms | proxy_index_hadms | unlab_index_hadms
    print(f'  Allowed hadm_ids: {len(all_allowed_hadms):,} total')
    print(f'    PTSD+ pre-dx: {len(ptsd_predx_hadms):,} | '
          f'PTSD+ index: {len(ptsd_index_hadms):,} | '
          f'Proxy index: {len(proxy_index_hadms):,} | '
          f'Unlabeled index: {len(unlab_index_hadms):,}')

    # ── Step 2: Stream discharge.csv and extract sections ─────────────────
    print('\n[2/4] Streaming discharge.csv (3.3 GB, may take several minutes) ...')
    # Collect: list of (subject_id, hadm_id, group, is_predx, note_text)
    collected = []
    total_notes = 0
    matched_notes = 0
    empty_sections = 0

    with open(DISCHARGE_F, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            total_notes += 1
            if total_notes % 50000 == 0:
                print(f'    ... processed {total_notes:,} notes, '
                      f'matched {matched_notes:,} so far')

            hadm_id = int(row['hadm_id'])
            if hadm_id not in all_allowed_hadms:
                continue

            subject_id, group = hadm_info[hadm_id]

            # Parse and filter sections
            sections = parse_sections(row['text'])
            note_text = concatenate_sections(sections)

            if not note_text.strip():
                empty_sections += 1
                continue

            matched_notes += 1

            # Determine if this is a pre-diagnosis note (PTSD+ only)
            is_predx = (group == 'ptsd_pos' and hadm_id in ptsd_predx_hadms)

            collected.append({
                'subject_id': subject_id,
                'hadm_id': hadm_id,
                'group': group,
                'is_prediagnosis': is_predx,
                'note_text': note_text,
            })

    print(f'  Total notes scanned: {total_notes:,}')
    print(f'  Matched notes with sections: {matched_notes:,}')
    print(f'  Skipped (no included sections): {empty_sections:,}')

    # ── Step 3: Post-process PTSD+ notes ──────────────────────────────────
    # For each PTSD+ patient: if they have ANY pre-diagnosis notes, drop
    # their index-admission note (primary path). Otherwise keep the index
    # note (fallback path).
    print('\n[3/4] Post-processing PTSD+ notes (primary vs fallback) ...')
    df = pd.DataFrame(collected)

    # Find PTSD+ patients who have at least one pre-diagnosis note
    ptsd_notes = df[df['group'] == 'ptsd_pos']
    patients_with_predx = set(
        ptsd_notes[ptsd_notes['is_prediagnosis']]['subject_id'].unique()
    )
    patients_index_only = set(
        ptsd_notes['subject_id'].unique()
    ) - patients_with_predx

    # Drop index-admission rows for patients who have pre-dx notes
    drop_mask = (
        (df['group'] == 'ptsd_pos') &
        (~df['is_prediagnosis']) &
        (df['subject_id'].isin(patients_with_predx))
    )
    n_dropped = drop_mask.sum()
    df = df[~drop_mask].copy()

    print(f'  PTSD+ patients with pre-diagnosis notes (primary): '
          f'{len(patients_with_predx):,}')
    print(f'  PTSD+ patients using index-admission notes (fallback): '
          f'{len(patients_index_only):,}')
    print(f'  Dropped {n_dropped:,} redundant PTSD+ index-admission rows')

    # ── Step 4: Save and report ───────────────────────────────────────────
    print('\n[4/4] Saving output ...')
    df.to_parquet(OUT_PARQUET, index=False)
    print(f'  Saved {len(df):,} rows → {OUT_PARQUET}')

    # ── Summary statistics ────────────────────────────────────────────────
    print('\n' + '=' * 65)
    print('SUMMARY')
    print('=' * 65)

    # Patients with notes per group
    for grp in ['ptsd_pos', 'proxy', 'unlabeled']:
        grp_df = df[df['group'] == grp]
        n_patients = grp_df['subject_id'].nunique()
        n_notes = len(grp_df)
        print(f'  {grp:12s}: {n_patients:,} patients, {n_notes:,} note rows')

    # PTSD+ breakdown
    ptsd_df = df[df['group'] == 'ptsd_pos']
    n_predx_patients = ptsd_df[ptsd_df['is_prediagnosis']]['subject_id'].nunique()
    n_fallback_patients = ptsd_df[~ptsd_df['is_prediagnosis']]['subject_id'].nunique()
    print(f'\n  PTSD+ pre-diagnosis patients: {n_predx_patients:,} (expect ~2,492)')
    print(f'  PTSD+ fallback patients:      {n_fallback_patients:,} (expect ~3,219)')

    # Note length distribution by group
    df['note_len'] = df['note_text'].str.len()
    print('\n  Note length (characters) by group:')
    for grp in ['ptsd_pos', 'proxy', 'unlabeled']:
        grp_lens = df[df['group'] == grp]['note_len']
        if len(grp_lens) == 0:
            print(f'    {grp:12s}: no notes')
            continue
        print(f'    {grp:12s}: '
              f'median {grp_lens.median():,.0f} | '
              f'mean {grp_lens.mean():,.0f} | '
              f'Q1 {grp_lens.quantile(0.25):,.0f} | '
              f'Q3 {grp_lens.quantile(0.75):,.0f} | '
              f'min {grp_lens.min():,} | '
              f'max {grp_lens.max():,}')

    print(f'\n  Columns: {list(df.columns)}')
    print('\nDone.')


if __name__ == '__main__':
    main()
