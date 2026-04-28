"""
ewang163_ptsd_corpus_build.py
=============================
Assembles the training corpus from extracted notes:
  - PTSD+ notes (ALL — pre-diagnosis AND fallback): masked with Ablation 1
  - Unlabeled notes: used as-is
  - Proxy notes: split out for post-training validation only

Fix 1 applied: masking is applied to ALL PTSD+ notes, not just fallback.
Pre-diagnosis notes can still contain "h/o PTSD from MVA 2012" in HPI/PMH
carried forward from outside records — this was a leakage path.

An audit step counts the pre-dx leakage hit rate before masking to quantify
the scope of the issue.

Outputs:
    ewang163_ptsd_corpus.parquet   — training rows (PTSD+ label=1, unlabeled label=0)
    ewang163_proxy_notes.parquet   — proxy group for post-training validation

RUN:
    source ptsd_env/bin/activate
    python ewang163_ptsd_corpus_build.py
"""

import re
import pandas as pd

# ── Paths ─────────────────────────────────────────────────────────────────
STUDENT_DIR  = '/oscar/data/class/biol1595_2595/students/ewang163'
DATA_NOTES   = f'{STUDENT_DIR}/data/notes'

NOTES_PARQUET  = f'{DATA_NOTES}/ewang163_ptsd_notes_raw.parquet'
CORPUS_PARQUET = f'{DATA_NOTES}/ewang163_ptsd_corpus.parquet'
PROXY_PARQUET  = f'{DATA_NOTES}/ewang163_proxy_notes.parquet'

# ── Ablation 1 masking patterns (case-insensitive) ────────────────────────
MASK_PATTERNS = [
    r'post-traumatic',
    r'post\s+traumatic',
    r'posttraumatic',
    r'trauma-related\s+stress',
    r'ptsd',
    r'f43\.1',
    r'309\.81',
]
# Combine into a single regex, longest patterns first to avoid partial matches
MASK_RE = re.compile('|'.join(MASK_PATTERNS), re.IGNORECASE)
MASK_TOKEN = '[PTSD_MASKED]'


def apply_masking(text):
    """Replace all PTSD-related strings with [PTSD_MASKED]."""
    return MASK_RE.sub(MASK_TOKEN, text)


def main():
    print('=' * 65)
    print('PTSD NLP Project — Corpus Build')
    print('=' * 65)

    # ── Step 1: Load notes ────────────────────────────────────────────────
    print('\n[1/4] Loading notes ...')
    df = pd.read_parquet(NOTES_PARQUET)
    print(f'  {len(df):,} note rows loaded')
    for grp in ['ptsd_pos', 'proxy', 'unlabeled']:
        n = (df['group'] == grp).sum()
        print(f'    {grp}: {n:,} rows')

    # ── Step 2a: Audit pre-diagnosis leakage (Fix 1) ───────────────────────
    print('\n[2a/5] Auditing pre-diagnosis leakage before masking ...')
    predx_mask = (df['group'] == 'ptsd_pos') & (df['is_prediagnosis'] == True)
    n_predx = predx_mask.sum()
    if n_predx > 0:
        predx_hit = df.loc[predx_mask, 'note_text'].str.contains(MASK_RE).sum()
        predx_hit_rate = predx_hit / n_predx
        print(f'  Pre-diagnosis PTSD+ notes: {n_predx:,}')
        print(f'  Of those, notes containing PTSD-related strings: {predx_hit:,} '
              f'({predx_hit_rate:.1%})')
        print(f'  *** This is the pre-Fix-1 leakage rate in the primary training set ***')
    else:
        predx_hit_rate = 0.0
        print(f'  No pre-diagnosis notes found (all PTSD+ are fallback)')

    # ── Step 2b: Apply masking to ALL PTSD+ notes (Fix 1) ────────────────
    print('\n[2b/5] Applying Ablation 1 masking to ALL PTSD+ notes (Fix 1) ...')
    all_pos_mask = (df['group'] == 'ptsd_pos')
    n_pos_total = all_pos_mask.sum()

    df['is_masked'] = False
    df.loc[all_pos_mask, 'note_text'] = df.loc[all_pos_mask, 'note_text'].apply(apply_masking)
    df.loc[all_pos_mask, 'is_masked'] = True

    n_with_subs = 0
    for idx in df[all_pos_mask].index:
        if MASK_TOKEN in df.at[idx, 'note_text']:
            n_with_subs += 1

    print(f'  Total PTSD+ notes masked: {n_pos_total:,} '
          f'(pre-dx: {n_predx:,}, fallback: {n_pos_total - n_predx:,})')
    print(f'  Of those, notes containing [PTSD_MASKED]: {n_with_subs:,} '
          f'({n_with_subs / n_pos_total * 100:.1f}%)')

    # ── Step 3: Assign labels and split ───────────────────────────────────
    print('\n[3/5] Assigning labels and splitting proxy ...')

    # Proxy → separate file, no label
    proxy_df = df[df['group'] == 'proxy'].copy()
    proxy_df = proxy_df[['subject_id', 'hadm_id', 'group', 'note_text']].copy()

    # Training corpus: PTSD+ (label=1) + unlabeled (label=0)
    train_df = df[df['group'].isin(['ptsd_pos', 'unlabeled'])].copy()
    train_df['ptsd_label'] = (train_df['group'] == 'ptsd_pos').astype(int)

    # Select output columns
    train_df = train_df[['subject_id', 'hadm_id', 'group', 'ptsd_label',
                         'note_text', 'is_masked']].copy()

    print(f'  Training corpus: {len(train_df):,} rows')
    print(f'    Positive (ptsd_label=1): {(train_df["ptsd_label"] == 1).sum():,}')
    print(f'    Unlabeled (ptsd_label=0): {(train_df["ptsd_label"] == 0).sum():,}')
    print(f'  Proxy validation set: {len(proxy_df):,} rows')

    # ── Step 4: Save ──────────────────────────────────────────────────────
    print('\n[4/5] Saving ...')
    train_df.to_parquet(CORPUS_PARQUET, index=False)
    print(f'  Training corpus → {CORPUS_PARQUET}')

    proxy_df.to_parquet(PROXY_PARQUET, index=False)
    print(f'  Proxy notes     → {PROXY_PARQUET}')

    # ── Summary ───────────────────────────────────────────────────────────
    print('\n' + '=' * 65)
    print('CORPUS SUMMARY')
    print('=' * 65)

    print(f'  Total training rows:   {len(train_df):,}')
    print(f'  Positive rows (PTSD+): {(train_df["ptsd_label"] == 1).sum():,}')
    print(f'  Unlabeled rows:        {(train_df["ptsd_label"] == 0).sum():,}')

    # Note length stats (estimate tokens as chars / 4)
    train_df['char_len'] = train_df['note_text'].str.len()
    train_df['est_tokens'] = train_df['char_len'] / 4

    print(f'\n  Note length (characters):')
    print(f'    mean {train_df["char_len"].mean():,.0f} | '
          f'median {train_df["char_len"].median():,.0f}')

    print(f'  Estimated tokens (chars/4):')
    print(f'    mean {train_df["est_tokens"].mean():,.0f} | '
          f'median {train_df["est_tokens"].median():,.0f}')

    n_masked = train_df['is_masked'].sum()
    print(f'\n  Masked notes: {n_masked:,} / {len(train_df):,} '
          f'({n_masked / len(train_df) * 100:.1f}%)')

    print('\nDone.')


if __name__ == '__main__':
    main()
