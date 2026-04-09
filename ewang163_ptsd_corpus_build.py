"""
ewang163_ptsd_corpus_build.py
=============================
Assembles the training corpus from extracted notes:
  - PTSD+ pre-diagnosis notes: used as-is (no leakage risk)
  - PTSD+ fallback (index-admission) notes: masked with Ablation 1
  - Unlabeled notes: used as-is
  - Proxy notes: split out for post-training validation only

Ablation 1 masking replaces PTSD-related strings with [PTSD_MASKED].

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
OUT = '/oscar/data/class/biol1595_2595/students/ewang163'

NOTES_PARQUET  = f'{OUT}/ewang163_ptsd_notes_raw.parquet'
CORPUS_PARQUET = f'{OUT}/ewang163_ptsd_corpus.parquet'
PROXY_PARQUET  = f'{OUT}/ewang163_proxy_notes.parquet'

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

    # ── Step 2: Apply Ablation 1 masking to fallback PTSD+ notes ──────────
    print('\n[2/4] Applying Ablation 1 masking to PTSD+ fallback notes ...')
    fallback_mask = (df['group'] == 'ptsd_pos') & (~df['is_prediagnosis'])
    n_fallback = fallback_mask.sum()

    df['is_masked'] = False
    df.loc[fallback_mask, 'note_text'] = df.loc[fallback_mask, 'note_text'].apply(apply_masking)
    df.loc[fallback_mask, 'is_masked'] = True

    # Count how many fallback notes actually had substitutions
    n_with_subs = 0
    for idx in df[fallback_mask].index:
        if MASK_TOKEN in df.at[idx, 'note_text']:
            n_with_subs += 1

    print(f'  Fallback notes masked: {n_fallback:,}')
    print(f'  Of those, notes containing [PTSD_MASKED]: {n_with_subs:,}')

    # ── Step 3: Assign labels and split ───────────────────────────────────
    print('\n[3/4] Assigning labels and splitting proxy ...')

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
    print('\n[4/4] Saving ...')
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
