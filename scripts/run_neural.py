"""Train and evaluate the BiLSTM or DistilBERT (GPU recommended; see notebooks/run_on_colab.ipynb).

Same protocol as run_classical.py: train on original or stripped text, select on validation,
test on original and stripped. Finished runs are skipped.

  python scripts/run_neural.py --model distilbert --setting in-domain --seeds 0 1 2
  python scripts/run_neural.py --model distilbert --setting in-domain --seeds 0 --explain 300
  python scripts/run_neural.py --model bilstm --setting cross-source --seeds 0
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from fnd import neural, results  # noqa: E402
from fnd.data import RESULTS_DIR, iter_splits, load_corpus  # noqa: E402
from fnd.preprocess import text_view  # noqa: E402
from fnd.stats import metrics  # noqa: E402

TEST_CONDS = {'original': ['original', 'stripped'], 'stripped': ['stripped']}
DISPLAY = {'distilbert': 'DistilBERT', 'bilstm': 'BiLSTM'}
VIEW = {'distilbert': 'raw', 'bilstm': 'light'}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', choices=list(DISPLAY), required=True)
    ap.add_argument('--setting', choices=['in-domain', 'cross-source'], default='in-domain')
    ap.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2, 3, 4])
    ap.add_argument('--train-conds', nargs='+', default=['original', 'stripped'])
    ap.add_argument('--max-len', type=int, default=None, help='tokens (default 256 DistilBERT, 300 BiLSTM)')
    ap.add_argument('--epochs', type=int, default=None, help='default 2 DistilBERT, up to 10 BiLSTM')
    ap.add_argument('--limit', type=int, default=None, help='subsample each split (for smoke tests)')
    ap.add_argument('--explain', type=int, default=0, help='Integrated Gradients on N test articles (DistilBERT)')
    args = ap.parse_args()
    if args.limit:  # keep smoke-test output away from the real results
        results.PRED_DIR = RESULTS_DIR / 'smoke' / 'preds'
        results.RUNS_DIR = RESULTS_DIR / 'smoke'

    df, stats = load_corpus()
    print('Corpus:', stats, flush=True)
    views = {c: text_view(df, c, VIEW[args.model]) for c in ['original', 'stripped']}
    y, row_ids = df['label'].to_numpy(), df['row_id'].to_numpy()
    name = DISPLAY[args.model]
    kwargs = {k: v for k, v in [('max_len', args.max_len), ('epochs', args.epochs)] if v is not None}

    for split_name, seed, split in iter_splits(df, args.setting, args.seeds):
        if args.limit:
            rng = np.random.default_rng(seed)
            split = {k: rng.choice(v, min(args.limit, len(v)), replace=False) for k, v in split.items()}
        for train_cond in args.train_conds:
            if all(results.done(split_name, train_cond, tc, name, seed) for tc in TEST_CONDS[train_cond]) \
                    and not args.explain:
                continue
            pick = lambda c, part: [views[c][i] for i in split[part]]
            tests = {tc: pick(tc, 'test') for tc in TEST_CONDS[train_cond]}
            y_tr, y_val, y_te = y[split['train']], y[split['val']], y[split['test']]
            print(f'== {name} {split_name} seed={seed} train={train_cond} '
                  f"(train {len(y_tr)}, val {len(y_val)}, test {len(y_te)})", flush=True)

            if args.model == 'bilstm':
                probs, seconds, info = neural.train_bilstm(pick(train_cond, 'train'), y_tr,
                                                           pick(train_cond, 'val'), y_val, tests, seed, **kwargs)
            else:
                probs, seconds, info, model, tok = neural.train_distilbert(
                    pick(train_cond, 'train'), y_tr, pick(train_cond, 'val'), y_val, tests, seed, **kwargs)
                if args.explain:
                    rng = np.random.default_rng(seed)
                    sample = rng.choice(len(y_te), min(args.explain, len(y_te)), replace=False)
                    for tc in TEST_CONDS[train_cond]:
                        words, arts = neural.explain_distilbert(model, tok, [tests[tc][i] for i in sample],
                                                                y_te[sample], max_len=args.max_len or 256)
                        out = results.PRED_DIR.parent / 'explain' /f'{split_name}__{train_cond}__{tc}__seed{seed}'
                        out.mkdir(parents=True, exist_ok=True)
                        words.to_csv(out / 'words.csv', index=False)
                        arts.to_csv(out / 'articles.csv', index=False)
                        print(f"  IG {train_cond}->{tc}: artifact share of attribution "
                              f"{arts['artifact_share'].mean():.3f}", flush=True)

            for tc, p in probs.items():
                pred = (p >= 0.5).astype(int)
                results.save_predictions(split_name, train_cond, tc, name, seed, row_ids[split['test']], y_te, pred, p)
                m = metrics(y_te, pred, p)
                results.log_run(family='neural', split=split_name, seed=seed, model=name, train_cond=train_cond,
                                test_cond=tc, params=kwargs, searched=False, train_seconds=seconds,
                                limit=args.limit, **info, **m)
                print(f"  {train_cond}->{tc} F1={m['f1']:.4f} acc={m['accuracy']:.4f} ({seconds:.0f}s)", flush=True)


if __name__ == '__main__':
    main()
