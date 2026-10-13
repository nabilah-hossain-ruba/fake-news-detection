"""BiLSTM (PyTorch) and fine-tuned DistilBERT, both selected on the validation split.

DistilBERT reads the raw article (its own tokenizer handles casing and punctuation); the
BiLSTM reads lowercase word tokens with stopwords kept. Integrated Gradients attributions
show which tokens DistilBERT relies on.
"""
import copy
import random
import shutil
import time
from collections import Counter

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------- BiLSTM

class BiLSTM(nn.Module):
    def __init__(self, vocab_size, embed_dim=128, hidden=64):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden, batch_first=True, bidirectional=True)
        self.head = nn.Sequential(nn.Dropout(0.5), nn.Linear(2 * hidden, 64), nn.ReLU(),
                                  nn.Dropout(0.4), nn.Linear(64, 1))

    def forward(self, ids, lengths):
        packed = pack_padded_sequence(self.embedding(ids), lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, (h, _) = self.lstm(packed)
        return self.head(torch.cat([h[-2], h[-1]], dim=1)).squeeze(1)


def _encode(texts, vocab, max_len):
    ids = np.zeros((len(texts), max_len), dtype=np.int64)
    lengths = np.ones(len(texts), dtype=np.int64)
    for i, t in enumerate(texts):
        seq = [vocab.get(w, 1) for w in t.split()[:max_len]] or [1]
        ids[i, :len(seq)], lengths[i] = seq, len(seq)
    return torch.from_numpy(ids), torch.from_numpy(lengths)


def _batches(n, size, shuffle, rng=None):
    order = rng.permutation(n) if shuffle else np.arange(n)
    for start in range(0, n, size):
        yield order[start:start + size]


@torch.no_grad()
def _bilstm_probs(model, ids, lengths, batch=512):
    model.eval()
    out = [torch.sigmoid(model(ids[b].to(DEVICE), lengths[b])).cpu() for b in _batches(len(ids), batch, False)]
    return torch.cat(out).numpy()


def train_bilstm(train_texts, y_tr, val_texts, y_val, test_sets, seed,
                 vocab_size=20000, max_len=300, epochs=10, patience=2, batch=128):
    """Returns ({test_name: probs}, train_seconds, info)."""
    set_seed(seed)
    counts = Counter(w for t in train_texts for w in t.split()[:max_len])
    vocab = {w: i + 2 for i, (w, _) in enumerate(counts.most_common(vocab_size - 2))}  # 0 pad, 1 OOV
    X_tr, L_tr = _encode(train_texts, vocab, max_len)
    X_val, L_val = _encode(val_texts, vocab, max_len)
    y_tr_t = torch.tensor(y_tr, dtype=torch.float32)

    model = BiLSTM(vocab_size).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    loss_fn = nn.BCEWithLogitsLoss()
    rng = np.random.default_rng(seed)
    best, best_loss, bad, history = None, np.inf, 0, []

    start = time.perf_counter()
    for epoch in range(epochs):
        model.train()
        for b in _batches(len(X_tr), batch, True, rng):
            opt.zero_grad()
            loss = loss_fn(model(X_tr[b].to(DEVICE), L_tr[b]), y_tr_t[b].to(DEVICE))
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        p_val = np.clip(_bilstm_probs(model, X_val, L_val), 1e-7, 1 - 1e-7)
        val_loss = float(-np.mean(y_val * np.log(p_val) + (1 - y_val) * np.log(1 - p_val)))
        history.append({'epoch': epoch + 1, 'val_loss': val_loss, 'val_f1': f1_score(y_val, p_val >= 0.5)})
        print(f"  BiLSTM epoch {epoch + 1}: val_loss={val_loss:.4f} val_f1={history[-1]['val_f1']:.4f}", flush=True)
        if val_loss < best_loss:
            best, best_loss, bad = copy.deepcopy(model.state_dict()), val_loss, 0
        else:
            bad += 1
            if bad >= patience:
                break
    seconds = time.perf_counter() - start

    model.load_state_dict(best)
    probs = {name: _bilstm_probs(model, *_encode(texts, vocab, max_len)) for name, texts in test_sets.items()}
    best_epoch = min(history, key=lambda h: h['val_loss'])
    return probs, seconds, {'epochs_run': len(history), 'best_epoch': best_epoch['epoch'],
                            'val_f1': best_epoch['val_f1']}


# ---------------------------------------------------------------- DistilBERT

MODEL_NAME = 'distilbert-base-uncased'


def train_distilbert(train_texts, y_tr, val_texts, y_val, test_sets, seed,
                     max_len=256, epochs=2, lr=2e-5, batch=16, out_dir='distilbert_run'):
    """Fine-tune, keep the epoch with the best validation F1. Returns (probs, seconds, info, model, tokenizer)."""
    from datasets import Dataset
    from transformers import (AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding,
                              Trainer, TrainingArguments)

    set_seed(seed)
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)

    def dataset(texts, labels=None):
        d = {'text': list(texts)}
        if labels is not None:
            d['label'] = [int(v) for v in labels]
        return Dataset.from_dict(d).map(lambda b: tok(b['text'], truncation=True, max_length=max_len),
                                        batched=True, remove_columns=['text'])

    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)
    args = TrainingArguments(
        output_dir=out_dir, num_train_epochs=epochs, learning_rate=lr, weight_decay=0.01,
        warmup_steps=int(0.06 * epochs * -(-len(y_tr) // batch)),  # 6% warmup; int works on transformers 4 and 5
        per_device_train_batch_size=batch, per_device_eval_batch_size=64,
        eval_strategy='epoch', save_strategy='epoch', save_total_limit=1,
        load_best_model_at_end=True, metric_for_best_model='f1',
        fp16=torch.cuda.is_available(), logging_steps=200, report_to='none', seed=seed)
    trainer = Trainer(model=model, args=args, train_dataset=dataset(train_texts, y_tr),
                      eval_dataset=dataset(val_texts, y_val), data_collator=DataCollatorWithPadding(tok),
                      compute_metrics=lambda p: {'f1': f1_score(p.label_ids, p.predictions.argmax(-1))})

    start = time.perf_counter()
    trainer.train()
    seconds = time.perf_counter() - start

    probs = {}
    for name, texts in test_sets.items():
        logits = trainer.predict(dataset(texts)).predictions
        probs[name] = torch.softmax(torch.tensor(logits, dtype=torch.float32), dim=-1)[:, 1].numpy()
    evals = [h for h in trainer.state.log_history if 'eval_f1' in h]
    info = {'val_f1': max(h['eval_f1'] for h in evals),
            'best_epoch': max(evals, key=lambda h: h['eval_f1'])['epoch']}
    shutil.rmtree(out_dir, ignore_errors=True)
    return probs, seconds, info, trainer.model, tok


# ---------------------------------------------------------------- Integrated Gradients

# Words that almost only occur as source artifacts (see artifacts.py). Deliberately narrow:
# ambiguous words such as "image" or "times" are left out, so the artifact share is a lower bound.
ARTIFACT_WORDS = {'reuters', 'getty', 'featured', 'pic', 'breitbart', 'nyt', '21wire', 'http', 'https', 'www', '@'}


def explain_distilbert(model, tok, texts, labels, max_len=256, n_steps=32, batch=8):
    """Integrated Gradients toward the Fake logit for each text.

    Returns (per-word attribution table, per-article summary with the share of |attribution|
    that falls on artifact words).
    """
    from captum.attr import LayerIntegratedGradients

    model.eval().to(DEVICE)
    forward = lambda ids, mask: model(input_ids=ids, attention_mask=mask).logits[:, 1]
    lig = LayerIntegratedGradients(forward, model.distilbert.embeddings)
    words, articles = [], []
    for start in range(0, len(texts), batch):
        enc = tok(list(texts[start:start + batch]), truncation=True, max_length=max_len,
                  padding=True, return_tensors='pt').to(DEVICE)
        ids, mask = enc['input_ids'], enc['attention_mask']
        base = torch.where((ids == tok.cls_token_id) | (ids == tok.sep_token_id), ids,
                           torch.full_like(ids, tok.pad_token_id))
        attr = lig.attribute(ids, baselines=base, additional_forward_args=(mask,),
                             n_steps=n_steps, internal_batch_size=batch).sum(-1)
        attr = (attr / attr.norm(dim=1, keepdim=True)).detach().cpu().numpy()
        for row, (seq, a) in enumerate(zip(ids.cpu().numpy(), attr)):
            merged = []  # join word pieces back into words
            for t, v in zip(tok.convert_ids_to_tokens(seq), a):
                if t in (tok.cls_token, tok.sep_token, tok.pad_token):
                    continue
                if t.startswith('##') and merged:
                    merged[-1][0] += t[2:]
                    merged[-1][1] += v
                else:
                    merged.append([t, v])
            total = sum(abs(v) for _, v in merged) or 1.0
            art = sum(abs(v) for w, v in merged if w in ARTIFACT_WORDS)
            articles.append({'label': int(labels[start + row]), 'artifact_share': art / total})
            words += [{'word': w, 'attribution': v, 'label': int(labels[start + row])} for w, v in merged]

    words = pd.DataFrame(words)
    table = (words.groupby('word')['attribution'].agg(['sum', 'mean', 'count'])
             .query('count >= 5').sort_values('sum'))
    table['is_artifact'] = table.index.isin(ARTIFACT_WORDS)
    return table.reset_index(), pd.DataFrame(articles)
