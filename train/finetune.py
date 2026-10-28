"""Fine-tuning do BERTimbau nas avaliações da Olist e publicação no Hugging Face Hub.

Rodar no Google Colab (GPU T4 gratuita, ~15 min por tarefa) -> notebooks/finetune_colab.ipynb
ou localmente:

    pip install -e ".[train]"
    python -m copiloto.data
    python train/finetune.py --task sentimento --push anapbatista/bertimbau-olist-sentimento

Usa a MESMA divisão temporal do baseline (copiloto/model.py), então os números são comparáveis.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from copiloto.data import load  # noqa: E402
from copiloto.model import temporal_split  # noqa: E402

TASKS = {
    "sentimento": {"column": "sentimento", "labels": ["negativo", "neutro", "positivo"]},
    "atraso": {"column": "atrasado", "labels": [0, 1]},
}


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=TASKS, default="sentimento")
    ap.add_argument("--model", default="neuralmind/bert-base-portuguese-cased")
    ap.add_argument("--epochs", type=float, default=2)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--max-len", type=int, default=128)
    ap.add_argument("--max-train", type=int, default=None, help="limita exemplos (teste rápido)")
    ap.add_argument("--max-steps", type=int, default=-1)
    ap.add_argument("--out", default=str(ROOT / "models" / "bertimbau"))
    ap.add_argument("--push", default=None, help="repo no Hub, ex.: usuario/bertimbau-olist-sentimento")
    return ap.parse_args()


def model_card(args, metrics: dict, n_train: int) -> str:
    rows = "\n".join(f"| {k} | {v} |" for k, v in metrics.items())
    return f"""---
language: pt
license: cc-by-nc-sa-4.0
base_model: {args.model}
tags: [text-classification, portuguese, reviews, marketplace, olist]
datasets: [olist]
pipeline_tag: text-classification
---

# BERTimbau · avaliações de marketplace ({args.task})

Classificador de **{args.task}** em avaliações de clientes em português, ajustado a partir do
[{args.model}](https://huggingface.co/{args.model}) com {n_train} avaliações reais da Olist.
Faz parte do [Copiloto do Parceiro](https://github.com/anapbatista/copiloto-parceiro).

## Avaliação (divisão temporal: treino até fev/2018, teste abr-ago/2018)

| Métrica | Valor |
|---|---|
{rows}

## Uso

```python
from transformers import pipeline
clf = pipeline("text-classification", model="{args.push or 'MODEL_ID'}")
clf("O produto chegou quebrado e ninguém responde.")
```

## Limitações

- Domínio de e-commerce (2016-2018); pode errar em gírias novas ou outros domínios.
- "neutro" (nota 3) é a classe mais difícil: avaliações mistas.
- Dados: Brazilian E-Commerce Public Dataset by Olist (CC BY-NC-SA 4.0), uso não comercial.
"""


def main():
    args = parse_args()
    import torch
    from datasets import Dataset
    from sklearn.metrics import accuracy_score, average_precision_score, f1_score, roc_auc_score
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        DataCollatorWithPadding,
        Trainer,
        TrainingArguments,
    )

    spec = TASKS[args.task]
    labels = spec["labels"]
    label2id = {str(lab): i for i, lab in enumerate(labels)}
    df = load()
    tr, va, te = temporal_split(df)
    if args.max_train:
        tr, va, te = tr.sample(min(args.max_train, len(tr)), random_state=0), va.head(args.max_train), te.head(args.max_train)

    tok = AutoTokenizer.from_pretrained(args.model)

    def to_ds(part):
        ds = Dataset.from_dict({"text": part.texto.tolist(), "label": [label2id[str(v)] for v in part[spec["column"]]]})
        return ds.map(lambda b: tok(b["text"], truncation=True, max_length=args.max_len), batched=True)

    ds_tr, ds_va, ds_te = to_ds(tr), to_ds(va), to_ds(te)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model, num_labels=len(labels), id2label={i: str(lab) for i, lab in enumerate(labels)}, label2id=label2id,
        ignore_mismatched_sizes=True)

    # Classes desbalanceadas (neutro ~9%, atraso ~10%): perda ponderada pelo inverso da frequência.
    counts = np.bincount(ds_tr["label"], minlength=len(labels))
    weights = torch.tensor(len(ds_tr) / (len(labels) * np.maximum(counts, 1)), dtype=torch.float)

    class WeightedTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            y = inputs.pop("labels")
            out = model(**inputs)
            loss = torch.nn.functional.cross_entropy(out.logits, y, weight=weights.to(out.logits.device))
            return (loss, out) if return_outputs else loss

    def compute_metrics(p):
        logits, y = p
        pred = logits.argmax(-1)
        m = {"macro_f1": f1_score(y, pred, average="macro"), "acuracia": accuracy_score(y, pred)}
        if len(labels) == 2:
            prob = torch.softmax(torch.tensor(logits), -1)[:, 1].numpy()
            m |= {"roc_auc": roc_auc_score(y, prob) if len(set(y)) > 1 else float("nan"),
                  "pr_auc": average_precision_score(y, prob), "f1": f1_score(y, pred)}
        return m

    total_steps = args.max_steps if args.max_steps > 0 else int(np.ceil(len(ds_tr) / args.batch) * args.epochs)
    targs = TrainingArguments(
        output_dir=args.out, num_train_epochs=args.epochs, learning_rate=args.lr, per_device_train_batch_size=args.batch,
        per_device_eval_batch_size=args.batch * 2, eval_strategy="epoch", save_strategy="epoch", load_best_model_at_end=True,
        metric_for_best_model="macro_f1", save_total_limit=1, warmup_steps=max(1, int(0.06 * total_steps)), weight_decay=0.01,
        fp16=torch.cuda.is_available(), max_steps=args.max_steps, report_to=[], logging_steps=50, seed=42)
    trainer = WeightedTrainer(model=model, args=targs, train_dataset=ds_tr, eval_dataset=ds_va,
                              data_collator=DataCollatorWithPadding(tok), compute_metrics=compute_metrics)
    trainer.train()

    test = trainer.evaluate(ds_te, metric_key_prefix="teste")
    metrics = {k.replace("teste_", ""): round(v, 4) for k, v in test.items()
               if k.startswith("teste_") and not k.endswith(("runtime", "per_second", "loss", "epoch"))}
    print(json.dumps({"task": args.task, "n_teste": len(ds_te), **metrics}, indent=2, ensure_ascii=False))

    trainer.save_model(args.out)
    tok.save_pretrained(args.out)
    Path(args.out, "README.md").write_text(model_card(args, metrics, len(ds_tr)), encoding="utf-8")
    Path(args.out, "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    if args.push:
        trainer.model.push_to_hub(args.push)
        tok.push_to_hub(args.push)
        from huggingface_hub import upload_file
        upload_file(path_or_fileobj=str(Path(args.out, "README.md")), path_in_repo="README.md", repo_id=args.push)
        print(f"publicado: https://huggingface.co/{args.push}")


if __name__ == "__main__":
    main()
