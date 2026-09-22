"""Compare lexical and frozen-transformer features under the same split."""

import argparse
import json
from pathlib import Path
from time import perf_counter

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.pipeline import make_pipeline
from threadpoolctl import threadpool_limits

from intentlab.common import environment, save_bundle, sha256, write_json
from intentlab.data import load_data
from intentlab.encoder import MODEL_ID, REVISION, Encoder, download_encoder
from intentlab.evaluation import (
    add_typo,
    fit_temperature,
    routing_metrics,
    scale_probabilities,
    scores,
    select_threshold,
)


def run(data_dir="data", output="artifacts", reports="reports"):
    data_dir, output, reports = Path(data_dir), Path(output), Path(reports)
    reports.mkdir(parents=True, exist_ok=True)
    train, test, splits, removed = load_data(data_dir)
    tr, ca, va = [splits[k] for k in ["train", "calibration", "validation"]]
    texts, labels = train.text.to_numpy(), train.category.to_numpy()
    encoder_path = download_encoder(data_dir / "encoder")
    encoder = Encoder(encoder_path)
    print("Encoding development text...", flush=True)
    embeddings = encoder.cached(texts.tolist(), data_dir / "embeddings")
    candidates = {
        "tfidf_logistic": make_pipeline(
            TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True),
            LogisticRegression(C=4, max_iter=1000, random_state=42),
        )
    }
    for c in [4, 16, 64]:
        candidates[f"minilm_logistic_c{c}"] = LogisticRegression(
            C=c, max_iter=1000, random_state=42
        )
    validation, temperatures = {}, {}
    for name, model in candidates.items():
        values = texts if name.startswith("tfidf") else embeddings
        model.fit(values[tr], labels[tr])
        temperature = fit_temperature(labels[ca], model.predict_proba(values[ca]), model.classes_)
        temperatures[name] = temperature
        validation[name] = scores(
            labels[va],
            scale_probabilities(model.predict_proba(values[va]), temperature),
            model.classes_,
        )
        print(name, validation[name], flush=True)
    selected = max(validation, key=lambda k: validation[k]["macro_f1"])
    model, temperature = candidates[selected], temperatures[selected]
    values = texts if selected.startswith("tfidf") else embeddings
    pv = scale_probabilities(model.predict_proba(values[va]), temperature)
    threshold, curve = select_threshold(labels[va], pv, model.classes_)
    print("Encoding held-out test text...", flush=True)
    et = encoder.cached(test.text.tolist(), data_dir / "embeddings")
    test_results, predictions = {}, {}
    for name, candidate in candidates.items():
        values_test = test.text.to_numpy() if name.startswith("tfidf") else et
        p = scale_probabilities(candidate.predict_proba(values_test), temperatures[name])
        predictions[name] = p
        test_results[name] = scores(test.category, p, candidate.classes_)
    p = predictions[selected]
    predicted = model.classes_[p.argmax(axis=1)]
    per_class = classification_report(test.category, predicted, output_dict=True)
    write_json(reports / "per_class_metrics.json", per_class)
    matrix = confusion_matrix(test.category, predicted, labels=model.classes_)
    np.fill_diagonal(matrix, 0)
    pairs = sorted(
        [
            (int(matrix[i, j]), str(a), str(b))
            for i, a in enumerate(model.classes_)
            for j, b in enumerate(model.classes_)
            if matrix[i, j]
        ],
        reverse=True,
    )
    pd.DataFrame(pairs, columns=["count", "actual", "predicted"]).to_csv(
        reports / "confusions.csv", index=False
    )

    def probabilities(sentences):
        values = sentences if selected.startswith("tfidf") else encoder.encode(sentences)
        return scale_probabilities(model.predict_proba(values), temperature)

    typo_text = test.text.map(add_typo).tolist()
    # Cache deterministic stress embeddings separately from the untouched test.
    typo_values = (
        typo_text
        if selected.startswith("tfidf")
        else encoder.cached(typo_text, data_dir / "embeddings")
    )
    typo_p = scale_probabilities(model.predict_proba(typo_values), temperature)
    probes = json.loads(Path("tests/fixtures/review_cases.json").read_text())
    probe_p = probabilities([case["text"] for case in probes])
    probe_rows = [
        {
            **case,
            "predicted": str(model.classes_[row.argmax()]),
            "confidence": float(row.max()),
            "reviewed": bool(row.max() < threshold),
        }
        for case, row in zip(probes, probe_p)
    ]
    write_json(reports / "review_probe_results.json", probe_rows)
    durations = []
    probabilities([test.text.iloc[0]])  # warm-up excluded
    for text in test.text.iloc[:60]:
        start = perf_counter()
        probabilities([text])
        durations.append(1000 * (perf_counter() - start))
    encoder_hashes = {
        name: sha256(encoder_path / name) for name in ["tokenizer.json", "model.onnx"]
    }
    bundle = {
        "model": model,
        "temperature": temperature,
        "threshold": threshold,
        "name": selected,
        "encoder_revision": REVISION,
        "encoder_hashes": encoder_hashes,
    }
    manifest = save_bundle(output, bundle)
    report = {
        "environment": environment(),
        "encoder": {
            "id": MODEL_ID,
            "revision": REVISION,
            "frozen": True,
            "backend": "ONNX Runtime CPU",
            "max_tokens": 256,
            "sha256": encoder_hashes,
        },
        "selected_model": selected,
        "selection_rule": "maximum validation macro-F1 among four fixed candidates",
        "split_counts": {**{k: len(v) for k, v in splits.items()}, "test": len(test)},
        "removed_training_duplicates_or_overlaps": removed,
        "validation": validation,
        "test": test_results,
        "temperatures": temperatures,
        "routing_validation": routing_metrics(labels[va], pv, model.classes_, threshold),
        "routing_test": routing_metrics(test.category, p, model.classes_, threshold),
        "routing_policy": "max coverage with validation Wilson 95% lower bound >= .95 and coverage >= .10",
        "typo_test": scores(test.category, typo_p, model.classes_),
        "typo_routing": routing_metrics(test.category, typo_p, model.classes_, threshold),
        "review_probes": {
            "n": len(probes),
            "reviewed": sum(r["reviewed"] for r in probe_rows),
            "description": "hand-written out-of-scope and ambiguous probes, not a representative OOD benchmark",
        },
        "latency_ms": {
            "p50": float(np.median(durations)),
            "p95": float(np.quantile(durations, 0.95)),
            "samples": len(durations),
            "batch_size": 1,
            "includes": "tokenization, encoder, classifier, calibration; excludes HTTP",
        },
        "artifact": manifest,
    }
    write_json(reports / "metrics.json", report)
    pd.DataFrame(curve).to_csv(reports / "validation_routing_curve.csv", index=False)
    split_rows = train[["text", "category"]].copy()
    split_rows["split"] = ""
    for name, idx in splits.items():
        split_rows.loc[idx, "split"] = name
    split_rows.to_csv(output / "split_manifest.csv", index=False)
    fig, ax = plt.subplots(figsize=(7, 4))
    visible = [r for r in curve if r["accepted"] >= 30]
    ax.plot(
        [r["coverage"] for r in visible],
        [r["accuracy"] for r in visible],
        label="Validation",
    )
    point = report["routing_test"]
    if point["accuracy"] is not None:
        ax.scatter(
            point["coverage"],
            point["accuracy"],
            color="#b45309",
            label="Frozen policy on test",
            zorder=3,
        )
    ax.set(
        xlabel="Fraction automatically routed",
        ylabel="Accuracy on routed requests",
        title="Accuracy versus automation coverage",
        ylim=(0.7, 1.01),
    )
    ax.legend()
    fig.tight_layout()
    fig.savefig(reports / "routing.svg")
    plt.close(fig)
    print(
        {
            "selected": selected,
            "test": test_results[selected],
            "routing": report["routing_test"],
        },
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--output", default="artifacts")
    parser.add_argument("--reports", default="reports")
    args = parser.parse_args()
    with threadpool_limits(limits=2):
        run(args.data_dir, args.output, args.reports)
