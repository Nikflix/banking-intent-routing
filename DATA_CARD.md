# Data and pretrained model card

**Dataset:** [BANKING77 by PolyAI](https://github.com/PolyAI-LDN/task-specific-datasets).
13,083 English banking customer-service queries across 77 intent categories;
10,003 official training examples and 3,080 official test examples.

**Citation:** Casanueva, I., Temcinas, T., Gerz, D., Henderson, M., and Vulić, I.
(2020). Efficient Intent Detection with Dual Sentence Encoders.
https://arxiv.org/abs/2003.04807. Dataset license: CC BY 4.0.

The official test split is preserved. Lowercase/whitespace-normalized exact
training duplicates, conflicting labels and test-overlapping training texts are
filtered before development splitting. Eleven training rows are removed in the
reference run. The remaining 9,992 split into 6,994 training, 1,499 calibration and
1,499 validation records, stratified by intent with seed 42. Near-duplicate
paraphrases may remain; exact matching is not semantic deduplication.

SHA-256:
- train.csv: `b06e26ac675513959a63135f11b94ea7786ed02da65db93a5650d8838cbc664b`
- test.csv: `d12d6e3bc4c3103966ae786dc435913c0c563dfa328f5a3646d0e62cfeeb474d`

**Encoder:** [sentence-transformers/all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2),
Apache 2.0, revision `1110a243fdf4706b3f48f1d95db1a4f5529b4d41`.
Downloaded ONNX and tokenizer hashes are recorded in the evaluation and checked
when serving. The encoder is frozen; this project does not retrain its weights.
Its pretraining provenance cannot rule out exposure to public benchmark text.
The measured result is a downstream benchmark, not a contamination-free claim.

The training corpus contains in-scope single-intent queries. It does not establish
performance on multilingual text, long conversations, arbitrary banking products,
multi-intent requests or unrelated messages. Text is truncated to 256 tokens.
Manually authored review probes and deterministic transpositions are stress
tests only, not new representative datasets.
