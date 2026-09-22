# Model card

**Owner:** Nikhil Shrivastava. **Version:** 1.0 portfolio experiment.

**Use:** Demonstrate fine-grained banking intent classification and human-review
routing. **Not established:** autonomous customer support, out-of-domain safety,
production ROI, or compliance with any institution's governance requirements.

**Architecture:** Pinned MiniLM ONNX encoder, attention-mask mean pooling,
L2 normalization, logistic regression C=16, temperature scaling and a
validation-selected confidence threshold. No transformer fine-tuning or GenAI
answer generation. The model classifies text; it never performs account actions.

**Output:** Suggested categories, confidence and route/review action. When review
is required, the final intent is null; candidates remain suggestions. The demo
queue is local session state, not a durable enterprise case-management system.

| Risk | Implemented control | Remaining limitation |
|---|---|---|
| Duplicate leakage | Normalize exact text; remove training/test overlaps | Near-duplicate paraphrases and encoder pretraining exposure unknown |
| Overconfident errors | Separate temperature calibration and threshold validation | Unsupported requests can still score highly |
| Noisy input | Deterministic typo stress test | Large measured performance drop remains |
| Incorrect automatic route | Review below confidence threshold; visible candidates | This policy is not a general OOD detector |
| Wrong model version | Hashes for classifier, tokenizer and encoder; readiness check | Trusted artifact origin is still required |
| Text exposure | No raw message logging by the API; session-only demo queue | No PII detector or enterprise retention policy |

**Monitoring proposal:** Track category mix, confidence, review coverage, measured
latency, and reviewer corrections. Join delayed labels to predictions by request
ID to evaluate accepted-request accuracy and per-intent recall. This is a plan;
the repository does not claim to operate a live monitoring system.

**Promotion and retirement:** Review current test/stress evidence with a human
owner before changing a model version. Preserve a rollback bundle. Stop automatic
routing if verified accuracy or input scope deteriorates; retrain only with an
appropriate new development/test protocol. These are project controls, not an
independent AI readiness approval.
