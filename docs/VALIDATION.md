# Evaluation and routing policy

## Experimental question

Does a compact frozen transformer improve fine-grained intent routing over a
lexical baseline, and how much coverage remains when uncertain requests go to a
person?

TF-IDF uses word unigrams/bigrams, min_df=2, sublinear term frequency and logistic
regression with C=4. MiniLM features feed logistic regression with C in {4,16,64}.
Vocabulary and classifiers fit on training only. A scalar temperature minimizes
calibration-partition log loss for each candidate. Maximum validation macro-F1
selects C=16; neither encoder weights nor classifier weights are fitted on test.

The review threshold is the lowest value on a 0.01 grid whose validation Wilson
95% lower confidence bound for accepted-request accuracy is at least 0.95, with
at least 10% coverage. If no threshold qualifies, automatic routing is disabled.
This is a selection heuristic, not a guarantee: checking many thresholds makes
the selected validation bound optimistic. The independent test result is the
evaluation of the frozen policy.

## Held-out findings

- Overall: 93.08% accuracy; 0.9308 macro-F1 on 3,080 test examples.
- Lexical baseline: 87.60% accuracy; improvement is 5.49 percentage points.
- Threshold: 0.68; 2,792 requests routed automatically, 288 reviewed.
- Accuracy among routed requests: 96.92%; Wilson 95% interval
  96.21%–97.50%. This interval assumes independent test requests; near-duplicates
  can weaken that assumption.
- Test log loss: 0.2646; 10-bin expected calibration error: 0.0129.
- Warm single-request CPU inference on this environment: median
  3.91 ms; p95 5.78 ms over 60 sequential
  requests. Includes tokenization, encoder, classifier and calibration; excludes
  network overhead and cold startup. This is not a load/concurrency benchmark.

Error analysis in `reports/confusions.csv` highlights overlapping categories:
pending transfers versus transfer timing, unrecognized card payments versus
compromised cards, and delivery estimates versus card arrival. Per-intent
precision/recall/F1 are in `reports/per_class_metrics.json`.

## Stress tests, without retuning

Swapping two characters in the first sufficiently long word of every test query
reduces overall accuracy to 80.03%. At the same threshold, 72.27% are routed and
93.49% of routed requests are correct. The effect is substantial: a clean-text
benchmark overstates robustness. This deterministic perturbation is not an
estimate of actual customer typo frequency.

The model sends 20 of 24 manually written unsupported/ambiguous probes to review.
Four still route automatically, including an unrelated dog-health question and a
mortgage application query outside the label set. The complete inputs and outputs
are preserved in `reports/review_probe_results.json`. The tests were diagnostic;
they were not used to adjust the threshold or cherry-pick the reported policy.

## Business interpretation

90.65% automated coverage is a routing simulation on an in-scope benchmark. It
does not establish a 90.65% reduction in handling time, staffing or cost. A pilot
would need real traffic, reviewer capacity, category-specific error costs, and
measured time for both accepted and escalated requests. Incorrect automatic
routing is still possible and can be costly.

## Next experiment

Collect a separate development set of genuinely unsupported, ambiguous and noisy
requests. Compare an explicit out-of-domain detector and character-aware features
with the current system, then freeze the revised policy and evaluate it on a new
test set. Preserve the current baseline; do not tune against the reported test
errors and present the same test scores as fresh evidence.
