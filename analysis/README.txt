Existing-pilot exploratory analysis.

Headline primary labels are the research assistant review of every complete greedy final-request decode. primary-review.tsv columns are record ID, explicit user evidence, implicit speaker-content evidence, explicit Assistant assessment evidence, explicit Assistant intention evidence. Indices are one-based representative evidence, not exhaustive annotations. A dash means no qualifying evidence under the conservative rule. User attribution requires an actor-linked mental state or goal, not a bare request heading. Fully specified helper offers count as Assistant wording; incomplete role prefixes alone do not.

The automated annotator uses the same intended categories but differs on borderline language; its original labels and all primary presence changes are preserved. Secondary anchor, control, fidelity and sample labels remain automated/provisional. No independent human adjudication was performed.

Run analyze_jpp.py and jpp_sensitivity.py against saved candidates. Run analyze_pilot.py to annotate 351 saved O-lens outputs, then summarize_analysis.py to merge annotations with the documented primary review. GPT-4.1 initial-pass aggregates were discarded after QA found category conflation. The final automatic pass uses GPT-5.4; local/analysis-v2 stores checkpoints and model metadata. API credentials are never published.

No new subject-model generation or activation capture was performed. All counts concern this fixed pilot, and correlated conditions are not treated as independent population draws.
