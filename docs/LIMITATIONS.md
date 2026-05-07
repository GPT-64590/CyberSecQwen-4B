# Limitations and responsible use

This is a research-stage defensive cybersecurity language model. It is not a production-grade security tool. Read this document before deploying.

## Out-of-scope use

The following uses are explicitly out-of-scope and not supported:

1. **Generating exploit code, weaponized PoC, or attacker tradecraft.** The model is fine-tuned on defensive analyst content with explicit refuse-and-deflect framing for offensive prompts, but no model can be relied on to refuse 100% of the time. Do not use this model in any pipeline that produces or ships executable attack content.

2. **Auto-executing security decisions without human review.** Outputs are advisory. Do not wire the model to:
   - Auto-blocklist IP addresses, accounts, or files
   - Auto-quarantine endpoints
   - Auto-revoke credentials
   - Auto-trigger incident response workflows
   - Make patch deployment decisions
   without a qualified human in the loop reviewing the model's reasoning.

3. **Legal, medical, or other regulated-advice contexts.** The training data is cybersecurity-domain only.

4. **Tasks outside cybersecurity.** General chat, code generation, summarization, translation, and other domains will produce significantly lower-quality output than purpose-built models for those tasks.

5. **Use that violates applicable laws** — including but not limited to unauthorized scanning, reconnaissance against systems you do not own or have explicit permission to test, and use that breaches the U.S. Computer Fraud and Abuse Act, the EU GDPR, or analogous local statutes.

## Known limitations

1. **Domain-specific knowledge limitations.** The model is fine-tuned on cybersecurity domain text and is not a general-purpose assistant.

2. **Time-anchored training data.** The CWE classification training data is from MITRE/NVD records dated 2021. Vulnerability classes that emerged or rose in prevalence after 2021 (recent supply-chain CWE patterns, AI/ML-specific weaknesses, novel exploit classes from 2022-2026) are under-represented in training and will be classified less accurately.

3. **English-only.** All training and evaluation data is in English. Multilingual cybersecurity tasks (Indonesian, Spanish, German, etc.) are unsupported and will degrade.

4. **CTI-RCM gap to Cisco-Instruct.** Foundation-Sec-Instruct-8B remains stronger by ~1 pp on CTI-RCM under our protocol. Production deployments where CWE classification is the primary metric should benchmark both models on their specific input distribution before choosing.

5. **No safety RLHF.** The model is supervised-fine-tuned only. The training data emphasizes defensive-analyst framing but no formal reinforcement-learning safety alignment was applied. Expect the model to occasionally produce outputs that would be more diplomatic with RLHF — for example, vulnerability descriptions that include detail an attacker might find useful even when the framing is defensive.

6. **Chat template note.** The repository ships with a minimal training-aligned chat template matching the format used during SFT (Qwen `<|im_start|>` / `<|im_end|>` user-and-assistant turns, no thinking-mode block). Inference via `tokenizer.apply_chat_template(messages, ...)` produces correctly-formatted prompts; downstream tooling that injects system prompts or thinking-mode toggles outside this template may degrade output quality.

7. **No safety-evaluation results.** This release does not include adversarial robustness, prompt-injection, jailbreak, or red-team evaluation results. If your deployment is safety-critical, you must run those evaluations against your specific input distribution before relying on the model.

## Recommended-use guardrails

When integrating this model into a workflow:

1. **Always have qualified security professionals review model outputs before implementation** for any operational use case (patch prioritization, CVE triage, ticket routing, incident response decisions).

2. **Use this model as an assistive tool rather than a replacement for expert human judgment** — especially for novel vulnerability classes (post-2021), supply-chain attacks, AI/ML security, or any zero-day analysis.

3. **Validate on your own input distribution** before production deployment. Public CTI-Bench performance does not perfectly transfer to internal advisory feeds, vendor-proprietary CWE taxonomies, or non-English content.

4. **Monitor for drift.** As new CVE / CWE patterns emerge, periodically re-evaluate. Consider supplementing the model with retrieval over a current vulnerability knowledge base for time-sensitive queries.

5. **Apply prompt-injection mitigations** when wrapping the model in agentic workflows that accept external content (advisory feeds, scraped pages, third-party reports). Domain-SFT does not confer prompt-injection resistance.

6. **Log model outputs and reviewer corrections** to feed back into your evaluation pipeline; that's the most reliable way to detect model regressions on your specific use cases over time.

## Reporting safety issues

If you discover a use case where this model produces clearly harmful output (e.g., unsolicited exploit code generation, unsolicited attacker tradecraft) please open a GitHub issue with the input that triggered it and the output (redacted as appropriate). We treat these as priority bugs even though this is a research release.
