# Periodic Rewrite Quality Loss

Enable with `algorithm.reskill_grpo.rewrite_quality.enable=True`. The Qwen3.5-2B four-GPU profile uses interval 5, old success threshold 0.75, loss coefficient 0.1, and unlikelihood beta 1.0. Existing skill SFT remains 0.1.

Before actor optimization, the original group execution supplies the old success estimate. Eligible parsed rewrites are executed with the same current actor on independent copies of the exact original ALFWorld task files. Original and rewritten executions use paired per-replica, per-turn request seeds and identical sampling settings. Temporary evaluation environments are closed afterward; their trajectories do not enter GRPO.

Quality is positive only for a strict success-rate improvement and nonzero normalized content edit distance. Positive examples use token-averaged NLL; negative examples use stable token-averaged unlikelihood. Both use the actual rewrite generation prompt and generated response tokens. Invalid rewrites are removed from ordinary positive skill SFT. Auxiliary gradients accumulate with native GRPO before its single optimizer step, using a global example mean across distributed shards; padded replicas have zero weight.

No eligible groups means no replay, payload, or additional gradient. Validation never rewrites or applies this loss. Per-step audit JSONL files preserve prompts, task identities, old/new outcomes, content distance, quality labels, and masking decisions. W&B records quality counts, replay latency, success rates, auxiliary losses, and phase GPU peaks.

Validation: `python -m unittest discover -s tests -p 'test_*.py' -v`; native fixed-task replay: `python launchers/local_train/check_fixed_game_replay.py`.

The active four-GPU profile disables thinking in both the tokenizer chat template and ALFWorld action instructions; actions request only `<action>...</action>`. Skill generation/rewrite prompts also prohibit reasoning output. The actor computation token budget is 40,960 (gradient accumulation preserves the configured logical PPO minibatch). The current entropy-inclusive packed full-model backward preflight passed with finite gradients and a sampled peak of 100.78 GiB. Native vLLM action-only inference produced 8–15 tokens per request in the four-prompt check, without thinking tags; action-format mistakes remain native environment feedback.

In no-thinking mode, the native action parser also disables its historical requirement for `<think>` tags. The same manager-level setting applies to training, validation, and paired quality re-execution. The full native replay integration test checks that action-only outputs receive valid-action feedback.

The expanded real backward preflight includes the unchanged entropy coefficient 0.001; at 40,960 tokens all gradients were finite and the sampled full-model peak was 100.78 GiB. Earlier 49,152-token production was stopped by the 130 GiB watchdog at step 3, so that cap is no longer used.
