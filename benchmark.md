# Does Zet help? A plain-language benchmark

**Yes, on this six-choice test.** Zet sent 41 of Laya's 45 wrong answers to review while still
accepting 227 of 350 answers automatically. It also sent 82 correct answers to review and let four
wrong answers through. This is a measured tradeoff, not a claim that Zet fixes Laya's answers.

## What we tested

We selected a simpler task: put a voice-assistant request into one of six distinct topics—alarms,
cooking, email, news, transport, or weather. We used the English and Swedish requests in [MASSIVE 1.1](https://github.com/alexa/massive),
which already have answers assigned by people. These were **not private emails**. The model was
Laya's [multilingual ONNX checkpoint](https://huggingface.co/soyelmismo/laya-multilingual-onnx),
which Zet pins to a specific [revision](zet/backends/weights.py).

There were 582 eligible requests per language. Zet used about 70% of their known answers to set
its sure/unsure rule. The remaining 175 English and 175 Swedish requests were held back for this
test. Zet made its decisions without using those held-back answers; we used them afterward to
count what was right and wrong. Both rows below use the **same Laya predictions**.

## The result: 350 held-back requests

| If we use... | Answers used automatically | Wrong automatic answers | Answers sent to review |
|---|---:|---:|---:|
| Laya alone | 350 | 45 | 0 |
| Laya with Zet | 227 | 4 | 123 |

Among the 123 answers Zet sent to review, **41 were Laya mistakes** and **82 were already right**.
So Zet flagged **41 of Laya's 45 mistakes (91%)**. Of the answers it accepted automatically, **4
of 227 (1.8%) were wrong**. A person would have to check 123 answers to find the 41 flagged
mistakes, assuming the person reviews them correctly.

That is why Zet helped **on this test**: it kept most wrong answers from being used automatically
while leaving about two thirds of requests automated. Zet did not rewrite Laya's answers or know
which individual new answer was correct. It learned a rule from the earlier labeled requests and
applied that rule to the held-back ones.

## What this does not show

- **No person performed the reviews.** “Sent to review” means Zet flagged an answer. It does not
  mean someone actually corrected it, so we did not measure final accuracy after review.
- **Review has a cost.** Of the 123 reviews, 82 would be for correct answers. Whether that tradeoff
  is worthwhile depends on the cost of a mistake and the cost of a review.
- **It does not work equally well on every task.** With 18 possible topics, Laya made 228 mistakes
  on 602 held-back requests, and Zet sent **all 602** to review. That avoids automatic mistakes by
  doing no automation. On a small synthetic support-email test, Zet also accepted nothing
  automatically at the same 5% error budget. The six topics above were chosen as an easier task,
  not randomly drawn from all 18.
- **This is evidence for these requests, not a promise for your emails.** New requests must
  resemble the labeled examples. The cautious 95% upper bound on the held-back automatic error
  rate was slightly above 5% for English (5.2%) and Swedish (5.7%) separately, even though the
  observed rates were lower. We need representative labeled emails and a real review trial to
  measure the benefit in an email workflow.

## Sources and how to repeat it

- [MASSIVE dataset and its label format](https://github.com/alexa/massive) and the
  [MASSIVE paper](https://arxiv.org/abs/2204.08582). The benchmark uses the `dev` partition of
  MASSIVE 1.1, not its separate `test` partition.
- [Laya project](https://github.com/NandhaKishorM/laya) and the
  [ONNX checkpoint](https://huggingface.co/soyelmismo/laya-multilingual-onnx). Zet pins revision
  `0966c4fa58da6878b39e7e14cb5e93313b82d828` in [weights.py](zet/backends/weights.py).
- [Benchmark script](benchmarks/massive/run.py), [raw results](benchmarks/massive/results.json),
  and [detailed results by language](benchmarks/massive/results.md). The recorded run used Zet
  `0.1.0.dev0`, seed `0`, and a 5% error budget on 2026-09-30.

From a local clone, after installing Zet with `python -m pip install -e .`, run:

```sh
python benchmarks/massive/run.py --languages en-US sv-SE --per-language 1000 --seed 0 --model multilingual --error-budget 0.05 --out outputs/massive-recheck
```

The script downloads the public dataset and model weights on first use. It writes `results.json`
and `results.md` in the output folder. The six-topic subset contains only 582 eligible `dev`
requests per language, so `--per-language 1000` uses all 582; the 18-topic test uses 1000 per
language. The data and model downloads stay outside the repository, and `outputs/` is ignored by
Git.
