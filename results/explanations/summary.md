| Model | Alerts | Grounded | Cites a feature SHAP didn't rank | Invented numbers | SHAP reasons used | Latency p50 |
|---|---:|---:|---:|---:|---:|---:|
| `qwen2.5:1.5b` | 40 | 65% | 20% | 12% | 84% | 1.3s |
| `llama3.2:3b` | 40 | 80% | 20% | 0% | 84% | 2.6s |

40 alerts sampled from the test period at the 50/day threshold (15% of them are labelled fraud).
