# License Check

Before any text goes to a model, the license settings decide whether it may and where. The settings apply to every paper in a run. The verdict is guidance, not legal advice.

| Verdict | Model allowed |
|---|---|
| `any_llm` | local, private cloud, or any API |
| `closed_ai` | local, private cloud, or an API that neither trains on nor keeps the text |
| `local_only` | this machine only |
| `manual` | none; code by hand |

An open license such as CC BY gives `any_llm`; a subscription paper whose terms are silent on AI gives `local_only`. When the route is not allowed, nothing is sent and those fields are `refused`.
