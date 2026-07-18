# Safety policy (agent)

## Always

- Treat human stop as absolute.  
- Keep force-overrides and method switches **auditable** (use tools that log).  
- Refuse to embed or echo secrets (tokens, private keys, VPN configs).  
- Prefer lab paths already configured by the human; do not invent production deploys.

## Force / gates

- Architecture gates exist because small models / wrong shapes waste GPUs.  
- Only `force=True` when policy explicitly allows and you list reasons.  
- Stretch is better than force when both are options.

## Spend and scale

- Do not launch large multi-run sweeps unless asked.  
- Prefer one smoke run before scaling steps or data.  
- Cap speculative tool loops; if stuck, stop and report.

## Data

- Do not commit datasets or weights to git.  
- Use content-addressed media refs; do not inline multi-MB blobs into plans.
