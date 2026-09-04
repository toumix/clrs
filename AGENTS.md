# AGENTS.md

## What

This is DeepMind's [CLRS Algorithmic Reasoning Benchmark](README.md), forked to `toumix/clrs` as
the beachhead for [discopy#678](https://github.com/discopy/discopy/issues/678): running CLRS tasks
as string diagrams, evaluated by a DisCoPy functor. See [PR #1](https://github.com/toumix/clrs/pull/1)
(`goi/`) for the shape of that work.

## Running experiments

PR #1's whole scorecard — three tasks, three seeds each, `pytest`/`python -m goi.run_<task>` —
ran on **CPU inside the sandbox, no Modal, no GPU** ("per the standing ruling"). Prefer that path:
it needs nothing external, nothing to authenticate, and nothing to go wrong. Reach for Modal only
when a run genuinely needs GPU or wall-clock the sandbox can't give it, and read the rest of this
file first — the credential story around it is not settled, and getting it wrong is how agents
end up inventing infra that `rel-int/infra`'s own team then has to clean up after.

### Why this file is blunt about that

Agents working across this org have already hallucinated missing checkpoints and invented bucket
deployments once — it's on record in `toumix/memory`'s `OTHERS/0x0f0f0f.md` as a standing
complaint from the person who owns `rel-int/infra` (an org Cloudflare Super Administrator, with
curator write access to the actual data lake). His `CLAUDE.md` there is unambiguous about why:
never mint a credential yourself, never invent a bucket or a path, never commit a dataset or a
checkpoint to git. This file exists so an agent here has somewhere to read that *before* running
into it, rather than after.

### Compute: Modal is reachable, with one gotcha

- `api.modal.com` is on the sandbox's default egress allowlist, but the `modal` SDK needs the
  `api-proxy-support` extra to actually use it — `pip install 'modal[api-proxy-support]'` (or add
  it to `requirements/`). Without it, the SDK fails with a misleading
  `Could not connect to the Modal server`; the real cause, two frames down, is
  `ModuleNotFoundError: python_socks`.
- `MODAL_TOKEN_ID`, `MODAL_TOKEN_SECRET` and `MODAL_ENVIRONMENT=notebooks` need to already be in
  the session's own environment — there is no `.env` and no in-repo credential, and none should
  ever be added. If they're missing, ask USER to set them on the Claude Code cloud environment
  rather than inventing or hardcoding a token. `toumix/ARC-AGI` already runs this way (its own
  launcher reads `MODAL_TOKEN_ID`/`MODAL_TOKEN_SECRET` from the environment) — mirror that rather
  than reinventing it.
- `modal.com`, `*.modal.run` and `*.modal.host` are blocked at the egress proxy; `modal run`
  doesn't need them from inside the sandbox.

### Storage: unresolved — don't freelance it

There is currently **no machine credential for R2 at all**. A Modal job has no Cloudflare Access
identity, and there is no sanctioned way today for one to read or write the data lake. See
[rel-int/infra#4](https://github.com/rel-int/infra/issues/4) (open) and `toumix/memory`'s
`WORK/infra/4.md` for the live state — USER has already ruled out an MCP server and `setup.py` as
the answer there (2026-09-02) and said a token in the session's own environment is what he does
today, but nothing has turned that into a bucket, a script or a written rule yet. In order:

1. **A run that fits in the PR needs nothing external.** PR #1 scored its whole table without
   touching R2 — prefer that path while it holds.
2. **If a run genuinely needs to persist something outside the PR — a checkpoint, a dataset —
   ask USER which of infra#4's options he wants before writing any code that touches R2.** There
   is no sanctioned recipe to copy yet; resolving that design question is USER's call, not an
   agent's to make by building around it.
3. **If USER has explicitly authorised the interim fast path**, it is: the shared Modal token
   already in the environment, plus one dashboard-minted key on `rel-int-scratch` (never
   `rel-int-data`) in a Modal Secret, accepting its 30-day life — and say so, plainly, in the PR
   body, rather than leaving it implicit.
4. **If reading or appending to the actual data lake is needed** (not scratch), the mechanism is
   `rel-int/infra`'s `data.rel-int.ai/v1/creds`, a `RELINT_TOKEN` bearer exchanged for short-lived
   (≤24h) scoped credentials — read that repo's `CLAUDE.md` in full before using it. `toumix`'s
   own row in `packages/data-lake/users.yaml` is `data:read`, `data:append`, `hub:run` — no
   `data:write`, so nothing done through it can delete or overwrite what's already in the lake; it
   can only read, and append new objects under `raw/`.

### Never

- Never mint a long-lived R2 key, widen a Cloudflare Access policy, or touch
  `rel-int/infra`'s `stacks/credentials.ts` admin-profile flow yourself — that repo's `CLAUDE.md`
  is explicit that this is never a service's, or a session's, call to make.
- Never commit a dataset, a checkpoint, or generated experiment output to this repo's git
  history — it belongs in R2 (`raw/`, `weights/`) or nowhere.
- Never invent a bucket or dataset path — `rel-int/infra`'s `packages/data-lake/datasets.toml` is
  the index of what exists and where.
