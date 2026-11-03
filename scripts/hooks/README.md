# Pre-push guard

A tracked git hook that stops a push if anything in it should not leave the team. It reads every commit being pushed (`<remote sha>..<local sha>` for each ref; for a new branch, the commits no ref of that remote has yet), including commit messages, plus the working-tree `evidence/` folder.

## Enable

Once per clone. It is not enabled globally and the repo does not do it for you:

```
git config core.hooksPath scripts/hooks
```

The hook needs `python3` (standard library only) and git. If `python3` cannot be found, git blocks the push. To switch it off in your clone: `git config --unset core.hooksPath`.

## What it blocks

It prints the rule and the file (and a short blob id, and a line number for text). **It never prints the matched text.**

| Rule | Detail |
|---|---|
| Kit secret values | Loaded at run time from the private kit folder (default `~/firstmate-homes/team13-kit`, or `TEAM13_KIT_DIR`, or `--kit-dir`): everything in `tokens/`, `a2a_token.txt`, `builder_admin_password.txt`, and the values of `kit-state.json` and `kit.json` (except a short list of public names, `PUBLIC_KIT_KEYS` in the script). Compared in memory only. The evidence ledger's private key is compared too. If the kit folder is missing the check is skipped with a warning; `PREPUSH_REQUIRE_KIT=1` makes that fatal. |
| AWS access keys | `AKIA...` and `ASIA...` |
| Private key blocks | `-----BEGIN ... PRIVATE KEY-----` |
| JWTs | `eyJ....eyJ....sig` |
| `Bearer` tokens | `Bearer ` followed by 20 or more token characters |
| IPv4 addresses | Except the explicit allow-list `ALLOWED_IPV4` in the script: loopback block, `0.0.0.0`, `255.255.255.0/255`, `10.0.0.x`, the three RFC 5737 documentation ranges, `169.254.169.254`, `10.255.255.255` and `192.168.1.20` (both used in the dashboard's code and tests). One file is waived for this rule (`PATH_WAIVERS`: the GSMA text, where 3GPP clause numbers look like addresses). |
| 12-digit account ids | Not touching longer digit runs, hex strings or decimals. Example ids are allowed (`ALLOWED_ACCOUNT_IDS`). |
| `amazoncognito`, `X-Amz-Signature`, `ec2-...amazonaws.com` | Hosted login addresses, presigned-URL signatures, EC2 public names |
| Organiser and secret files by name | `checkpoint_kit.py`, `*.whl`, `ran-rl-starter*`, `twin-host*`, `hackathon_docs/`, `tokens/`, `kit.json`, `kit-state.json`, `a2a_token*`, `builder_admin_password*`, `.env` and `.env.*` (not `.env.example`), `host.env`, `*.pem`, `.evidence-key` |
| Files over 5 MB | |

Binary files skip the text-only rules (IPv4, account id, Bearer, AWS secret assignment), because random bytes would match.

Every exemption is written in `prepush_scan.py` (`ALLOWED_IPV4`, `ALLOWED_ACCOUNT_IDS`, `PUBLIC_KIT_KEYS`, `PATH_WAIVERS`); add to those lists in a reviewed commit, not on the command line. There is deliberately no inline bypass. If the scan itself fails, the push is blocked (it fails closed).

## Use by hand

```
python3 -I scripts/hooks/prepush_scan.py --refs HEAD --no-evidence       # the whole history reachable from HEAD
python3 -I scripts/hooks/prepush_scan.py --range origin/main..HEAD         # what a push would send
python3 -I scripts/hooks/prepush_scan.py --evidence-only                   # only the evidence/ folder
```

Exit codes: 0 clean, 1 findings (or the scan could not complete), 2 bad arguments.

## Tests

```
python3 -m pytest -q scripts/hooks/tests
```

The tests build throw-away git repos and a throw-away kit folder, push through the real hook into a bare repository, and check that each rule fires, that allowed values pass, that nothing already on the remote is re-scanned, and that no fake secret ever appears in the output. All sensitive-looking values in the tests are fake and assembled at run time.

## Limits

- It checks content it can read. Secrets split across lines, encoded, or inside archives are not found.
- It cannot know about a secret that is not in the kit folder or matched by a pattern.
- It runs on the pusher's machine. A teammate who has not set `core.hooksPath` is not protected; the lead should run `--range` before any push to a shared branch.
