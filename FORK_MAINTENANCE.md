# Fork Maintenance

## One-time setup

Add the upstream remote:

```bash
git remote add upstream <upstream-repo-url>
```

Upstream:

```bash
git remote add upstream https://github.com/dograh-hq/dograh.git
```

Fetch upstream and init submodules:

```bash
git fetch upstream
git submodule update --init --recursive
```

## Keep fork synced (rebase)

Rebase your main branch onto upstream:

```bash
git fetch upstream
git checkout main
git rebase upstream/main
```

If there are conflicts:

```bash
git status
git add <resolved-files>
git rebase --continue
```

Abort if needed:

```bash
git rebase --abort
```

If you already pushed your branch:

```bash
git push --force-with-lease
```

## Keep feature branches current

```bash
git checkout <branch>
git rebase main
```

## Strategy for minimal conflicts

- Keep fork-only commits small and focused
- Avoid formatting-only changes
- Prefer minimal additions over refactors
- When conflicts happen, keep upstream logic and re-apply fork deltas
- Keep all fork changes squashed into a single commit on `main`

## Fork images in Docker Compose

After publishing to GHCR:

```bash
REGISTRY=ghcr.io/<your-org-or-user> docker compose up --pull always
```
