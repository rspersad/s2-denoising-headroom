#!/usr/bin/env bash
# Run from the directory that contains this scaffolding after copying code/, results/, configs/ into it.
set -euo pipefail
REPO=rspersad/s2-denoising-headroom

# 1. Repository, public from the start (nothing private is ever committed here)
gh repo create "$REPO" --public --description "Data and code for: Perceptual Quality Versus Detection Performance (Persad, 2026)" \
  --disable-wiki --source=. --remote=origin --push --homepage "https://github.com/$REPO"

# 2. Manifest, then the first commit and the immutable release tag
python3 tools/make_manifest.py code results configs docs
git add -A && git commit -m "Publish data and code of record (Praxis v1.12)" && git push -u origin main
git tag -a v1.0.0 -m "Praxis as approved by the committee, 12 November 2026" && git push origin v1.0.0

# 3. Branch protection: no direct pushes, no force-push, no deletion; PRs required even for the owner
gh api -X PUT "repos/$REPO/branches/main/protection" --input - <<'JSON'
{ "required_status_checks": null,
  "enforce_admins": true,
  "required_pull_request_reviews": { "required_approving_review_count": 0 },
  "restrictions": null,
  "allow_force_pushes": false,
  "allow_deletions": false,
  "required_linear_history": true }
JSON

# 4. Tag protection: released versions cannot be moved or deleted
gh api -X POST "repos/$REPO/tags/protection" -f pattern='v*'

# 5. Read-only posture: forks allowed (that is how others work with it), issues on, no outside collaborators
gh repo edit "$REPO" --enable-issues --enable-projects=false --enable-discussions=false --delete-branch-on-merge

# 6. GitHub Release carrying the manifest and a frozen snapshot
gh release create v1.0.0 --title "v1.0.0 — data and code of record" --notes-file RELEASE_NOTES.md MANIFEST.sha256
echo "done: https://github.com/$REPO"
