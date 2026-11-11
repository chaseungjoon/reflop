#!/bin/sh
# SPDX-License-Identifier: Apache-2.0
# Sign a draft reFLOP release and publish it (SEC_SWEEP M4).
#
#   scripts/sign-release.sh v0.1.6
#
# release.yml builds the release as a draft. This script, run on the
# maintainer's machine with the offline key:
#   1. downloads the draft's assets and checks them against checksums.txt,
#   2. checks that the draft was built from the tag's commit,
#   3. signs checksums.txt (ssh-keygen -Y sign -n reflop-release),
#   4. verifies the signature against the key embedded in install.sh,
#   5. uploads checksums.txt.sig and publishes the release.
#
# Environment:
#   REFLOP_RELEASE_KEY  private key (default ~/.config/reflop-release/release_ed25519)
set -eu

tag=${1:?usage: scripts/sign-release.sh vX.Y.Z}
key=${REFLOP_RELEASE_KEY:-$HOME/.config/reflop-release/release_ed25519}
repo=chaseungjoon/reflop
cd "$(dirname "$0")/.."

die() { printf 'sign-release: %s\n' "$*" >&2; exit 1; }
[ -f "$key" ] || die "no signing key at $key"
case "$(stat -c %a "$key" 2>/dev/null || stat -f %Lp "$key")" in
	600 | 400) ;;
	*) die "$key must be mode 600" ;;
esac

draft=$(gh release view "$tag" --repo "$repo" --json isDraft --jq .isDraft)
[ "$draft" = true ] || die "$tag is not a draft release (already published?)"

work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT
gh release download "$tag" --repo "$repo" --dir "$work"
[ ! -e "$work/checksums.txt.sig" ] || die "$tag already has a signature"
(cd "$work" && sha256sum --check --strict checksums.txt) || die "assets do not match checksums.txt"
# Every asset except checksums.txt must be listed, so nothing unsigned ships.
for f in "$work"/*; do
	n=$(basename "$f")
	[ "$n" = checksums.txt ] && continue
	grep -q "  $n\$" "$work/checksums.txt" || die "asset $n is not covered by checksums.txt"
done

# .goreleaser.yaml sets target_commitish to the commit it built.
git fetch -q --tags origin
want=$(git rev-parse "$tag^{commit}")
got=$(gh release view "$tag" --repo "$repo" --json targetCommitish --jq .targetCommitish)
[ "$got" = "$want" ] || die "the draft was built from $got, but $tag is $want"

ssh-keygen -Y sign -f "$key" -n reflop-release "$work/checksums.txt" >/dev/null
signers=$(sed -n "s/^RELEASE_SIGNERS='\(.*\)'$/\1/p" install.sh)
[ -n "$signers" ] || die "could not read RELEASE_SIGNERS from install.sh"
printf '%s\n' "$signers" > "$work/allowed_signers"
ssh-keygen -Y verify -f "$work/allowed_signers" -I reflop-release -n reflop-release \
	-s "$work/checksums.txt.sig" < "$work/checksums.txt" || die "the new signature does not verify against install.sh's key"

gh release upload "$tag" --repo "$repo" "$work/checksums.txt.sig"
case "$tag" in
	*-*) gh release edit "$tag" --repo "$repo" --draft=false ;; # a prerelease is never "latest"
	*) gh release edit "$tag" --repo "$repo" --draft=false --latest ;;
esac
printf 'sign-release: published %s\n' "$tag"
