#!/bin/sh
# SPDX-License-Identifier: Apache-2.0
# reFLOP installer: downloads a release from GitHub, verifies its SHA-256
# checksum, and installs it. Adapted from recom's installer (ADR-007).
#
#   Guest machine (macOS or Linux), the reflop client:
#     curl -fsSL https://raw.githubusercontent.com/chaseungjoon/reflop/main/install.sh | sh
#
#   Owner machine (Linux, the one with the GPU), reflopd:
#     curl -fsSL https://raw.githubusercontent.com/chaseungjoon/reflop/main/install.sh | sh -s -- --server
#
# Options:
#   --server            install reflopd instead of the client
#   --client            install the client (default)
#   --version vX.Y.Z    install a specific release (default: latest)
#
# Environment:
#   REFLOP_VERSION      same as --version
#   REFLOP_INSTALL_DIR  where reflopd (or the reflop command) goes (default: ~/.local/bin)
#   REFLOP_PYTHON       Python for the client (default: the active venv, else python3)
#
# Nothing here needs root. sudo is used only if REFLOP_INSTALL_DIR names a
# directory you cannot write to.

set -eu

REPO="chaseungjoon/reflop"
BASE_URL="https://github.com/${REPO}/releases"

say() { printf 'reflop-install: %s\n' "$*" >&2; }
die() { say "error: $*"; exit 1; }
has() { command -v "$1" >/dev/null 2>&1; }

download() { # url dest
	if has curl; then
		curl -fsSL --proto '=https' --tlsv1.2 -o "$2" "$1"
	elif has wget; then
		wget -q --https-only -O "$2" "$1"
	else
		die "curl or wget is required"
	fi
}

latest_tag() {
	# github.com/<repo>/releases/latest redirects to .../releases/tag/<tag>;
	# following the redirect avoids the rate-limited GitHub API.
	if has curl; then
		url=$(curl -fsSLI --proto '=https' -o /dev/null -w '%{url_effective}' "${BASE_URL}/latest" || true)
	else
		url=$(wget -q --https-only --spider -S "${BASE_URL}/latest" 2>&1 | sed -n 's/^ *[Ll]ocation: *//p' | tail -n 1)
	fi
	tag=${url##*/}
	tag=$(printf '%s' "$tag" | tr -d '\r')
	case "$tag" in
		v[0-9]*) printf '%s\n' "$tag" ;;
		*) die "could not find a published reflop release (got '${url}'); set REFLOP_VERSION, or build from source (see README)" ;;
	esac
}

sha256_of() {
	if has sha256sum; then
		sha256sum "$1" | cut -d ' ' -f 1
	elif has shasum; then
		shasum -a 256 "$1" | cut -d ' ' -f 1
	else
		die "sha256sum or shasum is required to verify the download"
	fi
}

verify() { # file name sums
	want=$(awk -v f="$2" '$2 == f || $2 == "*"f { print $1 }' "$3")
	[ -n "$want" ] || die "$2 is not listed in checksums.txt"
	got=$(sha256_of "$1")
	[ "$want" = "$got" ] || die "checksum mismatch for $2: expected ${want}, got ${got}"
}

detect_platform() {
	case "$(uname -s)" in
		Darwin) os=darwin ;;
		Linux) os=linux ;;
		*) die "unsupported OS: $(uname -s) (reflop supports macOS and Linux)" ;;
	esac
	case "$(uname -m)" in
		x86_64 | amd64) arch=amd64 ;;
		arm64 | aarch64) arch=arm64 ;;
		*) die "unsupported architecture: $(uname -m) (amd64 and arm64 are supported)" ;;
	esac
}

install_file() { # src dir name allow_sudo
	if [ -d "$2" ] && [ -w "$2" ]; then
		install -m 0755 "$1" "$2/$3"
	elif [ ! -e "$2" ] && mkdir -p "$2" 2>/dev/null; then
		install -m 0755 "$1" "$2/$3"
	elif [ "$4" = yes ] && has sudo; then
		say "installing to $2 requires sudo"
		sudo mkdir -p "$2"
		sudo install -m 0755 "$1" "$2/$3"
	else
		return 1
	fi
}

path_hint() { # dir
	case ":${PATH}:" in *":$1:"*) return ;; esac
	# The tildes below are printed for the user to read, not expanded.
	# shellcheck disable=SC2088
	case "${SHELL:-}" in
		*/zsh) rc='~/.zshrc' ;;
		*/bash) if [ "$os" = darwin ]; then rc='~/.bash_profile'; else rc='~/.bashrc'; fi ;;
		*/fish) say "note: $1 is not on your PATH; add it with: fish_add_path $1"; return ;;
		*) rc='your shell profile' ;;
	esac
	say "note: $1 is not on your PATH; add this line to ${rc} and open a new shell:"
	say "  export PATH=\"$1:\$PATH\""
}

target_dir() {
	if [ -n "${REFLOP_INSTALL_DIR:-}" ]; then
		dir=$REFLOP_INSTALL_DIR
		allow_sudo=yes
	else
		[ -n "${HOME:-}" ] || die "HOME is not set; set REFLOP_INSTALL_DIR"
		dir="${HOME}/.local/bin"
		allow_sudo=no
	fi
}

install_server() {
	[ "$os" = linux ] || die "reflopd runs on Linux (the machine with the GPU); on this machine install the client instead"
	archive="reflopd_${version}_${os}_${arch}.tar.gz"
	say "installing reflopd ${tag} (${os}/${arch})"
	download "${BASE_URL}/download/${tag}/${archive}" "$tmp/$archive" || die "download failed: ${archive}"
	verify "$tmp/$archive" "$archive" "$tmp/checksums.txt"
	say "checksum verified"
	tar -xzf "$tmp/$archive" -C "$tmp" reflopd
	[ -f "$tmp/reflopd" ] || die "the archive did not contain reflopd"
	target_dir
	install_file "$tmp/reflopd" "$dir" reflopd "$allow_sudo" || die "cannot write to ${dir}; set REFLOP_INSTALL_DIR"
	say "installed ${dir}/reflopd"
	"${dir}/reflopd" version >&2 || true
	path_hint "$dir"
	if ! [ -e /proc/driver/nvidia/version ] && ! [ -e /dev/dxg ]; then
		say "note: no NVIDIA driver found; reflopd will serve this machine's CPU only"
	fi
	say "next: reflopd up      (rootless; fetches the pinned runtime, runs a self-test, prints an invite)"
}

pick_python() {
	if [ -n "${REFLOP_PYTHON:-}" ]; then
		py=$REFLOP_PYTHON
	elif [ -n "${VIRTUAL_ENV:-}" ] && [ -x "${VIRTUAL_ENV}/bin/python" ]; then
		py="${VIRTUAL_ENV}/bin/python"
	elif has python3; then
		py=python3
	else
		py=""
	fi
}

install_client() {
	wheel="reflop-${version}-py3-none-any.whl"
	say "installing the reflop client ${tag}"
	download "${BASE_URL}/download/${tag}/${wheel}" "$tmp/$wheel" || die "download failed: ${wheel}"
	verify "$tmp/$wheel" "$wheel" "$tmp/checksums.txt"
	say "checksum verified"
	pick_python
	if [ -n "${VIRTUAL_ENV:-}" ] && [ -n "$py" ]; then
		# Inside a project venv: install the library where your code imports it.
		if has uv; then
			uv pip install --python "$py" "$tmp/$wheel"
		else
			"$py" -m pip install "$tmp/$wheel"
		fi
		say "installed reflop into ${VIRTUAL_ENV}"
	elif has uv; then
		uv tool install --force "$tmp/$wheel"
		say "installed the reflop command with uv (its own environment)"
	else
		[ -n "$py" ] || die "Python 3.10+ is required (or install uv: https://docs.astral.sh/uv/)"
		"$py" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)' || die "Python 3.10+ is required"
		venv="${HOME}/.local/share/reflop/client-venv"
		say "creating ${venv}"
		"$py" -m venv "$venv"
		"$venv/bin/python" -m pip install -q --upgrade pip
		"$venv/bin/python" -m pip install "$tmp/$wheel"
		target_dir
		mkdir -p "$dir"
		ln -sf "$venv/bin/reflop" "$dir/reflop"
		say "installed ${dir}/reflop"
		path_hint "$dir"
	fi
	url="${BASE_URL}/download/${tag}/${wheel}"
	say "to use reflop from your own training code, add it to that project's environment:"
	say "  uv add ${url}        # or: pip install ${url}"
	say "next: reflop pair <invite-code>   (ask the GPU Owner for a code from: reflopd invite --name <you>)"
}

main() {
	mode=client
	tag=${REFLOP_VERSION:-}
	while [ $# -gt 0 ]; do
		case "$1" in
			--server) mode=server ;;
			--client) mode=client ;;
			--version) shift; tag=${1:-} ;;
			--version=*) tag=${1#--version=} ;;
			-h | --help) sed -n '2,24p' "$0" 2>/dev/null || true; exit 0 ;;
			*) die "unknown option: $1" ;;
		esac
		shift
	done
	detect_platform
	[ -n "$tag" ] || tag=$(latest_tag)
	case "$tag" in v*) ;; *) tag="v${tag}" ;; esac
	version=${tag#v}
	tmp=$(mktemp -d 2>/dev/null || mktemp -d -t reflop)
	trap 'rm -rf "$tmp"' EXIT
	trap 'exit 130' INT TERM
	download "${BASE_URL}/download/${tag}/checksums.txt" "$tmp/checksums.txt" || die "download failed: checksums.txt for ${tag}"
	if [ "$mode" = server ]; then
		install_server
	else
		install_client
	fi
}

main "$@"
