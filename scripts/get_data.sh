#!/usr/bin/env bash
# Fetch the public-domain Pierce corpus into data/raw/ and verify it byte-for-byte.
set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
RAW_DIR="${REPO_ROOT}/data/raw"
OUT_PDF="${RAW_DIR}/pierce-peoples-common-sense-medical-adviser-1890.pdf"
PARTIAL_PDF="${OUT_PDF}.part"

IA_IDENTIFIER="peoplescommonsen00pier"
SOURCE_URL="https://archive.org/download/${IA_IDENTIFIER}/${IA_IDENTIFIER}.pdf"
EXPECTED_BYTES="65311598"
EXPECTED_SHA256="841b1feb55ff0aff5735c3aeb308eb52e217f91ae55c5d34e21feb6a640c8896"

sha256_file() {
  if command -v sha256sum >/dev/null 2>&1; then
    sha256sum "$1" | awk '{print $1}'
  elif command -v shasum >/dev/null 2>&1; then
    shasum -a 256 "$1" | awk '{print $1}'
  else
    echo "error: install sha256sum or shasum to verify the corpus" >&2
    return 1
  fi
}

size_bytes() {
  wc -c <"$1" | tr -d '[:space:]'
}

verify_pdf() {
  local file="$1"
  [[ -f "$file" ]] || return 1

  local actual_bytes actual_sha256
  actual_bytes="$(size_bytes "$file")"
  actual_sha256="$(sha256_file "$file")"

  if [[ "$actual_bytes" != "$EXPECTED_BYTES" ]]; then
    echo "error: size mismatch for ${file}: ${actual_bytes} != ${EXPECTED_BYTES}" >&2
    return 1
  fi
  if [[ "$actual_sha256" != "$EXPECTED_SHA256" ]]; then
    echo "error: SHA-256 mismatch for ${file}" >&2
    return 1
  fi
}

download_pdf() {
  if command -v curl >/dev/null 2>&1; then
    curl --fail --location --retry 3 --continue-at - \
      --output "$PARTIAL_PDF" "$SOURCE_URL"
  elif command -v wget >/dev/null 2>&1; then
    wget --continue --tries=3 --output-document="$PARTIAL_PDF" "$SOURCE_URL"
  else
    echo "error: install curl or wget to download the corpus" >&2
    return 1
  fi
}

mkdir -p "$RAW_DIR"

if verify_pdf "$OUT_PDF"; then
  rm -f "$PARTIAL_PDF"
  echo "Corpus already present and verified: ${OUT_PDF}"
  exit 0
fi

if [[ -f "$PARTIAL_PDF" ]]; then
  if verify_pdf "$PARTIAL_PDF"; then
    mv "$PARTIAL_PDF" "$OUT_PDF"
    echo "Recovered verified partial download: ${OUT_PDF}"
    exit 0
  fi
  if [[ "$(size_bytes "$PARTIAL_PDF")" -ge "$EXPECTED_BYTES" ]]; then
    echo "Discarding incompatible completed partial download." >&2
    rm -f "$PARTIAL_PDF"
  fi
fi

echo "Downloading Pierce (1890) from Internet Archive..."
download_pdf
if ! verify_pdf "$PARTIAL_PDF"; then
  rm -f "$PARTIAL_PDF"
  echo "error: downloaded corpus failed integrity verification" >&2
  exit 1
fi
mv "$PARTIAL_PDF" "$OUT_PDF"
echo "Corpus downloaded and verified: ${OUT_PDF}"
