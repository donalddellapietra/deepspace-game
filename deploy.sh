#!/usr/bin/env bash
set -euo pipefail

# Build React UI first
(cd ui && npm run build)

# Build WASM game + bundle React output
trunk build --release --public-url "./"
cp vercel.json dist/
cd dist && vercel deploy --prod --yes
