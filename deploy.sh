#!/usr/bin/env bash
set -euo pipefail

trunk build --release --public-url "./"
cp vercel.json dist/
cd dist && vercel deploy --prod --yes
