#!/bin/bash

# exit on error
set -e

cat <<'EOF'
        _ _           _
      _| | |_       _| |___ _ _ _ ___
     | . | . |     | . | . | | | |   |
     |___|___|_____|___|___|_____|_|_|
             |_____|
EOF

echo "\ntearing down postgres container..."
docker compose down --volumes

echo "removing dangling images..."
docker image prune --force

echo "database successfully torn down!"
