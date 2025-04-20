#!/bin/bash

# exit on error
set -e

cat <<'EOF'
        _ _
      _| | |_       _ _ ___
     | . | . |     | | | . |
     |___|___|_____|___|  _|
             |_____|   |_|
EOF

echo "\nsetting up postgres container..."
docker compose up --detach

echo "waiting for container to start..."
until docker exec my_postgres_container pg_isready --username=myuser --dbname=mydatabase; do
  sleep 2
done

echo "running schema/ddl..."
docker exec -i my_postgres_container psql --username=myuser --dbname=mydatabase < schema.sql

echo "running seed/dml..."
docker exec -i my_postgres_container psql --username=myuser --dbname=mydatabase < seed.sql

echo "fetching records from table 'cats'..."
docker exec -i my_postgres_container psql --username=myuser --dbname=mydatabase -c "select * from cats limit 1;"

echo "database successfully set up!"