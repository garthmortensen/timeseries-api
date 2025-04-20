# Postgres container

## start and stop container

start db container, create basic db:

```sh
bash db_up.sh
```

tear down db container:

```sh
bash db_down.sh
```

## maintenance commands

check if container running:

```sh
docker ps --all
```

run container (in the background):
```sh
docker-compose up --detach
```

connect to postgres and check the table found in `init.sql`:

```sh
docker exec --interactive --tty my_postgres_container psql --username=myuser --dbname=mydatabase
```

verify it's working:
```sql
select  * from cats
```

restart container:
```sh
docker compose down --volumes
docker compose up --detach
```
