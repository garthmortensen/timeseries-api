.PHONY: docker-clean

docker-clean:
	docker container prune -f
	docker image prune -f
	docker network prune -f

docker-clean-all:
	docker system prune -f

# Run this before starting development
dev-setup: docker-clean
	docker-compose up -d
