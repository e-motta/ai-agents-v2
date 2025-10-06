.PHONY: up up-build down build logs

up:
	docker compose -f docker-compose.yml -f docker-compose.dev.yml up -d

up-build:
	docker compose -f docker-compose.yml -f docker-compose.dev.yml up -d --build

down:
	docker compose -f docker-compose.yml -f docker-compose.dev.yml down

build:
	docker compose -f docker-compose.yml -f docker-compose.dev.yml build

logs:
	docker compose -f docker-compose.yml -f docker-compose.dev.yml logs -f $(service)
