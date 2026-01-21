# Docker Deployment

This directory contains Docker configuration for deploying the Professional Trading Bot.

## Quick Start

### 1. Configure Environment

```bash
cp .env.example .env
# Edit .env with your settings
```

### 2. Build and Run

**Simulation Mode (default):**
```bash
docker-compose up -d trading-bot
```

**With Settings UI:**
```bash
docker-compose up -d trading-bot settings-ui
```

**Full stack with database and cache:**
```bash
docker-compose --profile with-db --profile with-cache up -d
```

### 3. Access Services

- **Trading Bot API:** http://localhost:9000
- **Settings UI:** http://localhost:8001 (if enabled)
- **Health Check:** http://localhost:9000/health

## Services

### trading-bot
The main trading bot service.

**Environment Variables:**
- `TRADING_MODE`: `simulation`, `paper`, or `live`
- `SCHWAB_API_KEY`: Schwab API key
- `SCHWAB_API_SECRET`: Schwab API secret
- `SCHWAB_ACCOUNT_ID`: Schwab account ID
- `LOG_LEVEL`: Logging level (DEBUG, INFO, WARNING, ERROR)

### settings-ui
Web-based settings configuration UI.

### redis (optional)
Redis cache for performance optimization.

### postgres (optional)
PostgreSQL database for trade history and analytics.

## Volumes

- `trading-data`: Persistent trading state and data
- `trading-logs`: Log files
- `redis-data`: Redis persistence
- `postgres-data`: PostgreSQL data

## Commands

**View logs:**
```bash
docker-compose logs -f trading-bot
```

**Stop all services:**
```bash
docker-compose down
```

**Rebuild after code changes:**
```bash
docker-compose build --no-cache
docker-compose up -d
```

**Shell access:**
```bash
docker-compose exec trading-bot /bin/bash
```

## Production Deployment

For production:

1. Use strong passwords in `.env`
2. Enable TLS/SSL with a reverse proxy (nginx, traefik)
3. Set up proper logging and monitoring
4. Use secrets management (Docker secrets, Vault)
5. Configure proper network security

Example nginx configuration:
```nginx
server {
    listen 443 ssl;
    server_name trading.example.com;

    ssl_certificate /path/to/cert.pem;
    ssl_certificate_key /path/to/key.pem;

    location / {
        proxy_pass http://localhost:9000;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
    }
}
```

## Security Notes

- Never commit `.env` files with real credentials
- Use read-only volume mounts where possible
- Run containers as non-root user (default)
- Regularly update base images
- Monitor container resources and logs
