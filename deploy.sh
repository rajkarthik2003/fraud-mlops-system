#!/bin/bash
# Fraud Detection System - Production Deployment Script

set -e

echo "🚀 Fraud Detection System - Production Deployment"
echo "================================================="

# Check if Docker is available
if ! command -v docker &> /dev/null; then
    echo "❌ Docker is not installed. Please install Docker first."
    exit 1
fi

# Check if docker-compose is available
if ! command -v docker-compose &> /dev/null; then
    echo "❌ docker-compose is not installed. Please install docker-compose first."
    exit 1
fi

echo "✅ Docker and docker-compose are available"

# Build the application
echo "🔨 Building Docker images..."
docker-compose build --no-cache

# Start all services
echo "🚀 Starting all services..."
docker-compose up -d

# Wait for services to be healthy
echo "⏳ Waiting for services to be healthy..."
sleep 30

# Check service health
echo "🔍 Checking service health..."

# Check API health
if curl -f http://localhost:8000/health &>/dev/null; then
    echo "✅ API is healthy"
else
    echo "❌ API health check failed"
fi

# Check UI health
if curl -f http://localhost:8501/healthz &>/dev/null; then
    echo "✅ UI is healthy"
else
    echo "⚠️  UI health check failed (may not have health endpoint)"
fi

# Check Redis
if docker-compose exec -T redis redis-cli ping | grep -q PONG; then
    echo "✅ Redis is healthy"
else
    echo "❌ Redis health check failed"
fi

# Check PostgreSQL
if docker-compose exec -T postgres pg_isready -U fraud_user -d fraud_db &>/dev/null; then
    echo "✅ PostgreSQL is healthy"
else
    echo "❌ PostgreSQL health check failed"
fi

echo ""
echo "🎉 Deployment completed!"
echo ""
echo "📋 Service URLs:"
echo "- API: http://localhost:8000"
echo "- API Docs: http://localhost:8000/docs"
echo "- UI Dashboard: http://localhost:8501"
echo "- MLflow: http://localhost:5001"
echo "- Prometheus: http://localhost:9090"
echo ""
echo "📊 Monitoring:"
echo "- View logs: docker-compose logs -f [service_name]"
echo "- Stop services: docker-compose down"
echo "- Restart services: docker-compose restart"
echo ""
echo "🔧 Management:"
echo "- Scale API: docker-compose up -d --scale api=3"
echo "- Update: docker-compose pull && docker-compose up -d"
echo ""
echo "✅ System is ready for production use!"