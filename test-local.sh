#!/bin/bash

echo "Starting FR Machine II Demo local testing environment..."

# Build and start the Docker services
docker-compose up -d

echo "Services started!"
echo "Frontend available at: http://localhost:8080"
echo "Backend API available at: http://localhost:5001"
echo ""
echo "To check logs:"
echo "  Frontend: docker logs -f fr-machine-frontend"
echo "  Backend:  docker logs -f fr-machine-api"
echo ""
echo "To stop services: docker-compose down" 