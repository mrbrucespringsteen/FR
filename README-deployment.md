# FR Machine II Demo Deployment

This repository contains the frontend and backend code for the FR Machine II Demo.

## Project Structure

- `frontend/`: Static HTML, CSS, and JS files for deployment on AWS Amplify
- `backend/`: Python API server for deployment with Docker

## Frontend Deployment (AWS Amplify)

### Prerequisites
- AWS Account
- AWS CLI configured
- Git repository with this code (GitHub, GitLab, BitBucket, etc.)

### Steps to Deploy on AWS Amplify

1. Push this repository to your Git provider (GitHub, GitLab, BitBucket)

2. Log in to the AWS Management Console and navigate to AWS Amplify

3. Choose "New app" → "Host web app"

4. Connect to your Git provider and select the repository

5. Configure build settings:
   ```yaml
   version: 1
   frontend:
     phases:
       build:
         commands: []
     artifacts:
       baseDirectory: frontend
       files:
         - '**/*'
     cache:
       paths: []
   ```

6. Review and deploy

7. Once deployed, you'll need to update the API endpoint in your JavaScript files to point to your deployed backend URL.

## Backend Deployment (Docker)

### Local Testing

1. Build the Docker image:
   ```
   cd backend
   docker build -t fr-machine-api .
   ```

2. Run the container:
   ```
   docker run -p 5001:5001 fr-machine-api
   ```

3. The API will be available at `http://localhost:5001`

### Deployment Options

#### AWS ECS (Elastic Container Service)

1. Create an ECR repository:
   ```
   aws ecr create-repository --repository-name fr-machine-api
   ```

2. Tag and push your Docker image:
   ```
   aws ecr get-login-password | docker login --username AWS --password-stdin <your-aws-account-id>.dkr.ecr.<region>.amazonaws.com
   docker tag fr-machine-api:latest <your-aws-account-id>.dkr.ecr.<region>.amazonaws.com/fr-machine-api:latest
   docker push <your-aws-account-id>.dkr.ecr.<region>.amazonaws.com/fr-machine-api:latest
   ```

3. Create an ECS cluster, task definition, and service through the AWS console or using AWS CLI/CloudFormation

#### AWS Elastic Beanstalk

1. Install the EB CLI:
   ```
   pip install awsebcli
   ```

2. Initialize EB:
   ```
   cd backend
   eb init
   ```

3. Create an environment:
   ```
   eb create fr-machine-api-env
   ```

4. Deploy:
   ```
   eb deploy
   ```

## Connecting Frontend to Backend

After deploying both the frontend and backend, you'll need to update the API base URL in the frontend JavaScript.

1. In the AWS Amplify Console, navigate to "Environment variables"
2. Add a variable like `API_ENDPOINT` with your backend URL
3. Update your JavaScript to use this environment variable or hardcode the URL of your deployed backend 