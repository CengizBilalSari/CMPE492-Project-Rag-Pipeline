# CMPE492 Project GraphRAG Pipeline

## Project Structure

The main operational codebase, orchestrations, and microservices for the application are located in the `Product/` directory. You will need to navigate to this directory to configure your environment and run the application.

## Configuration (.env)

Before starting the application, you must configure the environment variables by creating a `.env` file inside the `Product/` directory. 

1. Navigate to the `Product` directory:
   ```bash
   cd Product
   ```

2. Copy the sample environment file to create your own:
   ```bash
   cp .env.sample .env
   # In Windows PowerShell, you can use: Copy-Item .env.sample .env
   ```

3. Open `.env` in a text editor and fill in the corresponding values. The `.env` file should contain at least:

   ```env
   # Required: Your OpenAI API key for LLM calls (unless you are exclusively using a local LLM via LMStudio)
   OPENAI_API_KEY=sk-...

   # Optional: Complete this if you intend to use a local LLM with LMStudio 
   LMSTUDIO_BASE_URL=http://host.docker.internal:1234/v1

   # Database settings configured by default for Docker - you generally do not need to change these
   DATABASE_URL=postgresql://graphrag:graphrag@postgres:5432/graphrag

   # Container document storage binding path
   STORAGE_BASE=/docs
   ```

## Running the Application

This project utilizes Docker Compose to manage the databases (Neo4j, PostgreSQL) and its microservices (Pipeliner API, Evaluator API, Dashboard UI).

1. Ensure you have Docker and Docker Compose installed and running on your system.
2. Inside your terminal, make sure you are in the `Product/` directory where the `docker-compose.yml` file is located:
   ```bash
   cd Product
   ```
3. Build the container images and start all services in the background (detached mode):
   ```bash
   docker compose up -d --build
   ```

You can view the logs of your running containers with:
```bash
docker compose logs -f
```

### Accessing the Services
Once the containers are properly built and started, here are the default endpoints to access the tools:
- **Neo4j Graph Database UI:** [http://localhost:7474](http://localhost:7474) (Default Login: `neo4j` / `12345678`)
- **Dashboard (Frontend React App):** [http://localhost:3000](http://localhost:3000)
- **Pipeliner FastAPI Backend:** [http://localhost:8000/docs](http://localhost:8000/docs)
- **Evaluator FastAPI Backend:** [http://localhost:8001/docs](http://localhost:8001/docs)

### Stopping the Services
To stop and remove the containers, networks, and services created by Docker Compose, run:
```bash
docker compose down
```
*(Note: If you want to also wipe the persistent database volumes, use `docker compose down -v`)*
