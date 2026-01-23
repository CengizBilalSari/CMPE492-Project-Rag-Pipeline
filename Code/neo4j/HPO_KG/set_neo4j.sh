#!/bin/bash



# For plugins, under plugins folder:
#wget https://github.com/neo4j/apoc/releases/download/5.14.0/apoc-5.14.0-core.jar \
#    -O apoc.jar
# wget https://github.com/neo4j-labs/neosemantics/releases/download/5.14.0/neosemantics-5.14.0.jar

#Neo4j Container Setup for HPO Knowledge Graph
# This script initializes a Neo4j container with necessary plugins 
# and security permissions for Neosemantics and APOC.
docker run --name neo4j-hpo-container \
    -p 7474:7474 -p 7687:7687 \
    -d \
    --restart unless-stopped \
    -v "$(pwd)/plugins:/plugins" \
    -v "$(pwd)/data:/var/lib/neo4j/data" \
    -v "$(pwd)/import:/var/lib/neo4j/import" \
    -e NEO4J_PLUGINS='["apoc", "graph-data-science"]' \
    -e NEO4J_AUTH=..../.... \
    -e NEO4J_dbms_security_procedures_unrestricted=apoc.*,n10s.*,gds.* \
    -e NEO4J_dbms_security_procedures_allowlist=apoc.*,n10s.*,gds.* \
    -e NEO4J_dbms_memory_heap_max__size=2G \
    neo4j:latest
# --- Command Explanation ---
# --name: Assigns a unique name to the container for easy management.
# -p 7474:7474: Maps the HTTP port for the Neo4j Browser.
# -p 7687:7687: Maps the Bolt port for Python driver connections.
# -d: Runs the container in "detached" mode (background).
# --restart: Ensures the DB restarts automatically if the system reboots.
# NEO4J_PLUGINS: Downloads APOC and Neosemantics (n10s) on startup.
# NEO4J_AUTH: Sets the initial username and password.
# unrestricted/allowlist: Grants plugins permission to run advanced procedures.