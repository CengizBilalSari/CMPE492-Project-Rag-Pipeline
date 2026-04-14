# Neo4j Container Setup for HPO Knowledge Graph
# This script initializes a Neo4j container with necessary plugins
# and security permissions for Neosemantics and APOC.

docker run --name neo4j-hpo-container `
    -p 7474:7474 -p 7687:7687 `
    -d `
    --restart unless-stopped `
    -v "${PWD}/plugins:/plugins" `
    -v "${PWD}/data:/var/lib/neo4j/data" `
    -v "${PWD}/import:/var/lib/neo4j/import" `
    -e NEO4J_PLUGINS='["apoc", "graph-data-science"]' `
    -e NEO4J_AUTH=neo4j/pass12345 `
    -e NEO4J_dbms_security_procedures_unrestricted="apoc.*,n10s.*,gds.*" `
    -e NEO4J_dbms_security_procedures_allowlist="apoc.*,n10s.*,gds.*" `
    -e NEO4J_dbms_memory_heap_max__size=2G `
    neo4j:latest
