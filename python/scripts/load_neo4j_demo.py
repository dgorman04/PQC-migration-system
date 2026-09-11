"""Loads a small, hand-written illustrative graph into Neo4j.

This is NOT the real graph loader — that's `pqc_graph.loader` (README §7.8),
which doesn't exist yet and will read from the Core API once that exists
too. This script exists purely so there is something real to look at in the
Neo4j Browser (http://localhost:7474) before either of those is built —
every node and relationship below is invented for illustration, matching
the schema shape spec §7 describes:

    (Application)-[:USES]->(Certificate)-[:SIGNED_WITH]->(Algorithm)-[:PROTECTS]->(DataAsset)

Two example asset stories, echoing the two example requests used to
demonstrate `pqc-inference` (README §7.6, docs/03):
  - payment-gateway: CRITICAL, uses an RSA-2048 cert, protects payment data
  - internal-wiki: LOW, already on ML-KEM, protects internal docs only

Usage:
    python scripts/load_neo4j_demo.py
    # then open http://localhost:7474 and try the query printed at the end
"""
from __future__ import annotations

import os

from neo4j import GraphDatabase

CYPHER = """
MERGE (payment:Application {name: "payment-gateway"})
  SET payment.criticality = "CRITICAL", payment.internet_accessible = true
MERGE (wiki:Application {name: "internal-wiki"})
  SET wiki.criticality = "LOW", wiki.internet_accessible = false

MERGE (cert1:Certificate {id: "cert-payment-001"})
  SET cert1.not_after = "2026-11-01"
MERGE (cert2:Certificate {id: "cert-wiki-002"})
  SET cert2.not_after = "2027-06-01"

MERGE (rsa:Algorithm {name: "RSA"})
  SET rsa.key_size = 2048, rsa.quantum_vulnerable = true
MERGE (mlkem:Algorithm {name: "ML-KEM"})
  SET mlkem.quantum_vulnerable = false

MERGE (paymentData:DataAsset {name: "customer-payment-data"})
  SET paymentData.classification = "financial"
MERGE (wikiDocs:DataAsset {name: "internal-docs"})
  SET wikiDocs.classification = "internal"

MERGE (payment)-[:USES]->(cert1)
MERGE (cert1)-[:SIGNED_WITH]->(rsa)
MERGE (rsa)-[:PROTECTS]->(paymentData)

MERGE (wiki)-[:USES]->(cert2)
MERGE (cert2)-[:SIGNED_WITH]->(mlkem)
MERGE (mlkem)-[:PROTECTS]->(wikiDocs)
"""

EXAMPLE_QUERY = """MATCH (a:Application)-[:USES]->(:Certificate)-[:SIGNED_WITH]->(alg:Algorithm {name: "RSA"})
WHERE a.criticality = "CRITICAL"
RETURN a.name, alg.name, alg.key_size"""


def main() -> None:
    uri = os.environ.get("NEO4J_URI", "bolt://localhost:7687")
    user, password = os.environ.get("NEO4J_AUTH", "neo4j/pqc_dev_only_change_me").split("/", 1)

    driver = GraphDatabase.driver(uri, auth=(user, password))
    with driver.session() as session:
        session.run(CYPHER)
        (count,) = session.run("MATCH (n) RETURN count(n) AS c").single()
        print(f"loaded demo graph — {count} nodes now in the database")

    driver.close()

    print("\nopen http://localhost:7474 (neo4j / the password in .env) and try:\n")
    print(EXAMPLE_QUERY)
    print("\n-> should return payment-gateway / RSA / 2048")


if __name__ == "__main__":
    main()
