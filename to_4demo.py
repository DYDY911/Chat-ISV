#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Neo4j 
"""

import json
import uuid
import re
from pathlib import Path
from typing import Dict, Any, Optional
from neo4j import GraphDatabase, basic_auth
from tqdm import tqdm

# ========================================
JSONL_PATH = Path(r"D:\\X\X\\X\X\\llmpdf_extracted.clean.jsonl")
NEO4J_URI  = "bolt://XXXXXXXX"
NEO4J_USER = "neo4j"
NEO4J_PASS = "XXXXXXXXX"
BATCH_SIZE = 200
# ==============================================

ENTITY_LABELS = [
    "Process", "EmissionSource", "VOCSpecies", "ControlTech",
    "Method", "Regulation", "Factor", "Mechanism", "Scenario"
]

def norm(x):
    if x is None:
        return None
    x = re.sub(r"\s+", " ", str(x).strip())
    return x or None

# ==================== Chunk ====================

def ensure_chunk(tx, chunk_id: str, doc_id: str, text: str):
    if text and text.strip():
        tx.run(
            """
            MERGE (c:Chunk {chunk_id: $cid})
            SET c.doc_id = $doc_id,
                c.text   = $text
            """,
            cid=chunk_id,
            doc_id=doc_id,
            text=text
        )
    else:
        tx.run(
            """
            MERGE (c:Chunk {chunk_id: $cid})
            SET c.doc_id = $doc_id
            """,
            cid=chunk_id,
            doc_id=doc_id
        )




# ==================== Entity ====================

def ensure_entity(tx, label: str, payload: Dict[str, Any], chunk_id: str):
    name = norm(payload.get("canonical_name") or payload.get("name"))
    if not name:
        return

    aliases = payload.get("aliases") or []
    aliases = [norm(a) for a in aliases if norm(a)]

    props = {
        "name": name,
        "aliases": aliases,
        "confidence": payload.get("confidence"),
        "evidence_span": payload.get("evidence_span"),
        "source_doc": payload.get("provenance", {}).get("doc_id")
    }

    props = {k: v for k, v in props.items() if v is not None}

    tx.run(
        f"""
        MERGE (n:{label} {{name: $name}})
        SET n += $props
        """,
        name=name,
        props=props
    )
    query = f"""
    MATCH (n:{label} {{name: $name}})
    MATCH (c:Chunk {{chunk_id: $cid}})
    MERGE (c)-[:MENTIONS]->(n)
    RETURN n.name as entity_name, c.chunk_id as chunk_id
    """
    
    tx.run(
        query,
        name=name,
        cid=chunk_id
    )

# ==================== Find Node ====================

def find_node(tx, name: str) -> Optional[int]:
    name = norm(name)
    if not name:
        return None

    for label in ENTITY_LABELS:
        res = tx.run(
            f"""
            MATCH (n:{label})
            WHERE n.name = $name OR $name IN n.aliases
            RETURN id(n) AS id LIMIT 1
            """,
            name=name
        ).single()
        if res:
            return res["id"]
    return None

# ==================== Relations ====================

def merge_relation(tx, head, rel, tail, props):
    hid = find_node(tx, head)
    tid = find_node(tx, tail)
    if hid is None or tid is None:
        return

    rel = re.sub(r"[^\w]+", "_", rel.upper())
    props = {k: v for k, v in props.items() if v is not None}

    tx.run(
        f"""
        MATCH (a) WHERE id(a)=$h
        MATCH (b) WHERE id(b)=$t
        MERGE (a)-[r:{rel}]->(b)
        SET r += $props
        """,
        h=hid,
        t=tid,
        props=props
    )

# ==================== Ingest ====================

def ingest_record(tx, rec: Dict[str, Any]):
    doc_id = rec.get("doc_id")

    evidence_texts = []

    def collect_text(t):
        """统一处理 str / list[str]"""
        if isinstance(t, str):
         if t.strip():
            evidence_texts.append(t.strip())
         elif isinstance(t, list):
          for x in t:
            if isinstance(x, str) and x.strip():
                evidence_texts.append(x.strip())
# 1️⃣ 
    for ents in (rec.get("entities") or {}).values():
     for e in ents or []:
        if isinstance(e, dict):
            collect_text(e.get("evidence_span"))
# 2️⃣ 
    for r in rec.get("relations") or []:
     collect_text(r.get("evidence_text"))
    # 3️⃣ 
    chunk_text = "\n".join(dict.fromkeys(evidence_texts))  

    if not chunk_text:
        return
    chunk_id = doc_id

    ensure_chunk(tx, chunk_id, doc_id, chunk_text)

    # ===============================
    for label, ents in (rec.get("entities") or {}).items():
        if label not in ENTITY_LABELS:
            continue
        for e in ents:
            if isinstance(e, dict):
                ensure_entity(tx, label, e, chunk_id)

    # ===============================
    for r in rec.get("relations") or []:
        merge_relation(
            tx,
            r.get("head"),
            r.get("relation"),
            r.get("tail"),
            r
        )



# ==================== Main ====================

def main():
    driver = GraphDatabase.driver(
        NEO4J_URI,
        auth=basic_auth(NEO4J_USER, NEO4J_PASS)
    )

    with driver.session() as session:
        total = sum(1 for _ in open(JSONL_PATH, encoding="utf-8"))
        buf = []

        with open(JSONL_PATH, encoding="utf-8") as f:
            for line in tqdm(f, total=total):
                rec = json.loads(line)
                buf.append(rec)
                if len(buf) >= BATCH_SIZE:
                    session.execute_write(lambda tx: [ingest_record(tx, r) for r in buf])
                    buf.clear()

            if buf:
                session.execute_write(lambda tx: [ingest_record(tx, r) for r in buf])

    driver.close()
    print("✅ finish")

if __name__ == "__main__":
    main()
