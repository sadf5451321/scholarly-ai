"""
Neo4j 知识图谱管理器
"""
from neo4j import GraphDatabase
from typing import List, Dict, Any
import logging

logger = logging.getLogger(__name__)


class Neo4jManager:
    def __init__(self, uri: str, user: str, password: str):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
    
    def close(self):
        self.driver.close()
    
    def create_paper_node(self, paper_id: str, title: str, metadata: Dict):
        """创建论文节点"""
        with self.driver.session() as session:
            session.run(
                """
                MERGE (p:Paper {id: $paper_id})
                SET p.title = $title,
                    p.abstract = $metadata.abstract,
                    p.year = $metadata.year,
                    p.venue = $metadata.venue
                RETURN p
                """,
                paper_id=paper_id,
                title=title,
                metadata=metadata
            )
    
    def create_author_node(self, author_name: str, affiliation: str = None):
        """创建作者节点"""
        with self.driver.session() as session:
            session.run(
                """
                MERGE (a:Author {name: $name})
                SET a.affiliation = $affiliation
                RETURN a
                """,
                name=author_name,
                affiliation=affiliation
            )
    
    def create_citation_relation(self, citing_paper: str, cited_paper: str):
        """创建引用关系"""
        with self.driver.session() as session:
            session.run(
                """
                MATCH (p1:Paper {id: $citing})
                MATCH (p2:Paper {id: $cited})
                MERGE (p1)-[r:CITES]->(p2)
                RETURN r
                """,
                citing=citing_paper,
                cited=cited_paper
            )
    
    def query_related_papers(self, paper_id: str, limit: int = 10) -> List[Dict]:
        """查询相关论文"""
        with self.driver.session() as session:
            result = session.run(
                """
                MATCH (p:Paper {id: $paper_id})-[r:CITES]-(related:Paper)
                RETURN related.id as id, related.title as title, type(r) as relation
                LIMIT $limit
                """,
                paper_id=paper_id,
                limit=limit
            )
            return [record.data() for record in result]