from sqlalchemy.orm import Session
from sqlalchemy import desc, func
from typing import List, Optional
from datetime import datetime
import logging
from ..models.database import KnowledgeBase
from ..models.schemas import KnowledgeBaseCreate, KnowledgeBaseUpdate

logger = logging.getLogger(__name__)

class KnowledgeBaseService:
    def __init__(self, db: Session):
        self.db = db
    
    async def find_relevant_knowledge_async(self, query: str, limit: int = 5) -> List[KnowledgeBase]:
        """Find relevant knowledge entries for a given query"""
        try:
            # Get active knowledge entries
            all_knowledge = self.db.query(KnowledgeBase).filter(KnowledgeBase.status == "Active").all()
            if not all_knowledge:
                logger.info("No active knowledge entries found")
                return []
            # Calculate relevance scores
            scored_entries = []
            for entry in all_knowledge:
                relevance = entry.calculate_relevance(query)
                if relevance > 0.1:
                    usefulness_boost = entry.usefulness_count * 0.05
                    combined_score = relevance + usefulness_boost
                    scored_entries.append((entry, combined_score))
            scored_entries.sort(key=lambda x: x[1], reverse=True)
            # Update usage stats for top matches (but do not commit here)
            result = []
            for entry, score in scored_entries[:limit]:
                entry.view_count += 1
                entry.last_used = datetime.utcnow()
                result.append(entry)
            logger.info(f"Found {len(result)} relevant knowledge entries for query: {query[:50]}...")
            return result
        except Exception as ex:
            logger.error(f"Error finding relevant knowledge: {ex}")
            return []
    
    def get_all_knowledge(self, skip: int = 0, limit: int = 100) -> List[KnowledgeBase]:
        """Get all knowledge entries with pagination"""
        return self.db.query(KnowledgeBase).offset(skip).limit(limit).all()
    
    def get_knowledge_by_id(self, knowledge_id: int) -> Optional[KnowledgeBase]:
        """Get knowledge entry by ID"""
        return self.db.query(KnowledgeBase).filter(KnowledgeBase.id == knowledge_id).first()
    
    def create_knowledge(self, knowledge_data: KnowledgeBaseCreate) -> KnowledgeBase:
        """Create new knowledge entry"""
        db_knowledge = KnowledgeBase(
            title=knowledge_data.title,
            content=knowledge_data.content,
            category=knowledge_data.category,
            type=knowledge_data.type,
            tags=knowledge_data.tags,
            keywords=knowledge_data.keywords,
            priority=knowledge_data.priority,
            source=knowledge_data.source,
            status=knowledge_data.status,
            created_by=knowledge_data.created_by,
            version_notes=knowledge_data.version_notes
        )
        
        self.db.add(db_knowledge)
        self.db.commit()
        self.db.refresh(db_knowledge)
        
        logger.info(f"Created knowledge entry with ID: {db_knowledge.id}")
        return db_knowledge
    
    def update_knowledge(self, knowledge_id: int, knowledge_update: KnowledgeBaseUpdate) -> Optional[KnowledgeBase]:
        """Update existing knowledge entry"""
        db_knowledge = self.get_knowledge_by_id(knowledge_id)
        if not db_knowledge:
            return None
        
        # Update only provided fields
        update_data = knowledge_update.dict(exclude_unset=True)
        
        for field, value in update_data.items():
            setattr(db_knowledge, field, value)
        
        db_knowledge.updated_at = datetime.utcnow()
        
        self.db.commit()
        self.db.refresh(db_knowledge)
        
        logger.info(f"Updated knowledge entry with ID: {knowledge_id}")
        return db_knowledge
    
    def delete_knowledge(self, knowledge_id: int) -> bool:
        """Delete knowledge entry"""
        db_knowledge = self.get_knowledge_by_id(knowledge_id)
        if not db_knowledge:
            return False
        
        self.db.delete(db_knowledge)
        self.db.commit()
        
        logger.info(f"Deleted knowledge entry with ID: {knowledge_id}")
        return True
    
    def get_knowledge_count(self) -> int:
        """Get total count of knowledge entries"""
        return self.db.query(KnowledgeBase).count()
    
    def search_knowledge(self, search_term: str) -> List[KnowledgeBase]:
        """Search knowledge entries by title, content, or keywords"""
        search_pattern = f"%{search_term}%"
        return self.db.query(KnowledgeBase).filter(
            (KnowledgeBase.title.like(search_pattern)) |
            (KnowledgeBase.content.like(search_pattern)) |
            (KnowledgeBase.keywords.like(search_pattern)) |
            (KnowledgeBase.category.like(search_pattern))
        ).all()
    
    def get_knowledge_by_category(self, category: str) -> List[KnowledgeBase]:
        """Get knowledge entries by category"""
        return self.db.query(KnowledgeBase).filter(KnowledgeBase.category == category).all()
    
    def get_knowledge_by_type(self, knowledge_type: str) -> List[KnowledgeBase]:
        """Get knowledge entries by type"""
        return self.db.query(KnowledgeBase).filter(KnowledgeBase.type == knowledge_type).all()
    
    def import_from_word_content(self, content: str, title: str, category: str = "", source: str = "Word Document Import") -> KnowledgeBase:
        """Import knowledge from Word document content"""
        
        # Extract keywords from content (simple approach)
        words = content.lower().split()
        common_words = {'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'can', 'must', 'shall', 'a', 'an', 'this', 'that', 'these', 'those'}
        keywords = [word for word in set(words) if len(word) > 3 and word not in common_words][:20]
        
        knowledge_data = KnowledgeBaseCreate(
            title=title,
            content=content,
            category=category,
            type="Procedure" if "procedure" in title.lower() or "step" in content.lower() else "Reference",
            keywords=", ".join(keywords),
            source=source,
            priority=2,  # Medium priority for imported content
            created_by="System Import"
        )
        
        return self.create_knowledge(knowledge_data)