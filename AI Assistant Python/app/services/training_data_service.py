from sqlalchemy.orm import Session
from sqlalchemy import desc, func
from typing import List, Optional
import logging
from ..models.database import TrainingData
from ..models.schemas import TrainingDataCreate, TrainingDataUpdate

logger = logging.getLogger(__name__)

class TrainingDataService:
    def __init__(self, db: Session):
        self.db = db
    
    async def find_relevant_examples_async(self, query: str, limit: int = 5) -> List[TrainingData]:
        """Find relevant training examples for a given query"""
        try:
            # Get all validated training data
            all_training = self.db.query(TrainingData).filter(TrainingData.is_validated == 1).all()
            
            if not all_training:
                logger.info("No validated training data found")
                return []
            
            # Calculate similarity scores
            scored_examples = []
            for example in all_training:
                similarity = example.calculate_similarity(query)
                if similarity > 0.1:  # Only include if some relevance
                    # Combine similarity score with usefulness count (weighted)
                    usefulness_boost = int(getattr(example, 'usefulness_count', 0) or 0) * 0.05  # Each useful mark adds 5% to score
                    combined_score = similarity + usefulness_boost
                    scored_examples.append((example, combined_score))
            
            # Sort by combined score (descending) - usefulness count is already factored in
            scored_examples.sort(key=lambda x: x[1], reverse=True)
            
            # Return top matches
            result = [example for example, _ in scored_examples[:limit]]
            logger.info(f"Found {len(result)} relevant training examples for query: {query[:50]}...")
            
            return result
            
        except Exception as ex:
            logger.error(f"Error finding relevant training examples: {ex}")
            return []
    
    def get_all_training_data(self, skip: int = 0, limit: int = 100) -> List[TrainingData]:
        """Get all training data with pagination"""
        return self.db.query(TrainingData).offset(skip).limit(limit).all()
    
    def get_training_data_by_id(self, training_id: int) -> Optional[TrainingData]:
        """Get training data by ID"""
        return self.db.query(TrainingData).filter(TrainingData.id == training_id).first()
    
    def create_training_data(self, training_data: TrainingDataCreate) -> TrainingData:
        """Create new training data"""
        db_training = TrainingData(
            incident_description=training_data.incident_description,
            expected_incident_type=training_data.expected_incident_type,
            expected_pattern_match=training_data.expected_pattern_match,
            expected_root_cause=training_data.expected_root_cause,
            expected_impact=training_data.expected_impact,
            expected_urgency=training_data.expected_urgency,
            category=training_data.category,
            tags=training_data.tags,
            notes=training_data.notes,
            created_by=training_data.created_by,
            is_validated=1 if training_data.is_validated else 0
        )
        
        # Set affected systems
        db_training.expected_affected_systems = training_data.expected_affected_systems
        
        self.db.add(db_training)
        self.db.commit()
        self.db.refresh(db_training)
        
        logger.info(f"Created training data with ID: {db_training.id}")
        return db_training
    
    def add_training_example(self, incident_description: str, resolution_steps: str, source: str = "", category: str = "") -> TrainingData:
        """Add a training example with simple parameters"""
        db_training = TrainingData(
            incident_description=incident_description,
            expected_root_cause=resolution_steps,  # Store resolution steps in root cause field
            category=category,
            created_by=source,
            is_validated=1  # Auto-validate imported data
        )
        
        self.db.add(db_training)
        self.db.commit()
        self.db.refresh(db_training)
        
        logger.info(f"Added training example with ID: {db_training.id}")
        return db_training

    def update_training_data(self, training_id: int, training_update: TrainingDataUpdate) -> Optional[TrainingData]:
        """Update existing training data"""
        db_training = self.get_training_data_by_id(training_id)
        if not db_training:
            return None
        
        # Update only provided fields
        update_data = training_update.dict(exclude_unset=True)
        
        for field, value in update_data.items():
            if field == "expected_affected_systems":
                db_training.expected_affected_systems = value
            elif field == "is_validated":
                db_training.is_validated = 1 if value else 0
            else:
                setattr(db_training, field, value)
        
        self.db.commit()
        self.db.refresh(db_training)
        
        logger.info(f"Updated training data with ID: {training_id}")
        return db_training
    
    def delete_training_data(self, training_id: int) -> bool:
        """Delete training data"""
        db_training = self.get_training_data_by_id(training_id)
        if not db_training:
            return False
        
        self.db.delete(db_training)
        self.db.commit()
        
        logger.info(f"Deleted training data with ID: {training_id}")
        return True
    
    def get_training_data_count(self) -> int:
        """Get total count of training data"""
        return self.db.query(TrainingData).count()
    
    def search_training_data(self, search_term: str) -> List[TrainingData]:
        """Search training data by description or type, prioritizing better matches"""
        search_pattern = f"%{search_term}%"
        # Use current schema fields (incident_description, expected_incident_type, category)
        results = self.db.query(TrainingData).filter(
            (TrainingData.incident_description.like(search_pattern)) |
            (TrainingData.expected_incident_type.like(search_pattern)) |
            (TrainingData.category.like(search_pattern))
        ).all()
        
        # Score results by match quality
        scored_results = []
        search_lower = search_term.lower()
        for result in results:
            desc_lower = (result.incident_description or "").lower()
            # Calculate match score
            score = 0
            # Exact substring match gets highest score
            if search_lower in desc_lower:
                score += 100
            # Count matching words
            search_words = set(search_lower.split())
            desc_words = set(desc_lower.split())
            matching_words = len(search_words & desc_words)
            score += matching_words * 10
            # Add usefulness as tiebreaker
            score += int(getattr(result, 'usefulness_count', 0) or 0)
            
            scored_results.append((result, score))
        
        # Sort by score (descending), then by usefulness
        scored_results.sort(key=lambda x: (x[1], int(getattr(x[0], 'usefulness_count', 0) or 0)), reverse=True)
        
        return [result for result, _ in scored_results]