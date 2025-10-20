from fastapi import APIRouter, Form, Depends
from sqlalchemy.orm import Session
from app.models.database import SolutionFeedback, KnowledgeBase, TrainingData
from app.database import get_db
from datetime import datetime
from typing import Optional

router = APIRouter()

@router.post("/api/unmark-step-useful")
async def unmark_step_useful(
    step_order: int = Form(...),
    step_description: str = Form(...),
    knowledge_base_id: Optional[int] = Form(None),
    training_data_id: Optional[int] = Form(None),
    rca_id: Optional[int] = Form(None),
    db: Session = Depends(get_db)
):
    """Unmark a specific resolution step as useful and remove/decrement feedback"""
    try:
        # Find the feedback entry
        feedback = db.query(SolutionFeedback).filter(
            SolutionFeedback.solution_description == step_description,
            SolutionFeedback.solution_order == step_order,
            SolutionFeedback.knowledge_base_id == knowledge_base_id,
            SolutionFeedback.training_data_id == training_data_id,
            SolutionFeedback.rca_id == rca_id
        ).first()
        if not feedback:
            return {"success": False, "message": "No feedback found to remove."}
        # Decrement usefulness count or delete
        if feedback.usefulness_count > 1:
            feedback.usefulness_count -= 1
            feedback.marked_at = datetime.utcnow()
        else:
            db.delete(feedback)
        # Also decrement usefulness count in source table
        if knowledge_base_id:
            kb = db.query(KnowledgeBase).filter_by(id=knowledge_base_id).first()
            if kb and kb.usefulness_count > 0:
                kb.usefulness_count -= 1
        elif training_data_id:
            td = db.query(TrainingData).filter_by(id=training_data_id).first()
            if td and td.usefulness_count and td.usefulness_count > 0:
                td.usefulness_count -= 1
        db.commit()
        return {"success": True, "message": "Step unmarked as useful."}
    except Exception as ex:
        db.rollback()
        return {"success": False, "error": str(ex)}
