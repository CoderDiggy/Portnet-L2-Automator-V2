from fastapi import FastAPI, Request, Form, Depends, UploadFile, File
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse, RedirectResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from sqlalchemy.orm import Session
from typing import List, Dict, Any, Optional
from starlette.middleware.sessions import SessionMiddleware
import logging
from datetime import datetime
import uuid
from dotenv import load_dotenv
import base64
import os

# Load environment variables from .env file
load_dotenv()


# Import the real services
from app.services.openai_service import OpenAIService
from app.services.knowledge_base_service import KnowledgeBaseService
from app.services.training_data_service import TrainingDataService
from app.services.incident_analyzer import IncidentAnalyzer
from app.services.log_analyzer_service import LogAnalyzerService
from app.services.operational_data_service import OperationalDataService
from app.services.escalation_service import EscalationService
from app.models.database import Base, ResolutionStep, SystemLog, RootCauseAnalysis, KnowledgeBase, TrainingData
from app.models.schemas import EscalationSummary, EscalationTemplate
from app.database import get_db, engine

# Import the unmark-step-useful API
from app.api_unmark_step_useful import router as unmark_step_router

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Create FastAPI app
app = FastAPI(title="AI Duty Officer Assistant", version="1.0.0")
app.include_router(unmark_step_router)

# Add session middleware for storing temporary data
app.add_middleware(SessionMiddleware, secret_key="your-secret-key-here-change-in-production")

# Setup static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Setup templates with correct path
script_dir = os.path.dirname(os.path.abspath(__file__))
templates_dir = os.path.join(script_dir, "app", "templates")
templates = Jinja2Templates(directory=templates_dir)

# Mock data classes for now
class MockIncident:
    def __init__(self, description, source="Manual"):
        self.id = str(uuid.uuid4())
        self.description = description
        self.source = source
        self.reported_at = datetime.now()
        self.status = "New"
        self.title = f"Incident - {source}"  # Add missing title attribute
        self.category = "System Issue"  # Add missing category attribute

# Create database tables
Base.metadata.create_all(bind=engine)

# Initialize the OpenAI service
openai_service = OpenAIService()

async def analyze_image_with_ai(image_content: bytes, content_type: str) -> str:
    """Analyze image using Azure OpenAI Vision API"""
    try:
        # Convert image to base64
        encoded_image = base64.b64encode(image_content).decode('utf-8')
        
        # Use Azure OpenAI Vision to analyze the image
        vision_analysis = await openai_service.analyze_image_async(encoded_image, "Maritime incident documentation")
        
        return f"Visual Analysis: {vision_analysis} "
        
    except Exception as ex:
        logger.error(f"Error analyzing image: {ex}")
        return "[Image analysis failed] "

class MockResolutionStep:
    def __init__(self, order, description, step_type="Analysis"):
        self.order = order
        self.description = description
        self.type = step_type
        self.query = ""

class MockResolutionPlan:
    def __init__(self, incident_type):
        self.summary = f"Analysis completed for {incident_type}"
        self.steps = [
            MockResolutionStep(1, "Initial assessment completed using AI analysis", "Analysis"),
            MockResolutionStep(2, "Investigate root cause based on analysis findings", "Investigation"),
            MockResolutionStep(3, "Implement resolution based on findings", "Resolution")
        ]
        self.diagnostic_queries = []
        self.resolution_queries = []

# Routes
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Home page"""
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/analyze", response_class=HTMLResponse)
async def analyze_get(request: Request):
    """Analyze page - GET"""
    # Hardcoded test cases for quick incident analysis
    test_cases = [
        {
            "title": "EDI Message Processing Error",
            "description": "EDI message stuck in processing queue with timeout error",
            "category": "EDI/API",
            "priority": "High",
            "icon": "fas fa-exchange-alt",
            "source": "System Alert"
        },
        {
            "title": "Container Status Duplication",
            "description": "Customer reports duplicate MSKU0000001 container showing in system",
            "category": "Container Management",
            "priority": "Medium",
            "icon": "fas fa-box",
            "source": "Customer Report"
        },
        {
            "title": "Vessel Arrival Time Conflict",
            "description": "Vessel ETA not updated causing scheduling conflicts at berth",
            "category": "Vessel Operations",
            "priority": "High",
            "icon": "fas fa-ship",
            "source": "Terminal Operations"
        },
        {
            "title": "API Gateway Timeout",
            "description": "API calls failing with 504 timeout error during peak hours",
            "category": "System Infrastructure",
            "priority": "Critical",
            "icon": "fas fa-server",
            "source": "Monitoring System"
        },
        {
            "title": "Gate Transaction Failure",
            "description": "Truck gate transactions failing with database connection timeout",
            "category": "Terminal Operations",
            "priority": "High",
            "icon": "fas fa-truck",
            "source": "Gate Operations"
        },
        {
            "title": "Timezone Display Issue",
            "description": "Container timestamps showing wrong timezone causing milestone confusion",
            "category": "Data Quality",
            "priority": "Medium",
            "icon": "fas fa-clock",
            "source": "User Report"
        },
        {
            "title": "Integration Service Down",
            "description": "PORTNET integration service not responding to external partner requests",
            "category": "External Integration",
            "priority": "Critical",
            "icon": "fas fa-plug",
            "source": "Partner Notification"
        },
        {
            "title": "Booking Reference Error",
            "description": "Unable to retrieve booking information for reference BKG123456789",
            "category": "Booking Management",
            "priority": "Medium",
            "icon": "fas fa-clipboard-list",
            "source": "Customer Service"
        }
    ]
    
    return templates.TemplateResponse("analyze.html", {
        "request": request, 
        "test_cases": test_cases
    })

@app.post("/analyze")
async def analyze_post(
    request: Request,
    incident_description: str = Form(...),
    incident_source: str = Form("Manual"),
    incident_images: List[UploadFile] = File(default=[]),
    db: Session = Depends(get_db)
):
    """Analyze incident - POST"""
    try:
        # AI-powered input validation
        if not await openai_service.is_valid_incident_async(incident_description):
            return RedirectResponse(url="/analyze?error=Invalid incident description. Please provide specific details about the maritime operations issue.", status_code=302)

        # Process uploaded images
        image_analysis = ""
        uploaded_images = []
        if incident_images and incident_images[0].filename:
            logger.info(f"Processing {len(incident_images)} uploaded images")
            uploads_dir = os.path.join(os.path.dirname(__file__), "static", "uploads")
            os.makedirs(uploads_dir, exist_ok=True)
            for image in incident_images:
                if image.filename and image.content_type.startswith('image/'):
                    file_extension = os.path.splitext(image.filename)[1]
                    unique_filename = f"{uuid.uuid4()}{file_extension}"
                    file_path = os.path.join(uploads_dir, unique_filename)
                    content = await image.read()
                    with open(file_path, "wb") as f:
                        f.write(content)
                    uploaded_images.append({
                        "filename": unique_filename,
                        "original_name": image.filename,
                        "path": f"/static/uploads/{unique_filename}",
                        "size": len(content)
                    })
                    image_analysis += await analyze_image_with_ai(content, image.content_type)

        combined_description = incident_description
        if image_analysis:
            combined_description += f"\n\nImage Analysis:\n{image_analysis}"

        incident = MockIncident(combined_description, incident_source)
        
        # Step 1: AI-powered incident analysis
        from app.services.incident_analyzer import IncidentAnalyzer
        analyzer = IncidentAnalyzer(db)
        analysis, knowledge_entries, training_examples = await analyzer.analyze_incident_async(combined_description)
        
        # Step 2: Generate resolution plan based on analysis
        resolution_data = await openai_service.generate_resolution_plan_async(
            combined_description, analysis=analysis, knowledge_entries=knowledge_entries, training_examples=training_examples, db=db)

        all_solutions = resolution_data.get('steps', [])
        total_count = len(all_solutions)
        
        logger.info(f"[Query Flow] Found {total_count} matching solutions")
        
        # For initial page load, return only first 15 solutions
        initial_limit = 15
        initial_solutions = all_solutions[:initial_limit]
        
        # Reassign order numbers starting from 1
        from app.models.database import SolutionFeedback
        for idx, solution in enumerate(initial_solutions, 1):
            solution['order'] = idx
            desc = solution.get('description', solution.get('content', ''))
            feedback_rows = db.query(SolutionFeedback).filter(
                SolutionFeedback.usefulness_count > 0
            ).all()
            user_verified = False
            usefulness_count = 0
            feedback_incident = None
            feedback_solution = None
            for fb in feedback_rows:
                if desc and fb.solution_description and (
                    desc.lower() in fb.solution_description.lower() or fb.solution_description.lower() in desc.lower()
                ):
                    user_verified = True
                    usefulness_count = fb.usefulness_count
                    feedback_incident = fb.incident_description
                    feedback_solution = fb.solution_description
                    logger.info(f"User verified match: Solution '{desc[:50]}' <-> Feedback '{fb.solution_description[:50]}' (count={fb.usefulness_count})")
                    break
            solution['user_verified'] = user_verified
            if user_verified:
                solution['usefulness_count'] = usefulness_count
                solution['feedback_incident_description'] = feedback_incident
                solution['feedback_solution_description'] = feedback_solution
        
        if initial_solutions:
            logger.info(f"Top solution: {initial_solutions[0]}")

        # Store full results in session/cache for lazy loading (using incident ID as key)
        # For now, we'll pass incident_id to frontend and use it to fetch more
        import json
        
        # Generate escalation summary for communication
        escalation_service = EscalationService()
        escalation_summary = escalation_service.generate_escalation_summary(
            incident=incident,
            analysis=analysis,
            solutions_count=total_count
        )
        escalation_templates = escalation_service.generate_escalation_templates(
            incident=incident,
            summary=escalation_summary
        )
        
        # Prepare view model for results.html
        class SolutionViewModel:
            def __init__(self, incident, analysis, resolution_data, uploaded_images=None, total_count=0, initial_limit=15, escalation_summary=None, escalation_templates=None):
                self.incident = incident
                self.analysis = analysis  # Add incident analysis
                self.summary = resolution_data.get("summary", "")
                self.solutions = resolution_data.get("steps", [])[:initial_limit]  # Only initial batch
                self.uploaded_images = uploaded_images or []
                self.total_count = total_count
                self.loaded_count = min(initial_limit, total_count)
                self.has_more = total_count > initial_limit
                self.escalation_summary = escalation_summary
                self.escalation_templates = escalation_templates

        view_model = SolutionViewModel(incident, analysis, resolution_data, uploaded_images, total_count, initial_limit, escalation_summary, escalation_templates)
        
        # Store the incident description for lazy loading endpoint
        request.session[f"incident_{incident.id}"] = {
            "description": combined_description,
            "all_solutions": all_solutions
        }
        
        return templates.TemplateResponse("results.html", {
            "request": request,
            "result": view_model,
            "uploaded_images": uploaded_images
        })
    except Exception as ex:
        import traceback
        logger.error(f"Error analyzing incident: {ex}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return RedirectResponse(url=f"/analyze?error=Analysis failed: {str(ex)}", status_code=302)

@app.get("/api/load-more-solutions/{incident_id}")
async def load_more_solutions(
    request: Request,
    incident_id: str,
    offset: int = 0,
    limit: int = 15
) -> JSONResponse:
    """Lazy loading endpoint: Load more solutions for an incident"""
    try:
        # Retrieve stored solutions from session
        session_key = f"incident_{incident_id}"
        incident_data = request.session.get(session_key)
        
        if not incident_data:
            return JSONResponse(
                status_code=404,
                content={"error": "Incident data not found. Please refresh the page."}
            )
        
        all_solutions = incident_data.get("all_solutions", [])
        
        # Calculate the slice
        start = offset
        end = offset + limit
        more_solutions = all_solutions[start:end]
        
        # Reassign order numbers to continue from current offset + 1
        for idx, solution in enumerate(more_solutions, start + 1):
            solution['order'] = idx
        
        has_more = end < len(all_solutions)
        
        logger.info(f"[Lazy Load] Returning solutions {start}-{end} of {len(all_solutions)} for incident {incident_id}")
        
        return JSONResponse(content={
            "solutions": more_solutions,
            "has_more": has_more,
            "total_count": len(all_solutions),
            "loaded_count": end if end < len(all_solutions) else len(all_solutions)
        })
        
    except Exception as ex:
        logger.error(f"Error loading more solutions: {ex}")
        return JSONResponse(
            status_code=500,
            content={"error": str(ex)}
        )

@app.get("/upload-knowledge")
async def upload_knowledge_get(request: Request):
    """Knowledge upload page"""
    return templates.TemplateResponse("upload_knowledge.html", {"request": request})

@app.get("/knowledge")
async def view_knowledge(request: Request, db: Session = Depends(get_db)):
    """View knowledge base entries"""
    try:
        knowledge_service = KnowledgeBaseService(db)
        entries = knowledge_service.get_all_knowledge(skip=0, limit=100)
        
        return templates.TemplateResponse("knowledge_list.html", {
            "request": request,
            "entries": entries
        })
    except Exception as ex:
        logger.error(f"Error retrieving knowledge: {ex}")
        return templates.TemplateResponse("knowledge_list.html", {
            "request": request,
            "entries": [],
            "error": f"Error loading knowledge base: {str(ex)}"
        })

@app.get("/training")
async def view_training(request: Request, db: Session = Depends(get_db)):
    """View training data entries"""
    try:
        from app.models.database import TrainingData
        training_data = db.query(TrainingData).order_by(TrainingData.created_at.desc()).all()
        
        return templates.TemplateResponse("training.html", {
            "request": request,
            "training_data": training_data
        })
    except Exception as ex:
        logger.error(f"Error retrieving training data: {ex}")
        return templates.TemplateResponse("training.html", {
            "request": request,
            "training_data": [],
            "error": f"Error loading training data: {str(ex)}"
        })

@app.get("/database-detailed")
async def database_detailed(request: Request, db: Session = Depends(get_db)):
    """View detailed database status and contents"""
    try:
        from app.models.database import KnowledgeBase, TrainingData
        
        # Count entries
        kb_count = db.query(KnowledgeBase).count()
        td_count = db.query(TrainingData).count()
        
        # Get recent knowledge entries
        recent_knowledge = db.query(KnowledgeBase).order_by(KnowledgeBase.created_at.desc()).limit(10).all()
        
        # Get recent training data
        recent_training = db.query(TrainingData).order_by(TrainingData.created_at.desc()).limit(5).all()
        
        return templates.TemplateResponse("database_status.html", {
            "request": request,
            "kb_count": kb_count,
            "td_count": td_count,
            "recent_knowledge": recent_knowledge,
            "recent_training": recent_training
        })
    except Exception as ex:
        logger.error(f"Error retrieving database status: {ex}")
        return {"error": str(ex)}

@app.get("/sql-export")
async def sql_export(request: Request):
    """Export database as SQL"""
    try:
        import sqlite3
        
        # Connect to database
        conn = sqlite3.connect('duty_officer_assistant.db')
        cursor = conn.cursor()
        
        sql_content = []
        sql_content.append("-- =====================================================")
        sql_content.append("-- DUTY OFFICER ASSISTANT DATABASE EXPORT")
        sql_content.append(f"-- Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        sql_content.append("-- =====================================================\n")
        
        # Get table schemas and data
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = cursor.fetchall()
        
        for table_name in tables:
            table = table_name[0]
            sql_content.append(f"\n-- ===== TABLE: {table.upper()} =====")
            
            # Get table schema
            cursor.execute(f"SELECT sql FROM sqlite_master WHERE name='{table}'")
            schema = cursor.fetchone()
            if schema:
                sql_content.append(f"-- Schema:")
                sql_content.append(schema[0] + ";")
                sql_content.append("")
            
            # Get table data count
            cursor.execute(f"SELECT COUNT(*) FROM {table}")
            count = cursor.fetchone()[0]
            sql_content.append(f"-- Records: {count}")
            
            if count > 0:
                # Get column names
                cursor.execute(f"PRAGMA table_info({table})")
                columns = [col[1] for col in cursor.fetchall()]
                
                # Get all data
                cursor.execute(f"SELECT * FROM {table}")
                rows = cursor.fetchall()
                
                sql_content.append(f"\n-- Data for {table}:")
                for i, row in enumerate(rows):
                    insert_values = []
                    for value in row:
                        if value is None:
                            insert_values.append("NULL")
                        elif isinstance(value, str):
                            escaped_value = value.replace("'", "''")
                            insert_values.append(f"'{escaped_value}'")
                        else:
                            insert_values.append(str(value))
                    
                    column_list = "(" + ", ".join(columns) + ")"
                    values_list = "(" + ", ".join(insert_values) + ")"
                    sql_content.append(f"INSERT INTO {table} {column_list}")
                    sql_content.append(f"VALUES {values_list};")
                    sql_content.append("")
            
            sql_content.append(f"-- End of {table.upper()}")
            sql_content.append("-" * 60)
        
        conn.close()
        
        # Return as plain text response
        from fastapi.responses import PlainTextResponse
        return PlainTextResponse("\n".join(sql_content), media_type="text/plain")
        
    except Exception as ex:
        logger.error(f"Error exporting SQL: {ex}")
        return PlainTextResponse(f"Error exporting database: {str(ex)}", media_type="text/plain")

@app.post("/upload-knowledge")
async def upload_knowledge_post(
    request: Request, 
    title: str = Form(...), 
    category: str = Form(""), 
    content: str = Form(...),
    db: Session = Depends(get_db)
):
    """Handle knowledge upload"""
    try:
        # Use the real knowledge base service
        knowledge_service = KnowledgeBaseService(db)
        result = knowledge_service.import_from_word_content(
            content=content,
            title=title,
            category=category if category else "General",
            source="Web Upload"
        )
        
        logger.info(f"Knowledge uploaded successfully: {title} (ID: {result.id})")
        
        # Return success response
        return templates.TemplateResponse("upload_knowledge.html", {
            "request": request,
            "success": True,
            "message": f"Knowledge document '{title}' uploaded successfully! (ID: {result.id})"
        })
        
    except Exception as ex:
        logger.error(f"Error uploading knowledge: {ex}")
        return templates.TemplateResponse("upload_knowledge.html", {
            "request": request,
            "error": True,
            "message": f"Error uploading document: {str(ex)}"
        })

@app.get("/upload-training", response_class=HTMLResponse)
async def upload_training(request: Request):
    """Upload training data page"""
    return templates.TemplateResponse("upload_training.html", {"request": request})

@app.post("/upload-training-data")
async def upload_training_data(
    request: Request,
    file: UploadFile = File(...),
    db: Session = Depends(get_db)
):
    """Handle training data upload"""
    try:
        if not file.filename.endswith(('.xlsx', '.xls')):
            return templates.TemplateResponse("upload_training.html", {
                "request": request,
                "error": True,
                "message": "Please upload an Excel file (.xlsx or .xls)"
            })
        
        # Excel/CSV upload is not supported for RCA/solution logic. Please use database import scripts for training data.
        return templates.TemplateResponse("upload_training.html", {
            "request": request,
            "error": True,
            "message": "Excel/CSV upload is not supported for RCA/solution logic. Please use database import scripts for training data."
        })
        
        # Intelligent column detection
        training_service = TrainingDataService(db)
        
        # Detect columns based on content patterns
        incident_col = None
        resolution_col = None
        
        # Look for incident-related columns
        for col in df.columns:
            col_lower = str(col).lower()
            sample_data = df[col].dropna().astype(str).str.lower()
            
            # Check if this looks like an incident column
            if any(keyword in col_lower for keyword in ['incident', 'problem', 'issue', 'description', 'summary', 'title']):
                incident_col = col
            # Check if this looks like a resolution column  
            elif any(keyword in col_lower for keyword in ['resolution', 'solution', 'fix', 'action', 'steps', 'procedure']):
                resolution_col = col
            # Content-based detection
            elif not incident_col and sample_data.str.contains('error|failed|down|issue|problem', na=False).any():
                incident_col = col
            elif not resolution_col and sample_data.str.contains('restart|check|verify|contact|replace', na=False).any():
                resolution_col = col
        
        # If no specific columns found, try first two text columns
        if not incident_col or not resolution_col:
            text_cols = [col for col in df.columns if df[col].dtype == 'object']
            if len(text_cols) >= 2:
                if not incident_col:
                    incident_col = text_cols[0]
                if not resolution_col:
                    resolution_col = text_cols[1]
            elif len(text_cols) == 1:
                incident_col = text_cols[0]
                resolution_col = text_cols[0]  # Use same column for both
        
        if not incident_col:
            return templates.TemplateResponse("upload_training.html", {
                "request": request,
                "error": True,
                "message": f"Could not identify incident column. Available columns: {list(df.columns)}"
            })
        
        # Process the data
        success_count = 0
        error_count = 0
        errors = []
        
        for index, row in df.iterrows():
            try:
                incident_text = str(row[incident_col]).strip()
                resolution_text = str(row[resolution_col]).strip() if resolution_col else ""
                
                if incident_text and incident_text.lower() not in ['nan', 'none', '']:
                    result = training_service.add_training_example(
                        incident_description=incident_text,
                        resolution_steps=resolution_text,
                        source=f"Excel Upload: {file.filename}",
                        category="Imported"
                    )
                    success_count += 1
                else:
                    error_count += 1
                    errors.append(f"Row {index + 1}: Empty incident description")
                    
            except Exception as ex:
                error_count += 1
                errors.append(f"Row {index + 1}: {str(ex)}")
        
        # Prepare result message
        message = f"Successfully imported {success_count} training examples"
        if incident_col:
            message += f" (Incident column: '{incident_col}'"
        if resolution_col and resolution_col != incident_col:
            message += f", Resolution column: '{resolution_col}'"
        if incident_col:
            message += ")"
        
        if error_count > 0:
            message += f". {error_count} errors occurred."
        
        return templates.TemplateResponse("upload_training.html", {
            "request": request,
            "success": True,
            "message": message,
            "details": {
                "success_count": success_count,
                "error_count": error_count,
                "errors": errors[:10],  # Show first 10 errors
                "incident_column": incident_col,
                "resolution_column": resolution_col,
                "total_rows": len(df)
            }
        })
        
    except Exception as ex:
        logger.error(f"Error uploading training data: {ex}")
        return templates.TemplateResponse("upload_training.html", {
            "request": request,
            "error": True,
            "message": f"Error processing file: {str(ex)}"
        })

@app.get("/view-training")
async def view_training_old(request: Request, db: Session = Depends(get_db)):
    """View training data"""
    try:
        from app.models.database import TrainingData
        training_data = db.query(TrainingData).order_by(TrainingData.created_at.desc()).limit(50).all()
        
        return templates.TemplateResponse("database_status.html", {
            "request": request,
            "training_data": training_data,
            "view_type": "training"
        })
    except Exception as ex:
        logger.error(f"Error retrieving training data: {ex}")
        return {"error": str(ex)}

@app.delete("/api/training/{training_id}")
async def delete_training(training_id: int, db: Session = Depends(get_db)):
    """Delete a training data entry"""
    try:
        from app.models.database import TrainingData
        
        # Find the training entry
        training_entry = db.query(TrainingData).filter(TrainingData.id == training_id).first()
        
        if not training_entry:
            return {"error": "Training data not found"}
        
        # Delete the entry
        db.delete(training_entry)
        db.commit()
        
        logger.info(f"Training data deleted: ID {training_id}")
        return {"message": "Training data deleted successfully"}
        
    except Exception as ex:
        logger.error(f"Error deleting training data: {ex}")
        db.rollback()
        return {"error": str(ex)}

@app.delete("/api/knowledge/{knowledge_id}")
async def delete_knowledge(knowledge_id: int, db: Session = Depends(get_db)):
    """Delete a knowledge base entry"""
    try:
        from app.models.database import KnowledgeBase
        
        # Find the knowledge entry
        knowledge_entry = db.query(KnowledgeBase).filter(KnowledgeBase.id == knowledge_id).first()
        
        if not knowledge_entry:
            return {"error": "Knowledge entry not found"}
        
        # Delete the entry
        db.delete(knowledge_entry)
        db.commit()
        
        logger.info(f"Knowledge entry deleted: ID {knowledge_id}")
        return {"message": "Knowledge entry deleted successfully"}
        
    except Exception as ex:
        logger.error(f"Error deleting knowledge entry: {ex}")
        db.rollback()
        return {"error": str(ex)}

@app.post("/api/mark-useful/{solution_type}/{solution_id}")
async def mark_solution_useful(solution_type: str, solution_id: int, db: Session = Depends(get_db)):
    """Mark a solution as useful and increment its usefulness count"""
    try:
        from app.models.database import KnowledgeBase, TrainingData, ResolutionStep
        
        if solution_type == "knowledge":
            solution = db.query(KnowledgeBase).filter(KnowledgeBase.id == solution_id).first()
        elif solution_type == "training":
            solution = db.query(TrainingData).filter(TrainingData.id == solution_id).first()
        elif solution_type == "step":
            solution = db.query(ResolutionStep).filter(ResolutionStep.id == solution_id).first()
        else:
            return {"error": "Invalid solution type"}
        
        if not solution:
            return {"error": "Solution not found"}
        
        # Increment usefulness count
        solution.usefulness_count += 1
        db.commit()
        
        logger.info(f"{solution_type.capitalize()} solution {solution_id} marked as useful. New count: {solution.usefulness_count}")
        return {"message": "Solution marked as useful", "usefulness_count": solution.usefulness_count}
        
    except Exception as ex:
        logger.error(f"Error marking solution as useful: {ex}")
        db.rollback()
        return {"error": str(ex)}

@app.post("/api/mark-step-useful")
async def mark_step_useful(
    request: Request,
    step_order: int = Form(...),
    step_description: str = Form(...),
    knowledge_base_id: Optional[int] = Form(None),
    training_data_id: Optional[int] = Form(None),
    rca_id: Optional[int] = Form(None),
    incident_description: Optional[str] = Form(None),
    step_type: Optional[str] = Form("Resolution"),
    db: Session = Depends(get_db)
):
    """Mark a specific resolution step as useful and store feedback"""
    try:
        from app.models.database import SolutionFeedback
        # Log received incident_description for debugging
        if not incident_description or incident_description.strip() == "":
            logger.warning(f"Received empty incident_description. Step: {step_description[:50]}...")
        else:
            logger.info(f"Received incident_description: {incident_description[:100]}...")
        # Determine source type
        source_type = ""
        if knowledge_base_id:
            source_type = "Knowledge Base"
        elif training_data_id:
            source_type = "Training Data"
        elif rca_id:
            source_type = "RCA History"
        # Use incident_description if provided, otherwise "Unknown incident"
        # Normalize: strip whitespace and convert to lowercase for consistency
        final_incident_description = incident_description.strip() if incident_description and incident_description.strip() else "Unknown incident"
        
        # Check if similar feedback already exists (match by solution, not incident)
        # This allows the same solution to accumulate usefulness across different incidents
        existing_feedback = db.query(SolutionFeedback).filter(
            SolutionFeedback.solution_description == step_description,
            SolutionFeedback.solution_order == step_order,
            SolutionFeedback.knowledge_base_id == knowledge_base_id,
            SolutionFeedback.training_data_id == training_data_id,
            SolutionFeedback.rca_id == rca_id
        ).first()
        if existing_feedback:
            # Increment existing count and update incident description if we have a better one
            existing_feedback.usefulness_count += 1
            existing_feedback.marked_at = datetime.utcnow()
            if final_incident_description != "Unknown incident":
                existing_feedback.incident_description = final_incident_description
            usefulness_count = existing_feedback.usefulness_count
        else:
            # Create new feedback entry
            new_feedback = SolutionFeedback(
                incident_description=final_incident_description,
                solution_description=step_description,
                solution_order=step_order,
                solution_type=step_type,
                source_type=source_type,
                knowledge_base_id=knowledge_base_id,
                training_data_id=training_data_id,
                rca_id=rca_id,
                usefulness_count=1
            )
            db.add(new_feedback)
            usefulness_count = 1
        # Also update the source table's usefulness count
        if knowledge_base_id:
            kb = db.query(KnowledgeBase).filter_by(id=knowledge_base_id).first()
            if kb:
                kb.usefulness_count += 1
        elif training_data_id:
            td = db.query(TrainingData).filter_by(id=training_data_id).first()
            if td:
                td.usefulness_count = (td.usefulness_count or 0) + 1
        db.commit()
        logger.info(f"Solution step {step_order} marked as useful. Source: {source_type}, Count: {usefulness_count}")
        return {"success": True, "usefulness_count": usefulness_count, "message": "Step marked as useful"}
    except Exception as ex:
        logger.error(f"Error marking step as useful: {ex}")
        db.rollback()
        return {"success": False, "error": str(ex)}

# ========== DATABASE STATUS ROUTES ==========

@app.get("/database-status", response_class=HTMLResponse)
async def database_status_new(request: Request, db: Session = Depends(get_db)):
    """Check database connection status"""
    
    status = {
        "ai_database": {"connected": False, "error": None, "info": {}},
        "ops_database": {"connected": False, "error": None, "info": {}}
    }
    
    # Test AI database
    try:
        from app.models.database import KnowledgeBase, TrainingData
        kb_count = db.query(KnowledgeBase).count()
        training_count = db.query(TrainingData).count()
        rca_count = db.query(RootCauseAnalysis).count()
        
        status["ai_database"]["connected"] = True
        status["ai_database"]["info"] = {
            "type": "SQLite",
            "knowledge_base_entries": kb_count,
            "training_data_entries": training_count,
            "rca_analyses": rca_count
        }
    except Exception as ex:
        status["ai_database"]["error"] = str(ex)
    
    # Test operational database (may not exist)
    try:
        from app.models.database import Vessel, Container, EDIMessage, APIEvent
        vessel_count = db.query(Vessel).count()
        container_count = db.query(Container).count()
        edi_count = db.query(EDIMessage).count()
        api_count = db.query(APIEvent).count()
        
        status["ops_database"]["connected"] = True
        status["ops_database"]["info"] = {
            "type": "SQLite",
            "vessels": vessel_count,
            "containers": container_count,
            "edi_messages": edi_count,
            "api_events": api_count
        }
    except Exception as ex:
        status["ops_database"]["error"] = str(ex)
    
    return templates.TemplateResponse("database_status.html", {
        "request": request,
        "status": status
    })

# ========== ROOT CAUSE ANALYSIS ROUTES ==========

@app.get("/rca", response_class=HTMLResponse)
async def rca_page(request: Request):
    """Root Cause Analysis page"""
    return templates.TemplateResponse("rca.html", {"request": request})

@app.post("/rca/analyze")
async def analyze_root_cause(
    request: Request,
    incident_description: str = Form(...),
    incident_start_time: str = Form(...),
    incident_end_time: str = Form(None),
    affected_systems: List[str] = Form([]),
    log_files: List[UploadFile] = File([]),
    search_window_hours: float = Form(2.0),
    include_error_patterns: bool = Form(False),
    include_warning_cascade: bool = Form(False),
    include_similar_incidents: bool = Form(False),
    include_sop: bool = Form(False),
    db: Session = Depends(get_db)
):
    """Perform root cause analysis with operational data correlation"""
    
    try:
        # Parse timestamps
        start_time = datetime.fromisoformat(incident_start_time)
        end_time = datetime.fromisoformat(incident_end_time) if incident_end_time else None
        
        # Generate incident ID
        incident_id = str(uuid.uuid4())
        
        # Initialize services
        log_analyzer = LogAnalyzerService(db)
        ops_service = OperationalDataService(db)
        
        # === NEW: OPERATIONAL DATA CORRELATION ===
        logger.info(f"🔍 Correlating incident with operational database...")
        ops_correlation = None
        try:
            ops_correlation = ops_service.correlate_incident(
                incident_description,
                start_time,
                int(search_window_hours)
            )
            logger.info(f"✅ Found {len(ops_correlation.get('findings', {}))} types of operational data")
        except Exception as ops_ex:
            logger.warning(f"⚠️ Operational database correlation failed (may not be available): {ops_ex}")
            ops_correlation = {"error": str(ops_ex)}
        
        # 1. Parse and save uploaded log files
        all_logs = []
        for log_file in log_files:
            if log_file.filename:
                content = await log_file.read()
                parsed_logs = await log_analyzer.parse_log_file(content, log_file.filename)
                all_logs.extend(parsed_logs)
                logger.info(f"Parsed {len(parsed_logs)} entries from {log_file.filename}")
        
        # Save logs to database
        total_logs_saved = 0
        if all_logs:
            total_logs_saved = log_analyzer.save_logs_to_db(all_logs, incident_id)
        
        # 2. Find logs around incident time
        relevant_logs = log_analyzer.find_logs_around_time(
            start_time, 
            window_minutes=int(search_window_hours * 60)
        )
        
        # 3. Detect error patterns
        error_patterns = []
        if include_error_patterns:
            error_patterns = log_analyzer.detect_error_patterns(relevant_logs)
        
        # 4. Detect error cascade
        error_cascade = []
        if include_warning_cascade:
            error_cascade = log_analyzer.detect_error_cascade(relevant_logs)
        
        # === AI-POWERED ROOT CAUSE GENERATION (ALWAYS RUN) ===
        # Search training data for similar incidents and use AI to generate hypotheses
        logger.info(f"🤖 Using AI to analyze incident description against 323 training examples...")
        training_service = TrainingDataService(db)
        
        # Get initial similar incidents (larger pool for filtering)
        initial_similar_incidents = await training_service.find_relevant_examples_async(
            incident_description, 
            limit=25  # Get more to filter from
        )
        
        # Enhanced filtering for more relevant incidents
        def calculate_incident_relevance(incident, target_description):
            """Calculate relevance score for incident based on multiple factors"""
            score = 0
            target_lower = target_description.lower()
            incident_lower = (incident.incident_description or "").lower()
            
            # Factor 1: Exact keyword matches (high value)
            import re
            target_keywords = set(re.findall(r'\b\w{4,}\b', target_lower))  # Words 4+ chars
            incident_keywords = set(re.findall(r'\b\w{4,}\b', incident_lower))
            keyword_overlap = len(target_keywords.intersection(incident_keywords))
            score += keyword_overlap * 10
            
            # Factor 2: Technical term matches (very high value)
            tech_terms = ['edifact', 'coarri', 'codeco', 'coprar', 'container', 'vessel', 'cntr', 'baplie']
            for term in tech_terms:
                if term in target_lower and term in incident_lower:
                    score += 50
            
            # Factor 3: Error pattern similarity
            error_patterns = ['error', 'fail', 'reject', 'invalid', 'timeout', 'duplicate', 'mismatch', 'stuck']
            matching_error_patterns = [p for p in error_patterns if p in target_lower and p in incident_lower]
            score += len(matching_error_patterns) * 15
            
            # Factor 4: Category similarity
            if hasattr(incident, 'category') and incident.category:
                category_keywords = incident.category.lower().split()
                for cat_word in category_keywords:
                    if cat_word in target_lower:
                        score += 25
            
            # Factor 5: Historical usefulness
            # Historical usefulness
            score += int(getattr(incident, 'usefulness_count', 0) or 0) * 5
            
            # Factor 6: Length similarity (prefer similar complexity)
            length_diff = abs(len(target_description) - len(incident.incident_description or ""))
            if length_diff < 100:  # Similar length incidents
                score += 20
            
            return score
        
        # Score and filter similar incidents
        scored_incidents = []
        for incident in initial_similar_incidents:
            relevance_score = calculate_incident_relevance(incident, incident_description)
            if relevance_score >= 30:  # Minimum relevance threshold
                scored_incidents.append({
                    'incident': incident,
                    'relevance_score': relevance_score
                })
        
        # Sort by relevance score and take top 5 most relevant
        scored_incidents.sort(key=lambda x: x['relevance_score'], reverse=True)
        similar_incidents = [item['incident'] for item in scored_incidents[:5]]
        
        logger.info(f"✅ Found {len(similar_incidents)} highly relevant past incidents (filtered from {len(initial_similar_incidents)} candidates)")
        if scored_incidents:
            avg_relevance = sum(item['relevance_score'] for item in scored_incidents[:5]) / min(5, len(scored_incidents))
            logger.info(f"📊 Average relevance score: {avg_relevance:.1f} (range: {scored_incidents[-1]['relevance_score'] if scored_incidents else 0}-{scored_incidents[0]['relevance_score'] if scored_incidents else 0})")
        
        # Log top 3 most relevant incidents for transparency
        logger.info("🎯 Top relevant past incidents:")
        for i, scored_item in enumerate(scored_incidents[:3], 1):
            incident = scored_item['incident']
            score = scored_item['relevance_score']
            desc_preview = (incident.incident_description or "")[:80] + "..." if len(incident.incident_description or "") > 80 else incident.incident_description or ""
            logger.info(f"   {i}. [Score: {score}] {desc_preview}")

        # === ENHANCED AI-POWERED RECOMMENDED SOLUTIONS & SOPs SEARCH ===
        logger.info(f"🔍 Performing enhanced AI-powered solution search...")
        
        # Enhanced keyword extraction from incident description
        def extract_enhanced_keywords(text):
            """Extract enhanced keywords with technical terms, error codes, and context"""
            import re
            keywords = set()
            
            # Basic word extraction (remove common words)
            common_words = {'a', 'an', 'the', 'is', 'are', 'was', 'were', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
            words = re.findall(r'\b\w+\b', text.lower())
            keywords.update([w for w in words if len(w) > 3 and w not in common_words])
            
            # Technical patterns
            keywords.update(re.findall(r'\b[A-Z]{3,}\b', text))  # Acronyms like EDI, API, SOP
            keywords.update(re.findall(r'\b\w+[-_]\w+\b', text))  # Hyphenated terms
            keywords.update(re.findall(r'\bERR[OR]*[-_]*\d+\b', text, re.IGNORECASE))  # Error codes
            keywords.update(re.findall(r'\b\w*[Ee]rror\w*\b', text))  # Error variants
            keywords.update(re.findall(r'\b\w*[Ff]ail\w*\b', text))  # Failure variants
            
            # Domain-specific terms
            edifact_terms = re.findall(r'\b(COARRI|CODECO|COPRAR|APERAK|IFTMIN|IFTSTA)\b', text, re.IGNORECASE)
            keywords.update(edifact_terms)
            
            # Container patterns
            container_terms = re.findall(r'\b(container|cntr|segment|translator|rejection)\b', text, re.IGNORECASE)
            keywords.update(container_terms)
            
            return list(keywords)
        
        enhanced_keywords = extract_enhanced_keywords(incident_description)
        logger.info(f"🔍 Extracted {len(enhanced_keywords)} enhanced keywords: {enhanced_keywords[:10]}")
        
        # Multi-stage search strategy
        kb_service = KnowledgeBaseService(db)
        
        # Stage 1: Exact phrase matching
        logger.info("🎯 Stage 1: Exact phrase matching")
        exact_matches_training = await training_service.find_relevant_examples_async(incident_description, limit=15)
        exact_matches_kb = await kb_service.find_relevant_knowledge_async(incident_description, limit=10)
        
        # Stage 2: Enhanced keyword-based search
        logger.info("🎯 Stage 2: Enhanced keyword search")
        keyword_matches_training = []
        keyword_matches_kb = []
        
        for keyword in enhanced_keywords[:8]:  # Top 8 keywords
            try:
                kw_training = await training_service.find_relevant_examples_async(keyword, limit=5)
                kw_kb = await kb_service.find_relevant_knowledge_async(keyword, limit=3)
                
                # Add unique matches only
                for match in kw_training:
                    if not any(m.id == match.id for m in keyword_matches_training):
                        keyword_matches_training.append(match)
                        
                for match in kw_kb:
                    if not any(m.id == match.id for m in keyword_matches_kb):
                        keyword_matches_kb.append(match)
            except Exception as e:
                logger.warning(f"Keyword search failed for '{keyword}': {e}")
        
        # Stage 3: Category-based fallback search
        logger.info("🎯 Stage 3: Category-based fallback")
        category_keywords = []
        if any(term in incident_description.lower() for term in ['edifact', 'edi', 'coarri', 'codeco']):
            category_keywords.extend(['EDI', 'EDIFACT', 'message', 'parsing', 'format'])
        if any(term in incident_description.lower() for term in ['container', 'cntr', 'duplicate']):
            category_keywords.extend(['Container', 'CNTR', 'duplication', 'booking'])
        if any(term in incident_description.lower() for term in ['vessel', 'ship', 'arrival', 'eta']):
            category_keywords.extend(['Vessel', 'Ship', 'arrival', 'scheduling'])
        
        category_matches_training = []
        category_matches_kb = []
        
        for cat_keyword in category_keywords:
            try:
                cat_training = await training_service.find_relevant_examples_async(cat_keyword, limit=3)
                cat_kb = await kb_service.find_relevant_knowledge_async(cat_keyword, limit=2)
                
                for match in cat_training:
                    if not any(m.id == match.id for m in category_matches_training + keyword_matches_training + exact_matches_training):
                        category_matches_training.append(match)
                        
                for match in cat_kb:
                    if not any(m.id == match.id for m in category_matches_kb + keyword_matches_kb + exact_matches_kb):
                        category_matches_kb.append(match)
            except Exception as e:
                logger.warning(f"Category search failed for '{cat_keyword}': {e}")
        
        # Combine and score all matches
        def calculate_enhanced_relevance_score(text_to_match, target_description, base_score=0):
            """Calculate enhanced relevance score with multiple factors"""
            if not text_to_match or not target_description:
                return base_score
            
            score = base_score
            text_lower = text_to_match.lower()
            target_lower = target_description.lower()
            
            # Factor 1: Exact phrase matches (high value)
            for keyword in enhanced_keywords:
                if keyword.lower() in text_lower:
                    if len(keyword) > 6:  # Longer keywords are more specific
                        score += 50
                    elif len(keyword) > 4:
                        score += 30
                    else:
                        score += 15
            
            # Factor 2: Technical term matches (very high value)
            tech_terms = ['EDIFACT', 'COARRI', 'CODECO', 'COPRAR', 'container', 'vessel', 'segment', 'translator']
            for term in tech_terms:
                if term.lower() in text_lower and term.lower() in target_lower:
                    score += 100
            
            # Factor 3: Error pattern similarity
            error_patterns = ['error', 'fail', 'reject', 'invalid', 'timeout', 'duplicate', 'mismatch']
            matching_patterns = [p for p in error_patterns if p in text_lower and p in target_lower]
            score += len(matching_patterns) * 25
            
            # Factor 4: Text similarity (simple word overlap)
            text_words = set(text_lower.split())
            target_words = set(target_lower.split())
            overlap = len(text_words.intersection(target_words))
            if len(text_words) > 0:
                similarity_ratio = overlap / len(text_words.union(target_words))
                score += int(similarity_ratio * 100)
            
            return score
        
        # Score and rank training data matches with deduplication
        all_training_matches = exact_matches_training + keyword_matches_training + category_matches_training
        
        # Deduplicate training matches by ID
        unique_training_matches = []
        seen_training_ids = set()
        for match in all_training_matches:
            if match.id not in seen_training_ids:
                unique_training_matches.append(match)
                seen_training_ids.add(match.id)
        
        scored_training = []
        for match in unique_training_matches:
            base_score = 100 + (int(getattr(match, 'usefulness_count', 0) or 0) * 10)  # Base score with usefulness
            text_to_score = f"{match.incident_description} {match.expected_root_cause} {match.notes}".strip()
            relevance_score = calculate_enhanced_relevance_score(text_to_score, incident_description, base_score)
            
            scored_training.append({
                "match": match,
                "relevance_score": relevance_score,
                "match_type": "exact" if match in exact_matches_training else ("keyword" if match in keyword_matches_training else "category")
            })
        
        # Score and rank knowledge base matches with deduplication
        all_kb_matches = exact_matches_kb + keyword_matches_kb + category_matches_kb
        
        # Deduplicate knowledge base matches by ID
        unique_kb_matches = []
        seen_kb_ids = set()
        for match in all_kb_matches:
            if match.id not in seen_kb_ids:
                unique_kb_matches.append(match)
                seen_kb_ids.add(match.id)
        
        scored_kb = []
        for match in unique_kb_matches:
            base_score = 50 + (match.usefulness_count * 5)  # Lower base score for templates
            text_to_score = f"{match.title} {match.content}".strip()
            relevance_score = calculate_enhanced_relevance_score(text_to_score, incident_description, base_score)
            
            scored_kb.append({
                "match": match,
                "relevance_score": relevance_score,
                "match_type": "exact" if match in exact_matches_kb else ("keyword" if match in keyword_matches_kb else "category")
            })
        
        # Sort by relevance score (highest first)
        scored_training.sort(key=lambda x: x["relevance_score"], reverse=True)
        scored_kb.sort(key=lambda x: x["relevance_score"], reverse=True)
        
        # Take top 10 training and top 5 KB matches
        top_training = scored_training[:10]
        top_kb = scored_kb[:5]
        
        logger.info(f"✅ Enhanced search complete: {len(top_training)} training solutions, {len(top_kb)} KB solutions")
        logger.info(f"🔄 Deduplication: Training {len(all_training_matches)}→{len(unique_training_matches)}, KB {len(all_kb_matches)}→{len(unique_kb_matches)}")
        
        # Log KB match details for debugging
        logger.info("🔍 KB matches found:")
        for i, scored_match in enumerate(top_kb[:3], 1):
            match = scored_match["match"]
            logger.info(f"   {i}. ID:{match.id} Score:{scored_match['relevance_score']:.1f} Title:'{match.title[:50]}...'")
        
        # Additional check for final SOP uniqueness
        final_sop_titles = []
        unique_top_kb = []
        for scored_match in top_kb:
            match = scored_match["match"]
            if match.title not in final_sop_titles:
                final_sop_titles.append(match.title)
                unique_top_kb.append(scored_match)
        
        if len(unique_top_kb) < len(top_kb):
            logger.info(f"⚠️  Further SOP deduplication: {len(top_kb)}→{len(unique_top_kb)} (removed duplicate titles)")
            top_kb = unique_top_kb
        
        # Format for database storage and template display
        # === ENRICH SOLUTIONS WITH FEEDBACK ===
        from app.models.database import SolutionFeedback
        feedback_rows = db.query(SolutionFeedback).filter(SolutionFeedback.usefulness_count > 0).all()
        recommended_solutions_data = []
        for idx, scored_match in enumerate(top_training, 1):
            match = scored_match["match"]
            solution = match.expected_root_cause or match.incident_description or "See training data"
            # Feedback matching (bidirectional, case-insensitive)
            user_verified = False
            usefulness_count = int(getattr(match, 'usefulness_count', 0) or 0)
            feedback_incident = None
            feedback_solution = None
            desc = solution
            for fb in feedback_rows:
                if desc and fb.solution_description and (
                    desc.lower() in fb.solution_description.lower() or fb.solution_description.lower() in desc.lower()
                ):
                    user_verified = True
                    usefulness_count = fb.usefulness_count
                    feedback_incident = fb.incident_description
                    feedback_solution = fb.solution_description
                    logger.info(f"[RCA] User verified match: Solution '{desc[:50]}' <-> Feedback '{fb.solution_description[:50]}' (count={fb.usefulness_count})")
                    break
            solution_obj = {
                "order": idx,
                "title": f"Solution #{idx}: {match.category}" if match.category else f"Solution #{idx}",
                "description": solution[:500] + "..." if len(solution) > 500 else solution,
                "type": "Training Case",
                "source": f"Training Data (Case #{match.id})",
                "category": match.category or "General",
                "relevance_score": min(99, max(60, int(scored_match["relevance_score"] / 10))),  # Normalize to 60-99%
                "match_type": scored_match["match_type"],
                "usefulness_count": usefulness_count,
                "user_verified": user_verified,
                "feedback_incident_description": feedback_incident,
                "feedback_solution_description": feedback_solution
            }
            recommended_solutions_data.append(solution_obj)

        # === DEDUPLICATE: Remove solutions with identical descriptions ===
        unique_solutions = []
        seen_descriptions = set()
        for solution in recommended_solutions_data:
            desc_lower = solution['description'].lower().strip()
            if desc_lower not in seen_descriptions:
                unique_solutions.append(solution)
                seen_descriptions.add(desc_lower)
            else:
                logger.info(f"[RCA] Removed duplicate solution: '{solution['description'][:50]}...'")
        
        logger.info(f"[RCA] Deduplication: {len(recommended_solutions_data)} → {len(unique_solutions)} solutions")
        recommended_solutions_data = unique_solutions

        # === SORT: User-verified solutions first (by usefulness_count desc), then unverified ===
        recommended_solutions_data.sort(key=lambda s: (not s['user_verified'], -s['usefulness_count']))
        
        # Re-assign order numbers after sorting
        for idx, solution in enumerate(recommended_solutions_data, 1):
            solution['order'] = idx
        
        logger.info(f"[RCA] Solutions sorted: User-verified solutions at top, sorted by usefulness_count")
        
        sop_references_data = []
        for idx, scored_match in enumerate(top_kb, 1):
            match = scored_match["match"]
            content_preview = match.content[:300] + "..." if len(match.content) > 300 else match.content
            
            # Ensure unique title formatting
            sop_title = match.title
            if sop_title in [existing["title"] for existing in sop_references_data]:
                sop_title = f"{match.title} (#{match.id})"
            
            sop_references_data.append({
                "order": idx,
                "title": sop_title,
                "description": match.content[:200] + "..." if len(match.content) > 200 else match.content,
                "content_preview": content_preview,
                "relevance_score": min(95, max(70, int(scored_match["relevance_score"] / 8))),  # Normalize to 70-95%
                "match_type": scored_match["match_type"],
                "usefulness_count": match.usefulness_count,
                "category": match.category or "SOP",
                "kb_id": match.id  # Add ID for debugging
            })
        
        logger.info(f"🎯 Solution accuracy enhanced: {len(recommended_solutions_data)} solutions with avg relevance {sum(s['relevance_score'] for s in recommended_solutions_data) / len(recommended_solutions_data) if recommended_solutions_data else 0:.1f}%")
        logger.info(f"📋 SOP references enhanced: {len(sop_references_data)} SOPs with avg relevance {sum(s['relevance_score'] for s in sop_references_data) / len(sop_references_data) if sop_references_data else 0:.1f}%")
        
        # Debug output for troubleshooting
        if recommended_solutions_data:
            logger.info(f"✅ Sample solution: {recommended_solutions_data[0]['title']} (Score: {recommended_solutions_data[0]['relevance_score']}%)")
        else:
            logger.warning("⚠️  No recommended solutions found!")
            
        if sop_references_data:
            logger.info(f"✅ Sample SOP: {sop_references_data[0]['title']} (Score: {sop_references_data[0]['relevance_score']}%)")
        else:
            logger.warning("⚠️  No SOP references found!")
            
        # === FALLBACK: Ensure we always have some solutions for testing ===
        if not recommended_solutions_data:
            logger.info("🔄 Creating fallback solutions for display...")
            recommended_solutions_data = [
                {
                    "order": 1,
                    "title": "Fallback Solution #1: Check System Logs",
                    "description": "Review system logs around the incident time to identify error patterns and root causes.",
                    "type": "General Troubleshooting",
                    "source": "Fallback Guidance",
                    "category": "General",
                    "relevance_score": 75,
                    "match_type": "fallback",
                    "usefulness_count": 0
                },
                {
                    "order": 2,
                    "title": "Fallback Solution #2: Contact Support Team",
                    "description": "Escalate to technical support team with incident details and timeline information.",
                    "type": "Escalation Procedure",
                    "source": "Fallback Guidance",
                    "category": "Support",
                    "relevance_score": 70,
                    "match_type": "fallback",
                    "usefulness_count": 0
                }
            ]
            
        if not sop_references_data:
            logger.info("🔄 Creating fallback SOPs for display...")
            sop_references_data = [
                {
                    "order": 1,
                    "title": "Standard Incident Response Procedure",
                    "description": "Follow standard incident response protocol including documentation, analysis, and resolution steps.",
                    "content_preview": "1. Document incident details\n2. Analyze system logs\n3. Identify root cause\n4. Implement solution\n5. Monitor resolution\n6. Update documentation",
                    "relevance_score": 80,
                    "match_type": "fallback",
                    "usefulness_count": 0,
                    "category": "Standard Procedure",
                    "kb_id": "fallback-1"
                }
            ]
        
        # Generate AI-powered root cause hypotheses with enhanced context
        hypotheses = []
        if similar_incidents:
            # Use the most similar incident's solution as primary hypothesis
            most_similar = similar_incidents[0]
            from app.services.log_analyzer_service import RootCauseHypothesis
            
            # Enhanced hypothesis with solution context
            primary_description = most_similar.expected_root_cause if most_similar.expected_root_cause else "Root cause identified from similar incidents"
            
            # Add solution context if available
            solution_context = ""
            if recommended_solutions_data:
                top_solution = recommended_solutions_data[0]
                solution_context = f" Recommended solution: {top_solution['description'][:100]}..."
            
            hypotheses.append(RootCauseHypothesis(
                description=primary_description + solution_context,
                confidence=0.85,
                evidence=[
                    f"Similar incident found: {most_similar.incident_description[:150]}...",
                    f"Category: {most_similar.category}",
                    f"Based on {len(similar_incidents)} similar past incidents",
                    f"Enhanced with {len(recommended_solutions_data)} AI-matched solutions"
                ],
                contributing_factors=[
                    most_similar.expected_root_cause[:200] if most_similar.expected_root_cause else "See similar incidents for details",
                    f"SOP references: {len(sop_references_data)} relevant procedures found"
                ]
            ))
        
        # Also generate from logs if available
        if relevant_logs:
            log_hypotheses = log_analyzer.extract_root_cause_candidates(
                relevant_logs, 
                incident_description
            )
            hypotheses.extend(log_hypotheses)
        
        # If no hypotheses from AI or logs, create a generic one
        if not hypotheses:
            from app.services.log_analyzer_service import RootCauseHypothesis
            hypotheses.append(RootCauseHypothesis(
                description="Unable to determine root cause - insufficient data",
                confidence=0.0,
                evidence=["No similar incidents found", "No log files uploaded"],
                contributing_factors=["Please provide more details or upload log files"]
            ))
        
        # === NEW: ENHANCE HYPOTHESES WITH OPERATIONAL DATA ===
        if ops_correlation and "findings" in ops_correlation:
            findings = ops_correlation["findings"]
            
            # Enhance root cause with container findings
            if "containers" in findings:
                for container in findings["containers"]:
                    if container.get("duplication_analysis", {}).get("has_duplicates"):
                        dup = container["duplication_analysis"]
                        enhanced_hypothesis = f"Container {container['cntr_no']} duplication detected: {dup['issue_type']}. "
                        if "root_cause" in dup:
                            enhanced_hypothesis += dup["root_cause"]
                        
                        from app.services.log_analyzer_service import RootCauseHypothesis
                        hypotheses.insert(0, RootCauseHypothesis(
                            description=enhanced_hypothesis,
                            confidence=0.95,
                            evidence=[f"Database shows {dup['count']} records for {container['cntr_no']}"],
                            contributing_factors=["Composite primary key (cntr_no, created_at)", "Possible race condition or double-submit"]
                        ))
            
            # Enhance root cause with vessel findings
            if "vessels" in findings:
                for vessel in findings["vessels"]:
                    advice = vessel.get("advice_conflict", {})
                    if advice.get("has_conflict") and advice.get("error_type") == "VESSEL_ERR_4":
                        enhanced_hypothesis = f"VESSEL_ERR_4: {advice['root_cause']}"
                        
                        from app.services.log_analyzer_service import RootCauseHypothesis
                        hypotheses.insert(0, RootCauseHypothesis(
                            description=enhanced_hypothesis,
                            confidence=0.98,
                            evidence=[
                                f"Active vessel advice #{advice['active_advice_no']} exists since {advice['active_since']}",
                                f"Unique constraint prevents multiple active advices for same vessel name"
                            ],
                            contributing_factors=[
                                "Vessel advice lifecycle not properly managed",
                                f"Solution: {advice['solution']}"
                            ]
                        ))
            
            # Enhance with EDI error findings
            if "edi_messages" in findings:
                for edi in findings["edi_messages"]:
                    if edi.get("root_cause"):
                        from app.services.log_analyzer_service import RootCauseHypothesis
                        hypotheses.insert(0, RootCauseHypothesis(
                            description=f"EDI {edi['type']} error: {edi['root_cause']}",
                            confidence=0.90,
                            evidence=[f"Message {edi['message_ref']}: {edi['error_text']}"],
                            contributing_factors=[edi.get("solution", "Review EDI message structure")]
                        ))
        
        
        # 7. Search SOPs from knowledge base (always enabled for better solutions)
        relevant_sops = []
        if include_sop:
            kb_service = KnowledgeBaseService(db)
            relevant_sops = kb_service.search_knowledge(incident_description)
        
        # 8. Build timeline
        timeline = log_analyzer.build_timeline(relevant_logs, start_time, end_time)
        
        # 9. Save RCA to database
        root_cause = hypotheses[0].description if hypotheses else "Unable to determine root cause"
        confidence = hypotheses[0].confidence if hypotheses else 0.0
        
        # Add operational data summary to evidence
        evidence_list = [h.evidence for h in hypotheses[:1]] if hypotheses else []
        if ops_correlation and "findings" in ops_correlation:
            evidence_list.append([f"Operational Data: {len(ops_correlation['findings'])} data source(s) analyzed"])
        
        rca = RootCauseAnalysis(
            incident_id=incident_id,
            incident_description=incident_description,
            incident_start_time=start_time,
            incident_end_time=end_time,
            affected_systems=affected_systems,
            root_cause=root_cause,
            confidence_score=confidence,
            evidence=evidence_list,
            contributing_factors=[h.contributing_factors for h in hypotheses[:1]] if hypotheses else [],
            error_cascade=error_cascade,
            similar_incidents=[{"id": s.id, "description": s.incident_description[:100]} for s in similar_incidents],
            recommended_solutions=recommended_solutions_data,  # Use enhanced AI solutions
            sop_references=sop_references_data,  # Use enhanced SOP references
            timeline=timeline,
            search_window_hours=int(search_window_hours),
            total_logs_analyzed=len(relevant_logs),
            status="Completed"
        )
        db.add(rca)
        db.commit()
        db.refresh(rca)
        
        logger.info(f"RCA completed for incident {incident_id}: {root_cause}")
        
        # 10. Return results (enhanced with operational data and AI solutions)
        return templates.TemplateResponse("rca_results.html", {
            "request": request,
            "rca": rca,
            "hypotheses": hypotheses,
            "timeline": timeline,
            "error_patterns": error_patterns,
            "error_cascade": error_cascade,
            "similar_incidents": similar_incidents,
            "recommended_solutions": recommended_solutions_data,  # Enhanced AI solutions
            "sop_references": sop_references_data,  # Enhanced SOP references  
            "log_evidence": relevant_logs[:20],  # Top 20 relevant logs
            "total_logs_uploaded": len(all_logs),
            "total_logs_analyzed": len(relevant_logs),
            "ops_correlation": ops_correlation  # NEW: Pass operational data to template
        })
        
    except Exception as ex:
        import traceback
        logger.error(f"RCA error: {ex}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return RedirectResponse(url=f"/rca?error={str(ex)}", status_code=302)

@app.get("/rca/history", response_class=HTMLResponse)
async def rca_history(
    request: Request, 
    db: Session = Depends(get_db),
    status: str = None,
    resolution: str = None,
    confidence: str = None,
    date_from: str = None,
    date_to: str = None,
    page: int = 1,
    per_page: int = 20
):
    """View RCA history with filters and pagination"""
    try:
        from datetime import datetime, timedelta
        
        # Build query with filters
        query = db.query(RootCauseAnalysis)
        
        # Status filter
        if status:
            query = query.filter(RootCauseAnalysis.status == status)
        
        # Resolution filter
        if resolution:
            query = query.filter(RootCauseAnalysis.resolution_status == resolution)
        
        # Confidence filter
        if confidence:
            if confidence == "high":
                query = query.filter(RootCauseAnalysis.confidence_score >= 0.7)
            elif confidence == "medium":
                query = query.filter(
                    RootCauseAnalysis.confidence_score >= 0.4,
                    RootCauseAnalysis.confidence_score < 0.7
                )
            elif confidence == "low":
                query = query.filter(RootCauseAnalysis.confidence_score < 0.4)
        
        # Date range filter
        if date_from:
            date_from_obj = datetime.strptime(date_from, "%Y-%m-%d")
            query = query.filter(RootCauseAnalysis.analyzed_at >= date_from_obj)
        
        if date_to:
            date_to_obj = datetime.strptime(date_to, "%Y-%m-%d")
            date_to_obj = date_to_obj.replace(hour=23, minute=59, second=59)
            query = query.filter(RootCauseAnalysis.analyzed_at <= date_to_obj)
        
        # Count total for pagination
        total_count = query.count()
        total_pages = (total_count + per_page - 1) // per_page
        
        # Get paginated results
        analyses = query.order_by(RootCauseAnalysis.analyzed_at.desc())\
                       .offset((page - 1) * per_page)\
                       .limit(per_page)\
                       .all()
        
        # Calculate statistics
        all_analyses = db.query(RootCauseAnalysis).all()
        high_confidence_count = len([a for a in all_analyses if a.confidence_score >= 0.7])
        open_count = len([a for a in all_analyses if a.resolution_status == "Open"])
        
        # This week count
        week_ago = datetime.now() - timedelta(days=7)
        this_week_count = db.query(RootCauseAnalysis).filter(
            RootCauseAnalysis.analyzed_at >= week_ago
        ).count()
        
        return templates.TemplateResponse("rca_history.html", {
            "request": request,
            "analyses": analyses,
            "total_count": len(all_analyses),
            "high_confidence_count": high_confidence_count,
            "open_count": open_count,
            "this_week_count": this_week_count,
            "current_page": page,
            "total_pages": total_pages,
            "per_page": per_page
        })
        
    except Exception as ex:
        import traceback
        logger.error(f"Error retrieving RCA history: {ex}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return templates.TemplateResponse("rca_history.html", {
            "request": request,
            "analyses": [],
            "total_count": 0,
            "high_confidence_count": 0,
            "open_count": 0,
            "this_week_count": 0,
            "current_page": 1,
            "total_pages": 1,
            "per_page": per_page,
            "error": str(ex)
        })

@app.get("/rca/{rca_id}", response_class=HTMLResponse)
async def view_rca(request: Request, rca_id: int, db: Session = Depends(get_db)):
    """View specific RCA details"""
    try:
        rca = db.query(RootCauseAnalysis).filter(RootCauseAnalysis.id == rca_id).first()
        
        if not rca:
            return RedirectResponse(url="/rca/history?error=RCA not found", status_code=302)
        
        # Get related logs
        logs = db.query(SystemLog).filter(SystemLog.incident_id == rca.incident_id).all()
        
        return templates.TemplateResponse("rca_results.html", {
            "request": request,
            "rca": rca,
            "log_evidence": logs[:20],
            "from_history": True
        })
        
    except Exception as ex:
        logger.error(f"Error retrieving RCA {rca_id}: {ex}")
        return RedirectResponse(url=f"/rca/history?error={str(ex)}", status_code=302)

@app.get("/rca/{rca_id}/export")
async def export_rca(rca_id: int, db: Session = Depends(get_db)):
    """Export RCA as JSON"""
    try:
        from fastapi.responses import JSONResponse
        
        rca = db.query(RootCauseAnalysis).filter(RootCauseAnalysis.id == rca_id).first()
        
        if not rca:
            return JSONResponse(
                status_code=404,
                content={"error": "RCA not found"}
            )
        
        # Convert to dict using the model's to_dict method
        rca_data = rca.to_dict()
        
        # Create response with download header
        return JSONResponse(
            content=rca_data,
            headers={
                "Content-Disposition": f"attachment; filename=rca_{rca_id}_{rca.incident_id}.json"
            }
        )
        
    except Exception as ex:
        logger.error(f"Error exporting RCA {rca_id}: {ex}")
        return JSONResponse(
            status_code=500,
            content={"error": str(ex)}
        )

@app.delete("/rca/{rca_id}")
async def delete_rca(rca_id: int, db: Session = Depends(get_db)):
    """Delete an RCA"""
    try:
        from fastapi.responses import JSONResponse
        
        rca = db.query(RootCauseAnalysis).filter(RootCauseAnalysis.id == rca_id).first()
        
        if not rca:
            return JSONResponse(
                status_code=404,
                content={"error": "RCA not found"}
            )
        
        # Store incident_id for response
        incident_id = rca.incident_id
        
        # Delete the RCA
        db.delete(rca)
        db.commit()
        
        return JSONResponse(
            content={
                "success": True,
                "message": f"RCA for incident {incident_id} deleted successfully"
            }
        )
        
    except Exception as ex:
        db.rollback()
        logger.error(f"Error deleting RCA {rca_id}: {ex}")
        return JSONResponse(
            status_code=500,
            content={"error": str(ex)}
        )

# ========== ESCALATION API ENDPOINTS ==========

@app.post("/api/generate-escalation-summary")
async def generate_escalation_summary_api(
    incident_description: str = Form(...),
    incident_type: str = Form("System Issue"),
    urgency: str = Form("Medium"),
    affected_systems: str = Form(""),  # Comma-separated string
    solutions_count: int = Form(0),
    db: Session = Depends(get_db)
) -> JSONResponse:
    """API endpoint to generate escalation summary independently"""
    try:
        # Create incident object
        from app.models.schemas import Incident, IncidentAnalysis
        
        incident = Incident(
            description=incident_description,
            title=f"Incident - {incident_type}",
            category=incident_type
        )
        
        # Create analysis object
        affected_systems_list = [s.strip() for s in affected_systems.split(",") if s.strip()] if affected_systems else []
        analysis = IncidentAnalysis(
            incident_type=incident_type,
            urgency=urgency,
            affected_systems=affected_systems_list,
            root_cause="Generated via API call",
            pattern_match=f"API-based analysis for {incident_type}"
        )
        
        # Generate escalation summary
        escalation_service = EscalationService()
        escalation_summary = escalation_service.generate_escalation_summary(
            incident=incident,
            analysis=analysis,
            solutions_count=solutions_count
        )
        
        escalation_templates = escalation_service.generate_escalation_templates(
            incident=incident,
            summary=escalation_summary
        )
        
        return JSONResponse(content={
            "success": True,
            "escalation_summary": escalation_summary.dict(),
            "escalation_templates": escalation_templates.dict()
        })
        
    except Exception as ex:
        logger.error(f"Error generating escalation summary: {ex}")
        return JSONResponse(
            status_code=500,
            content={"error": str(ex)}
        )

@app.get("/escalation-generator")
async def escalation_generator_page(request: Request):
    """Standalone escalation summary generator page"""
    return templates.TemplateResponse("escalation_generator.html", {"request": request})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="localhost", port=8002)