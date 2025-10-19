from fastapi import FastAPI, Request, Form, Depends, UploadFile, File
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse, RedirectResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from sqlalchemy.orm import Session
from typing import List, Dict, Any
from starlette.middleware.sessions import SessionMiddleware
import logging
from datetime import datetime
import uuid
from dotenv import load_dotenv
import pandas as pd
import io
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
from app.models.database import Base, ResolutionStep, SystemLog, RootCauseAnalysis
from app.database import get_db, engine

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(title="AI Duty Officer Assistant", version="1.0.0")

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
    test_cases = [
        {
            "description": "Customer on PORTNET is seeing 2 identical containers information for CMAU0000020",
            "source": "Email",
            "priority": "Medium", 
            "title": "Container Duplication Issue",
            "icon": "fas fa-box",
            "category": "Container Management"
        },
        {
            "description": "VESSEL_ERR_4 when creating vessel advice for MV Lion City 07",
            "source": "Email",
            "priority": "High",
            "title": "Vessel Operations Error", 
            "icon": "fas fa-ship",
            "category": "Vessel Operations"
        },
        {
            "description": "EDI message REF-IFT-0007 stuck in ERROR status, ack_at is NULL",
            "source": "SMS",
            "priority": "High",
            "title": "EDI Processing Failure",
            "icon": "fas fa-exchange-alt",
            "category": "Data Integration"
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
        # Use new OpenAIService query flow: extract error type, search KB and training, return all matches sorted by usefulness
        resolution_data = await openai_service.generate_resolution_plan_async(
            combined_description, analysis=None, knowledge_entries=None, training_examples=None, db=db)

        all_solutions = resolution_data.get('steps', [])
        total_count = len(all_solutions)
        
        logger.info(f"[Query Flow] Found {total_count} matching solutions")
        
        # For initial page load, return only first 15 solutions
        initial_limit = 15
        initial_solutions = all_solutions[:initial_limit]
        
        # Reassign order numbers starting from 1
        for idx, solution in enumerate(initial_solutions, 1):
            solution['order'] = idx
        
        if initial_solutions:
            logger.info(f"Top solution: {initial_solutions[0]}")

        # Store full results in session/cache for lazy loading (using incident ID as key)
        # For now, we'll pass incident_id to frontend and use it to fetch more
        import json
        
        # Prepare view model for results.html
        class SolutionViewModel:
            def __init__(self, incident, resolution_data, uploaded_images=None, total_count=0, initial_limit=15):
                self.incident = incident
                self.summary = resolution_data.get("summary", "")
                self.solutions = resolution_data.get("steps", [])[:initial_limit]  # Only initial batch
                self.uploaded_images = uploaded_images or []
                self.total_count = total_count
                self.loaded_count = min(initial_limit, total_count)
                self.has_more = total_count > initial_limit

        view_model = SolutionViewModel(incident, resolution_data, uploaded_images, total_count, initial_limit)
        
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

@app.get("/test-case")
async def test_case(request: Request, description: str = ""):
    """Test case with preloaded description"""
    return templates.TemplateResponse("analyze.html", {
        "request": request,
        "test_cases": [],
        "preloaded_description": description
    })

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

@app.get("/database-status")
async def database_status(request: Request, db: Session = Depends(get_db)):
    """View database status and contents"""
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
        
        # Read Excel file
        content = await file.read()
        df = pd.read_excel(io.BytesIO(content))
        
        if df.empty:
            return templates.TemplateResponse("upload_training.html", {
                "request": request,
                "error": True,
                "message": "Excel file is empty"
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
    incident_id: str = Form(...),
    step_order: int = Form(...),
    step_description: str = Form(...),
    db: Session = Depends(get_db)
):
    """Mark a specific resolution step as useful"""
    try:
        # Check if step already exists in DB
        db_step = db.query(ResolutionStep).filter_by(
            incident_id=incident_id,
            order=step_order,
            description=step_description
        ).first()
        
        if db_step:
            # Increment existing count
            db_step.usefulness_count += 1
        else:
            # Create new step entry
            db_step = ResolutionStep(
                incident_id=incident_id,
                order=step_order,
                description=step_description,
                usefulness_count=1
            )
            db.add(db_step)
        
        db.commit()
        logger.info(f"Step {step_order} for incident {incident_id} marked as useful. Count: {db_step.usefulness_count}")
        
        return {"success": True, "usefulness_count": db_step.usefulness_count, "message": "Step marked as useful"}
        
    except Exception as ex:
        logger.error(f"Error marking step as useful: {ex}")
        db.rollback()
        return {"success": False, "error": str(ex)}

# ========== DATABASE STATUS ROUTES ==========

@app.get("/database-status", response_class=HTMLResponse)
async def database_status(request: Request, db: Session = Depends(get_db)):
    """Check database connection status"""
    
    status = {
        "database": {"connected": False, "error": None, "info": {}}
    }
    
    # Test database
    try:
        from app.models.database import KnowledgeBase, TrainingData, Vessel, Container, EDIMessage, APIEvent
        kb_count = db.query(KnowledgeBase).count()
        training_count = db.query(TrainingData).count()
        rca_count = db.query(RootCauseAnalysis).count()
        vessel_count = db.query(Vessel).count()
        container_count = db.query(Container).count()
        edi_count = db.query(EDIMessage).count()
        api_count = db.query(APIEvent).count()
        
        status["database"]["connected"] = True
        status["database"]["info"] = {
            "type": "SQLite",
            "knowledge_base_entries": kb_count,
            "training_data_entries": training_count,
            "rca_analyses": rca_count,
            "vessels": vessel_count,
            "containers": container_count,
            "edi_messages": edi_count,
            "api_events": api_count
        }
    except Exception as ex:
        status["database"]["error"] = str(ex)
    
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
        similar_incidents = await training_service.find_relevant_examples_async(
            incident_description, 
            limit=10
        )
        logger.info(f"✅ Found {len(similar_incidents)} similar past incidents")
        
        # Generate AI-powered root cause hypotheses
        hypotheses = []
        if similar_incidents:
            # Use the most similar incident's solution as primary hypothesis
            most_similar = similar_incidents[0]
            from app.services.log_analyzer_service import RootCauseHypothesis
            
            hypotheses.append(RootCauseHypothesis(
                description=most_similar.expected_root_cause if most_similar.expected_root_cause else "Root cause identified from similar incidents",
                confidence=0.85,
                evidence=[
                    f"Similar incident found: {most_similar.incident_description[:150]}...",
                    f"Category: {most_similar.category}",
                    f"Based on {len(similar_incidents)} similar past incidents"
                ],
                contributing_factors=[
                    most_similar.expected_root_cause[:200] if most_similar.expected_root_cause else "See similar incidents for details"
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
            recommended_solutions=[{"id": sop.id, "title": sop.title, "content": sop.content[:200]} for sop in relevant_sops[:5]],
            sop_references=[{"id": sop.id, "title": sop.title} for sop in relevant_sops],
            timeline=timeline,
            search_window_hours=int(search_window_hours),
            total_logs_analyzed=len(relevant_logs),
            status="Completed"
        )
        db.add(rca)
        db.commit()
        db.refresh(rca)
        
        logger.info(f"RCA completed for incident {incident_id}: {root_cause}")
        
        # 10. Return results (enhanced with operational data)
        return templates.TemplateResponse("rca_results.html", {
            "request": request,
            "rca": rca,
            "hypotheses": hypotheses,
            "timeline": timeline,
            "error_patterns": error_patterns,
            "error_cascade": error_cascade,
            "similar_incidents": similar_incidents,
            "recommended_solutions": relevant_sops[:5],
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

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="localhost", port=8002)