from sqlalchemy import Column, Integer, String, DateTime, Text, Float, ForeignKey, JSON, Enum, DECIMAL, TIMESTAMP, Integer, SmallInteger, Date
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any
import json
import enum

Base = declarative_base()

# ============================================================================
# OPERATIONAL DATABASE MODELS (MySQL - PORTNET Domain)
# ============================================================================

class Vessel(Base):
    """Vessel reference data from operational database"""
    __tablename__ = "vessel"
    __table_args__ = {'extend_existing': True}
    
    vessel_id = Column(Integer, primary_key=True, autoincrement=True)
    imo_no = Column(Integer, unique=True, nullable=False, index=True)
    vessel_name = Column(String(100), nullable=False)
    call_sign = Column(String(20))
    operator_name = Column(String(100))
    flag_state = Column(String(50))
    built_year = Column(SmallInteger)
    capacity_teu = Column(Integer)
    loa_m = Column(DECIMAL(6, 2))
    beam_m = Column(DECIMAL(5, 2))
    draft_m = Column(DECIMAL(4, 2))
    last_port = Column(String(5))  # UN/LOCODE
    next_port = Column(String(5))  # UN/LOCODE
    created_at = Column(TIMESTAMP, nullable=False, server_default=func.current_timestamp())
    
    # Relationships
    containers = relationship("Container", back_populates="vessel")
    edi_messages = relationship("EDIMessage", back_populates="vessel")
    api_events = relationship("APIEvent", back_populates="vessel")


class ContainerStatusEnum(str, enum.Enum):
    IN_YARD = "IN_YARD"
    ON_VESSEL = "ON_VESSEL"
    GATE_OUT = "GATE_OUT"
    GATE_IN = "GATE_IN"
    DISCHARGED = "DISCHARGED"
    LOADED = "LOADED"
    TRANSHIP = "TRANSHIP"


class Container(Base):
    """Container instances with versioned snapshots"""
    __tablename__ = "container"
    __table_args__ = {'extend_existing': True}
    
    container_id = Column(Integer, unique=True, autoincrement=True, index=True)
    cntr_no = Column(String(11), primary_key=True, nullable=False)
    iso_code = Column(String(4), nullable=False)
    size_type = Column(String(10), nullable=False)
    gross_weight_kg = Column(DECIMAL(10, 2))
    status = Column(Enum(ContainerStatusEnum), nullable=False)
    origin_port = Column(String(5), nullable=False)
    tranship_port = Column(String(5), nullable=False, default='SGSIN')
    destination_port = Column(String(5), nullable=False)
    hazard_class = Column(String(10))
    vessel_id = Column(Integer, ForeignKey("vessel.vessel_id", onupdate="CASCADE", ondelete="SET NULL"))
    eta_ts = Column(DateTime)
    etd_ts = Column(DateTime)
    last_free_day = Column(Date)
    created_at = Column(TIMESTAMP, primary_key=True, nullable=False, server_default=func.current_timestamp())
    
    # Relationships
    vessel = relationship("Vessel", back_populates="containers")
    edi_messages = relationship("EDIMessage", back_populates="container")
    api_events = relationship("APIEvent", back_populates="container")


class EDIMessageTypeEnum(str, enum.Enum):
    COPARN = "COPARN"
    COARRI = "COARRI"
    CODECO = "CODECO"
    IFTMCS = "IFTMCS"
    IFTMIN = "IFTMIN"


class EDIDirectionEnum(str, enum.Enum):
    IN = "IN"
    OUT = "OUT"


class EDIStatusEnum(str, enum.Enum):
    RECEIVED = "RECEIVED"
    PARSED = "PARSED"
    ACKED = "ACKED"
    ERROR = "ERROR"


class EDIMessage(Base):
    """EDI messages (COPARN, COARRI, CODECO, etc.)"""
    __tablename__ = "edi_message"
    __table_args__ = {'extend_existing': True}
    
    edi_id = Column(Integer, primary_key=True, autoincrement=True)
    container_id = Column(Integer, ForeignKey("container.container_id", onupdate="CASCADE", ondelete="SET NULL"), index=True)
    vessel_id = Column(Integer, ForeignKey("vessel.vessel_id", onupdate="CASCADE", ondelete="SET NULL"), index=True)
    message_type = Column(Enum(EDIMessageTypeEnum), nullable=False, index=True)
    direction = Column(Enum(EDIDirectionEnum), nullable=False)
    status = Column(Enum(EDIStatusEnum), nullable=False, default=EDIStatusEnum.RECEIVED)
    message_ref = Column(String(50), nullable=False)
    sender = Column(String(100), nullable=False)
    receiver = Column(String(100), nullable=False)
    sent_at = Column(DateTime, nullable=False, index=True)
    ack_at = Column(DateTime)
    error_text = Column(String(500))
    raw_text = Column(Text)
    created_at = Column(TIMESTAMP, nullable=False, server_default=func.current_timestamp())
    
    # Relationships
    container = relationship("Container", back_populates="edi_messages")
    vessel = relationship("Vessel", back_populates="edi_messages")


class APIEventTypeEnum(str, enum.Enum):
    GATE_IN = "GATE_IN"
    GATE_OUT = "GATE_OUT"
    LOAD = "LOAD"
    DISCHARGE = "DISCHARGE"
    CUSTOMS_CLEAR = "CUSTOMS_CLEAR"
    HOLD = "HOLD"
    RELEASE = "RELEASE"


class APIEvent(Base):
    """API-sourced operational events"""
    __tablename__ = "api_event"
    __table_args__ = {'extend_existing': True}
    
    api_id = Column(Integer, primary_key=True, autoincrement=True)
    container_id = Column(Integer, ForeignKey("container.container_id", onupdate="CASCADE", ondelete="SET NULL"), index=True)
    vessel_id = Column(Integer, ForeignKey("vessel.vessel_id", onupdate="CASCADE", ondelete="SET NULL"))
    event_type = Column(Enum(APIEventTypeEnum), nullable=False, index=True)
    source_system = Column(String(50), nullable=False)
    http_status = Column(SmallInteger)
    correlation_id = Column(String(64))
    event_ts = Column(DateTime, nullable=False, index=True)
    payload_json = Column(JSON)
    created_at = Column(TIMESTAMP, nullable=False, server_default=func.current_timestamp())
    
    # Relationships
    container = relationship("Container", back_populates="api_events")
    vessel = relationship("Vessel", back_populates="api_events")


class VesselAdvice(Base):
    """Vessel advice (arrival/port program advice)"""
    __tablename__ = "vessel_advice"
    __table_args__ = {'extend_existing': True}
    
    vessel_advice_no = Column(Integer, primary_key=True, autoincrement=True)
    vessel_name = Column(String(100), nullable=False)
    system_vessel_name = Column(String(20), nullable=False, index=True)
    effective_start_datetime = Column(DateTime, nullable=False, index=True)
    effective_end_datetime = Column(DateTime)
    created_at = Column(TIMESTAMP, nullable=False, server_default=func.current_timestamp())
    
    # Relationships
    berth_applications = relationship("BerthApplication", back_populates="vessel_advice")


class BerthApplication(Base):
    """Berth application referencing vessel advice"""
    __tablename__ = "berth_application"
    __table_args__ = {'extend_existing': True}
    
    application_no = Column(Integer, primary_key=True, autoincrement=True)
    vessel_advice_no = Column(Integer, ForeignKey("vessel_advice.vessel_advice_no", onupdate="CASCADE", ondelete="CASCADE"), nullable=False, index=True)
    vessel_close_datetime = Column(DateTime)
    deleted = Column(String(1), nullable=False, default='N')
    berthing_status = Column(String(1), nullable=False, default='A')
    created_at = Column(TIMESTAMP, nullable=False, server_default=func.current_timestamp())
    
    # Relationships
    vessel_advice = relationship("VesselAdvice", back_populates="berth_applications")


# ============================================================================
# AI ASSISTANT DATABASE MODELS (SQLite)
# ============================================================================

# ResolutionStep table for per-step usefulness tracking
class ResolutionStep(Base):
    __tablename__ = "resolution_step"
    id = Column(Integer, primary_key=True, index=True)
    incident_id = Column(String(255), default="")  # Link to Incident if needed
    order = Column(Integer, default=1)
    description = Column(Text, nullable=False)
    query = Column(Text, default="")
    type = Column(String(50), default="Diagnostic")  # Diagnostic, Resolution, Verification
    usefulness_count = Column(Integer, default=0)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

# New table for usefulness count per error type and solution
class ErrorTypeUsefulness(Base):
    __tablename__ = "error_type_usefulness"
    id = Column(Integer, primary_key=True, index=True)
    error_type = Column(String(255), nullable=False, index=True)
    knowledge_id = Column(Integer, ForeignKey("knowledge_base.id"), nullable=True)
    training_id = Column(Integer, ForeignKey("training_data.id"), nullable=True)
    usefulness_count = Column(Integer, default=0)
    last_marked = Column(DateTime, default=datetime.utcnow)

    knowledge = relationship("KnowledgeBase", backref="usefulness_entries")
    training = relationship("TrainingData", backref="usefulness_entries")

class TrainingData(Base):
    __tablename__ = "training_data"

    # Match existing SQLite schema columns
    id = Column(Integer, primary_key=True, index=True)
    incident_description = Column(Text, nullable=False)
    expected_incident_type = Column(String(255), default="")
    expected_pattern_match = Column(String(255), default="")
    expected_root_cause = Column(Text, default="")
    expected_impact = Column(Text, default="")
    expected_urgency = Column(String(50), default="")
    expected_affected_systems_json = Column(Text, default="")
    category = Column(String(255), default="")
    tags = Column(Text, default="")
    notes = Column(Text, default="")
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    created_by = Column(String(255), default="")
    is_validated = Column(Integer, default=0)
    usefulness_count = Column(Integer, default=0)

    # Helpers used by services/templates
    @property
    def usefulness_count_int(self) -> int:
        """Backward-compatible int accessor used by services/templates.
        Some parts of the app reference `usefulness_count_int`; ensure it exists
        and always returns a safe integer value.
        """
        try:
            return int(self.usefulness_count or 0)
        except Exception:
            return 0

    @property
    def expected_affected_systems(self) -> List[str]:
        if not self.expected_affected_systems_json:
            return []
        try:
            return json.loads(self.expected_affected_systems_json)
        except Exception:
            return []

    @expected_affected_systems.setter
    def expected_affected_systems(self, value: List[str]):
        self.expected_affected_systems_json = json.dumps(value)
    
    def calculate_similarity(self, query: str) -> float:
        """Calculate similarity score with a query"""
        query_lower = query.lower()
        description_lower = self.incident_description.lower()
        
        # Simple keyword matching score
        query_words = set(query_lower.split())
        description_words = set(description_lower.split())
        
        if not query_words:
            return 0.0
        
        # Jaccard similarity
        intersection = query_words.intersection(description_words)
        union = query_words.union(description_words)
        
        jaccard = len(intersection) / len(union) if union else 0.0
        
        # Bonus for exact phrase matches
        phrase_bonus = 0.2 if query_lower in description_lower else 0.0
        
        # Category match bonus
        category_bonus = 0.1 if self.category.lower() in query_lower else 0.0
        
        return min(jaccard + phrase_bonus + category_bonus, 1.0)


class KnowledgeBase(Base):
    __tablename__ = "knowledge_base"
    
    id = Column(Integer, primary_key=True, index=True)
    title = Column(String(500), nullable=False)
    content = Column(Text, nullable=False)
    category = Column(String(255), default="")
    type = Column(String(100), default="")  # Procedure, FAQ, Solution, Reference
    tags = Column(Text, default="")
    keywords = Column(Text, default="")
    priority = Column(Integer, default=1)  # 1=Low, 2=Medium, 3=High, 4=Critical
    source = Column(String(255), default="")  # Word Doc, Manual Entry, Import
    status = Column(String(50), default="Active")  # Active, Inactive, Draft
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    created_by = Column(String(255), default="")
    version_notes = Column(Text, default="")
    view_count = Column(Integer, default=0)
    usefulness_count = Column(Integer, default=0)
    last_used = Column(DateTime, nullable=True)
    
    def calculate_relevance(self, query: str) -> float:
        """Calculate relevance score for a given query"""
        query_lower = query.lower()
        content_lower = self.content.lower()
        title_lower = self.title.lower()
        keywords_lower = self.keywords.lower()
        
        score = 0.0
        
        # Title exact match (highest weight)
        if query_lower in title_lower:
            score += 0.4
        
        # Content contains query
        if query_lower in content_lower:
            score += 0.3
        
        # Keywords match
        if query_lower in keywords_lower:
            score += 0.2
        
        # Word-level matching
        query_words = set(query_lower.split())
        title_words = set(title_lower.split())
        content_words = set(content_lower.split())
        
        # Title word matches
        title_matches = len(query_words.intersection(title_words))
        if title_matches > 0:
            score += 0.2 * (title_matches / len(query_words))
        
        # Content word matches
        content_matches = len(query_words.intersection(content_words))
        if content_matches > 0:
            score += 0.1 * (content_matches / len(query_words))
        
        # Priority bonus
        priority_bonus = self.priority * 0.05
        score += priority_bonus
        
        # Usage bonus (recently used items get slight boost)
        if self.last_used and self.view_count > 0:
            days_since_use = (datetime.utcnow() - self.last_used).days
            usage_bonus = min(0.1, self.view_count * 0.01 / max(days_since_use, 1))
            score += usage_bonus
        
        return min(score, 1.0)


# System Logs table for RCA
class SystemLog(Base):
    __tablename__ = "system_logs"
    
    id = Column(Integer, primary_key=True, index=True)
    incident_id = Column(String(255), index=True, nullable=True)  # Link to RCA
    timestamp = Column(DateTime, index=True, nullable=False)
    level = Column(String(50), index=True)  # ERROR, WARN, INFO, DEBUG
    source_file = Column(String(255))  # app.log, error.log, etc.
    service = Column(String(255))  # Which microservice/component
    message = Column(Text, nullable=False)
    stack_trace = Column(Text, nullable=True)
    error_code = Column(String(100), nullable=True)
    additional_data = Column(JSON, nullable=True)  # Any extra structured data
    uploaded_at = Column(DateTime, default=datetime.utcnow)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "id": self.id,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "level": self.level,
            "source_file": self.source_file,
            "service": self.service,
            "message": self.message,
            "stack_trace": self.stack_trace,
            "error_code": self.error_code,
            "additional_data": self.additional_data
        }


# Root Cause Analysis table
class RootCauseAnalysis(Base):
    __tablename__ = "root_cause_analyses"
    
    id = Column(Integer, primary_key=True, index=True)
    incident_id = Column(String(255), unique=True, index=True, nullable=False)
    incident_description = Column(Text, nullable=False)
    incident_start_time = Column(DateTime, nullable=False)
    incident_end_time = Column(DateTime, nullable=True)
    affected_systems = Column(JSON, nullable=True)  # List of affected systems
    
    # Analysis results
    root_cause = Column(Text, nullable=True)
    confidence_score = Column(Float, default=0.0)  # 0.0 - 1.0
    evidence = Column(JSON, nullable=True)  # List of log entries, metrics
    contributing_factors = Column(JSON, nullable=True)  # List of factors
    error_cascade = Column(JSON, nullable=True)  # Error propagation timeline
    
    # Linked data
    similar_incidents = Column(JSON, nullable=True)  # IDs of similar past incidents
    recommended_solutions = Column(JSON, nullable=True)  # From KB/training data
    sop_references = Column(JSON, nullable=True)  # Relevant SOP documents
    
    # Timeline
    timeline = Column(JSON, nullable=True)  # List of timeline events
    
    # Metadata
    analyzed_at = Column(DateTime, default=datetime.utcnow)
    analyzed_by = Column(String(255), default="AI Assistant")
    search_window_hours = Column(Integer, default=2)
    total_logs_analyzed = Column(Integer, default=0)
    
    # Status
    status = Column(String(50), default="Completed")  # Analyzing, Completed, Failed
    resolution_status = Column(String(50), default="Open")  # Open, Resolved, Monitoring
    resolved_at = Column(DateTime, nullable=True)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "id": self.id,
            "incident_id": self.incident_id,
            "incident_description": self.incident_description,
            "incident_start_time": self.incident_start_time.isoformat() if self.incident_start_time else None,
            "incident_end_time": self.incident_end_time.isoformat() if self.incident_end_time else None,
            "affected_systems": self.affected_systems,
            "root_cause": self.root_cause,
            "confidence_score": self.confidence_score,
            "evidence": self.evidence,
            "contributing_factors": self.contributing_factors,
            "error_cascade": self.error_cascade,
            "similar_incidents": self.similar_incidents,
            "recommended_solutions": self.recommended_solutions,
            "sop_references": self.sop_references,
            "timeline": self.timeline,
            "analyzed_at": self.analyzed_at.isoformat() if self.analyzed_at else None,
            "status": self.status,
            "resolution_status": self.resolution_status
        }

# New table to track solution feedback - which solutions worked for which problems
class SolutionFeedback(Base):
    __tablename__ = "solution_feedback"
    
    id = Column(Integer, primary_key=True, index=True)
    
    # The problem that was being solved
    incident_description = Column(Text, nullable=False)
    
    # The solution that was marked as useful
    solution_description = Column(Text, nullable=False)
    solution_order = Column(Integer, default=1)
    solution_type = Column(String(50), default="Resolution")  # Analysis, Investigation, Resolution, Verification
    
    # Source of the solution (Knowledge Base, Training Data, or RCA History)
    source_type = Column(String(50), default="")  # "Knowledge Base", "Training Data", "RCA History"
    knowledge_base_id = Column(Integer, ForeignKey("knowledge_base.id"), nullable=True)
    training_data_id = Column(Integer, ForeignKey("training_data.id"), nullable=True)
    rca_id = Column(Integer, ForeignKey("root_cause_analyses.id"), nullable=True)
    
    # Tracking
    usefulness_count = Column(Integer, default=1)
    marked_at = Column(DateTime, default=datetime.utcnow)
    user_identifier = Column(String(255), default="")  # Optional: track who marked it useful
    
    # Relationships
    knowledge_base = relationship("KnowledgeBase")
    training_data = relationship("TrainingData")
    rca = relationship("RootCauseAnalysis")
