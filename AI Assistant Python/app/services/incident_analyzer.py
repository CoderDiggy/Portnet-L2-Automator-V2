from sqlalchemy.orm import Session
from typing import List
import logging
from .openai_service import OpenAIService
from .training_data_service import TrainingDataService
from .knowledge_base_service import KnowledgeBaseService
from ..models.schemas import IncidentAnalysis

logger = logging.getLogger(__name__)


# Utility for error type extraction
import re
def extract_error_type(description: str) -> str:
    """
    Extracts a simple error type from a problem statement using pattern matching.
    Examples:
    "Unexpected qualifier 'BN' in EQD segment" -> "unexpected_qualifier"
    "Time zone drift UTC+0 for Partner-E" -> "timezone_drift"
    "Spike in DLQ messages" -> "dlq_spike"
    "VESSEL_ERR_4 when creating vessel advice" -> "vessel_err"
    "EDI message stuck in ERROR status" -> "edi_error"
    "Duplicate container info" -> "container_duplication"
    "Timeout occurred" -> "timeout"
    "DLQ spike" -> "dlq_spike"
    "Database deadlock" -> "deadlock"
    "Connection refused" -> "connection_refused"
    "Invalid format" -> "invalid_format"
    "Missing field" -> "missing_field"
    "Authentication failed" -> "auth_failed"
    "Permission denied" -> "permission_denied"
    "File not found" -> "file_not_found"
    "Memory leak" -> "memory_leak"
    "High CPU usage" -> "high_cpu"
    "Disk full" -> "disk_full"
    "Network unreachable" -> "network_unreachable"
    "Service unavailable" -> "service_unavailable"
    "Unknown error" -> "unknown_error"
    """
    patterns = [
        # Enhanced EDIFACT and port-specific patterns
        (r"unexpected qualifier.*['\"]\w+['\"]\s+in\s+\w+\s+segment", "edifact_unexpected_qualifier"),
        (r"coarri.*container.*translation|container.*coarri.*error", "coarri_container_error"),
        (r"edifact.*parse|edifact.*format|edifact.*message", "edifact_parsing_error"),
        (r"codeco.*error|codeco.*reject", "codeco_error"),
        (r"coprar.*error|coprar.*reject", "coprar_error"),
        (r"baplie.*error|baplie.*reject", "baplie_error"),
        (r"edi.*message.*stuck|edi.*stuck.*error", "edi_message_stuck"),
        (r"segment.*error|segment.*reject|segment.*invalid", "edi_segment_error"),
        (r"time ?zone drift", "timezone_drift"),
        (r"dlq.*spike|spike.*dlq|dlq messages", "dlq_spike"),
        (r"vessel_err|vessel error", "vessel_err"),
        (r"duplicate.*container|container.*duplication", "container_duplication"),
        (r"cntr.*duplicate|cntr.*error", "container_reference_error"),
        (r"booking.*duplicate|booking.*conflict", "booking_conflict"),
        (r"timeout", "timeout"),
        (r"deadlock", "deadlock"),
        (r"connection refused", "connection_refused"),
        (r"invalid format", "invalid_format"),
        (r"missing field", "missing_field"),
        (r"auth.*fail|authentication failed", "auth_failed"),
        (r"permission denied", "permission_denied"),
        (r"file not found", "file_not_found"),
        (r"memory leak", "memory_leak"),
        (r"high cpu", "high_cpu"),
        (r"disk full", "disk_full"),
        (r"network unreachable", "network_unreachable"),
        (r"service unavailable", "service_unavailable"),
        (r"unknown error", "unknown_error"),
    ]
    desc_lower = description.lower()
    for pat, etype in patterns:
        if re.search(pat, desc_lower):
            return etype
    # Fallback: use first 2 words joined with underscore
    words = re.findall(r"\w+", desc_lower)
    return "_".join(words[:2]) if words else "unknown_error"

class IncidentAnalyzer:
    def __init__(self, db: Session):
        self.db = db
        self.openai_service = OpenAIService()
        self.training_service = TrainingDataService(db)
        self.knowledge_service = KnowledgeBaseService(db)
    
    async def analyze_incident_async(self, description: str):
        """
        Analyze incident using AI with training data and knowledge base context
        Returns: tuple of (analysis, knowledge_entries, training_examples)
        """
        try:
            logger.info(f"Analyzing incident: {description[:100]}...")

            # Extract error type
            error_type = extract_error_type(description)
            logger.info(f"Extracted error type: {error_type}")

            # Use FULL DESCRIPTION for searching (not just error_type) to get better matches
            training_examples = await self.training_service.find_relevant_examples_async(description, 3)
            knowledge_entries = await self.knowledge_service.find_relevant_knowledge_async(description, 5)

            logger.info(f"Found {len(training_examples)} training examples and {len(knowledge_entries)} knowledge entries")

            # Use OpenAI service for analysis
            analysis = await self.openai_service.analyze_incident_async(description, training_examples, knowledge_entries)

            logger.info(f"Analysis completed. Type: {analysis.incident_type}, Urgency: {analysis.urgency}")

            return analysis, knowledge_entries, training_examples

        except Exception as ex:
            logger.error(f"Error analyzing incident: {ex}")
            # Return fallback analysis
            fallback = IncidentAnalysis(
                incident_type="System Issue",
                pattern_match="Error during analysis",
                root_cause="Analysis failed - requires manual investigation",
                impact="Unknown - manual assessment required",
                urgency="Medium",
                affected_systems=["Unknown"]
            )
            return fallback, [], []