"""
Operational Data Service - Queries PORTNET operational database
Correlates incidents with actual container, vessel, EDI, and API event data
"""
from sqlalchemy.orm import Session
from sqlalchemy import func, desc, and_, or_
from typing import List, Dict, Optional, Any, Tuple
from datetime import datetime, timedelta
import re

from app.models.database import (
    Vessel, Container, EDIMessage, APIEvent,
    VesselAdvice, BerthApplication,
    EDIMessageTypeEnum, EDIStatusEnum, ContainerStatusEnum
)


class OperationalDataService:
    """Service for querying and analyzing operational database"""
    
    def __init__(self, ops_db: Session):
        self.ops_db = ops_db
    
    # ========================================================================
    # ENTITY EXTRACTION FROM INCIDENT DESCRIPTION
    # ========================================================================
    
    def extract_identifiers(self, text: str) -> Dict[str, List[str]]:
        """
        Extract container numbers, vessel names, error codes from text
        Returns: {
            'containers': ['CMAU0000020', ...],
            'vessels': ['MV Lion City 07', ...],
            'error_codes': ['VESSEL_ERR_4', ...],
            'edi_refs': ['REF-COP-0001', ...],
            'correlation_ids': ['corr-0001', ...]
        }
        """
        identifiers = {
            'containers': [],
            'vessels': [],
            'error_codes': [],
            'edi_refs': [],
            'correlation_ids': [],
            'imo_numbers': []
        }
        
        # Container numbers (4 letter prefix + 7 digits)
        container_pattern = r'\b[A-Z]{4}\d{7}\b'
        identifiers['containers'] = re.findall(container_pattern, text, re.IGNORECASE)
        
        # Vessel names (MV/MS + name)
        vessel_pattern = r'\b(?:MV|MS|MT)\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*(?:\s+\d+)?\b'
        identifiers['vessels'] = re.findall(vessel_pattern, text, re.IGNORECASE)
        
        # Error codes (WORD_ERR_NUM or similar)
        error_pattern = r'\b[A-Z]+_(?:ERR|ERROR|WARN)_\d+\b'
        identifiers['error_codes'] = re.findall(error_pattern, text, re.IGNORECASE)
        
        # EDI message references
        edi_pattern = r'\bREF-[A-Z]+-\d+\b'
        identifiers['edi_refs'] = re.findall(edi_pattern, text, re.IGNORECASE)
        
        # Correlation IDs
        corr_pattern = r'\bcorr-\d+\b'
        identifiers['correlation_ids'] = re.findall(corr_pattern, text, re.IGNORECASE)
        
        # IMO numbers
        imo_pattern = r'\bIMO\s*(\d{7})\b|\b(\d{7})\b'
        imo_matches = re.findall(imo_pattern, text, re.IGNORECASE)
        identifiers['imo_numbers'] = [m[0] or m[1] for m in imo_matches if len(m[0] or m[1]) == 7]
        
        return identifiers
    
    # ========================================================================
    # CONTAINER QUERIES
    # ========================================================================
    
    def get_container_by_number(self, cntr_no: str) -> List[Container]:
        """Get all versions of a container (composite PK may have multiple rows)"""
        return self.ops_db.query(Container).filter(
            Container.cntr_no == cntr_no.upper()
        ).order_by(Container.created_at.desc()).all()
    
    def detect_container_duplicates(self, cntr_no: str) -> Dict[str, Any]:
        """
        Check for duplicate container records (same cntr_no, multiple created_at)
        Returns analysis of duplication issue
        """
        containers = self.get_container_by_number(cntr_no)
        
        if len(containers) <= 1:
            return {"has_duplicates": False, "count": len(containers)}
        
        # Analyze duplicates
        analysis = {
            "has_duplicates": True,
            "count": len(containers),
            "records": [],
            "identical_data": True,
            "issue_type": "composite_pk_versioning"
        }
        
        for c in containers:
            record = {
                "created_at": c.created_at.isoformat(),
                "status": c.status.value if hasattr(c.status, 'value') else c.status,
                "vessel_id": c.vessel_id,
                "origin": c.origin_port,
                "destination": c.destination_port
            }
            analysis["records"].append(record)
        
        # Check if all records are identical (true duplicates vs intentional versions)
        first = containers[0]
        for c in containers[1:]:
            if (c.status != first.status or c.vessel_id != first.vessel_id or
                c.origin_port != first.origin_port or c.destination_port != first.destination_port):
                analysis["identical_data"] = False
                analysis["issue_type"] = "data_inconsistency"
                break
        
        if analysis["identical_data"]:
            time_diff = (containers[0].created_at - containers[-1].created_at).total_seconds()
            if time_diff < 5:  # Less than 5 seconds apart
                analysis["issue_type"] = "rapid_duplicate_insert"
                analysis["root_cause"] = f"Multiple inserts within {time_diff:.1f}s - likely race condition or double-submit"
        
        return analysis
    
    def find_containers_by_criteria(self, 
                                    status: Optional[str] = None,
                                    vessel_id: Optional[int] = None,
                                    origin: Optional[str] = None,
                                    destination: Optional[str] = None,
                                    hazardous: Optional[bool] = None,
                                    time_window: Optional[Tuple[datetime, datetime]] = None,
                                    limit: int = 50) -> List[Container]:
        """Find containers matching criteria"""
        query = self.ops_db.query(Container)
        
        if status:
            query = query.filter(Container.status == status)
        if vessel_id:
            query = query.filter(Container.vessel_id == vessel_id)
        if origin:
            query = query.filter(Container.origin_port == origin)
        if destination:
            query = query.filter(Container.destination_port == destination)
        if hazardous is not None:
            if hazardous:
                query = query.filter(Container.hazard_class.isnot(None))
            else:
                query = query.filter(Container.hazard_class.is_(None))
        if time_window:
            start, end = time_window
            query = query.filter(Container.created_at.between(start, end))
        
        return query.order_by(Container.created_at.desc()).limit(limit).all()
    
    # ========================================================================
    # VESSEL QUERIES
    # ========================================================================
    
    def get_vessel_by_name(self, vessel_name: str) -> Optional[Vessel]:
        """Get vessel by name (case-insensitive partial match)"""
        return self.ops_db.query(Vessel).filter(
            Vessel.vessel_name.ilike(f"%{vessel_name}%")
        ).first()
    
    def get_vessel_by_imo(self, imo_no: int) -> Optional[Vessel]:
        """Get vessel by IMO number"""
        return self.ops_db.query(Vessel).filter(Vessel.imo_no == imo_no).first()
    
    def get_vessel_advice_by_name(self, system_vessel_name: str) -> List[VesselAdvice]:
        """Get vessel advice records (including historical)"""
        return self.ops_db.query(VesselAdvice).filter(
            VesselAdvice.system_vessel_name == system_vessel_name
        ).order_by(VesselAdvice.effective_start_datetime.desc()).all()
    
    def detect_vessel_advice_conflict(self, vessel_name: str) -> Dict[str, Any]:
        """
        Detect vessel advice conflicts (VESSEL_ERR_4: active advice already exists)
        """
        advices = self.get_vessel_advice_by_name(vessel_name)
        
        if not advices:
            return {"has_conflict": False}
        
        # Find active advice (effective_end_datetime IS NULL)
        active = [a for a in advices if a.effective_end_datetime is None]
        
        if not active:
            return {
                "has_conflict": False,
                "historical_count": len(advices),
                "message": f"No active advice found, {len(advices)} historical record(s) exist"
            }
        
        if len(active) > 1:
            return {
                "has_conflict": True,
                "error_type": "MULTIPLE_ACTIVE",
                "active_count": len(active),
                "root_cause": "Data integrity violation - multiple active advices found (should be prevented by unique constraint)",
                "advices": [{"id": a.vessel_advice_no, "start": a.effective_start_datetime.isoformat()} for a in active]
            }
        
        # Single active advice exists
        active_advice = active[0]
        return {
            "has_conflict": True,
            "error_type": "VESSEL_ERR_4",
            "active_advice_no": active_advice.vessel_advice_no,
            "active_since": active_advice.effective_start_datetime.isoformat(),
            "root_cause": f"Cannot create new advice - vessel '{vessel_name}' already has active advice #{active_advice.vessel_advice_no}",
            "solution": "Expire the existing advice by setting effective_end_datetime before creating new advice",
            "berth_applications": len(active_advice.berth_applications)
        }
    
    # ========================================================================
    # EDI MESSAGE QUERIES
    # ========================================================================
    
    def get_edi_by_reference(self, message_ref: str) -> Optional[EDIMessage]:
        """Get EDI message by reference"""
        return self.ops_db.query(EDIMessage).filter(
            EDIMessage.message_ref == message_ref
        ).first()
    
    def get_edi_errors(self, time_window: Optional[Tuple[datetime, datetime]] = None,
                       limit: int = 50) -> List[EDIMessage]:
        """Get EDI messages with ERROR status"""
        query = self.ops_db.query(EDIMessage).filter(EDIMessage.status == EDIStatusEnum.ERROR)
        
        if time_window:
            start, end = time_window
            query = query.filter(EDIMessage.sent_at.between(start, end))
        
        return query.order_by(EDIMessage.sent_at.desc()).limit(limit).all()
    
    def get_edi_for_container(self, container_id: int, limit: int = 20) -> List[EDIMessage]:
        """Get EDI messages for a specific container"""
        return self.ops_db.query(EDIMessage).filter(
            EDIMessage.container_id == container_id
        ).order_by(EDIMessage.sent_at.desc()).limit(limit).all()
    
    def analyze_edi_error(self, message_ref: str) -> Dict[str, Any]:
        """Analyze EDI error message"""
        edi = self.get_edi_by_reference(message_ref)
        
        if not edi:
            return {"found": False}
        
        analysis = {
            "found": True,
            "message_ref": message_ref,
            "type": edi.message_type.value if hasattr(edi.message_type, 'value') else edi.message_type,
            "status": edi.status.value if hasattr(edi.status, 'value') else edi.status,
            "error_text": edi.error_text,
            "sent_at": edi.sent_at.isoformat(),
            "sender": edi.sender,
            "receiver": edi.receiver
        }
        
        # Add container context if available
        if edi.container:
            analysis["container"] = {
                "cntr_no": edi.container.cntr_no,
                "status": edi.container.status.value if hasattr(edi.container.status, 'value') else edi.container.status,
                "vessel_id": edi.container.vessel_id
            }
        
        # Determine root cause from error text
        if edi.error_text:
            error_lower = edi.error_text.lower()
            if "segment missing" in error_lower:
                analysis["root_cause"] = "EDI message structure incomplete - required segment not found"
                analysis["solution"] = "Verify sender's EDI message template and segment ordering"
            elif "validation" in error_lower:
                analysis["root_cause"] = "EDI message validation failed - invalid data format or values"
                analysis["solution"] = "Check data type constraints and code list values"
            elif "timeout" in error_lower:
                analysis["root_cause"] = "EDI processing timeout - message too large or system overload"
                analysis["solution"] = "Review message size limits and system performance"
        
        return analysis
    
    # ========================================================================
    # API EVENT QUERIES
    # ========================================================================
    
    def get_api_events_by_correlation(self, correlation_id: str) -> List[APIEvent]:
        """Get all API events with same correlation ID (distributed tracing)"""
        return self.ops_db.query(APIEvent).filter(
            APIEvent.correlation_id == correlation_id
        ).order_by(APIEvent.event_ts).all()
    
    def get_api_events_in_timerange(self, 
                                    start: datetime, 
                                    end: datetime,
                                    event_type: Optional[str] = None,
                                    source_system: Optional[str] = None,
                                    limit: int = 100) -> List[APIEvent]:
        """Get API events within time range"""
        query = self.ops_db.query(APIEvent).filter(
            APIEvent.event_ts.between(start, end)
        )
        
        if event_type:
            query = query.filter(APIEvent.event_type == event_type)
        if source_system:
            query = query.filter(APIEvent.source_system == source_system)
        
        return query.order_by(APIEvent.event_ts).limit(limit).all()
    
    def detect_api_event_cascade(self, 
                                 start: datetime,
                                 end: datetime,
                                 cascade_window_seconds: int = 10) -> List[Dict[str, Any]]:
        """
        Detect cascading API failures (errors within X seconds of each other)
        """
        # Get all 4xx/5xx responses
        events = self.ops_db.query(APIEvent).filter(
            and_(
                APIEvent.event_ts.between(start, end),
                APIEvent.http_status >= 400
            )
        ).order_by(APIEvent.event_ts).all()
        
        if not events:
            return []
        
        cascades = []
        current_cascade = [events[0]]
        
        for i in range(1, len(events)):
            time_diff = (events[i].event_ts - current_cascade[-1].event_ts).total_seconds()
            
            if time_diff <= cascade_window_seconds:
                current_cascade.append(events[i])
            else:
                if len(current_cascade) >= 2:  # Only report cascades with 2+ events
                    cascades.append({
                        "start_time": current_cascade[0].event_ts.isoformat(),
                        "end_time": current_cascade[-1].event_ts.isoformat(),
                        "duration_seconds": (current_cascade[-1].event_ts - current_cascade[0].event_ts).total_seconds(),
                        "event_count": len(current_cascade),
                        "events": [
                            {
                                "api_id": e.api_id,
                                "event_type": e.event_type.value if hasattr(e.event_type, 'value') else e.event_type,
                                "source_system": e.source_system,
                                "http_status": e.http_status,
                                "timestamp": e.event_ts.isoformat()
                            }
                            for e in current_cascade
                        ]
                    })
                current_cascade = [events[i]]
        
        # Don't forget last cascade
        if len(current_cascade) >= 2:
            cascades.append({
                "start_time": current_cascade[0].event_ts.isoformat(),
                "end_time": current_cascade[-1].event_ts.isoformat(),
                "duration_seconds": (current_cascade[-1].event_ts - current_cascade[0].event_ts).total_seconds(),
                "event_count": len(current_cascade),
                "events": [
                    {
                        "api_id": e.api_id,
                        "event_type": e.event_type.value if hasattr(e.event_type, 'value') else e.event_type,
                        "source_system": e.source_system,
                        "http_status": e.http_status,
                        "timestamp": e.event_ts.isoformat()
                    }
                    for e in current_cascade
                ]
            })
        
        return cascades
    
    # ========================================================================
    # COMPREHENSIVE INCIDENT CORRELATION
    # ========================================================================
    
    def correlate_incident(self, 
                          incident_description: str,
                          incident_time: datetime,
                          search_window_hours: int = 2) -> Dict[str, Any]:
        """
        Main correlation function - extract identifiers and query all relevant data
        """
        # Extract identifiers from description
        identifiers = self.extract_identifiers(incident_description)
        
        # Time window for queries
        start_time = incident_time - timedelta(hours=search_window_hours)
        end_time = incident_time + timedelta(hours=search_window_hours)
        
        correlation_result = {
            "identifiers": identifiers,
            "time_window": {
                "start": start_time.isoformat(),
                "end": end_time.isoformat(),
                "hours": search_window_hours
            },
            "findings": {}
        }
        
        # Query containers
        if identifiers['containers']:
            container_findings = []
            for cntr_no in identifiers['containers']:
                dup_analysis = self.detect_container_duplicates(cntr_no)
                containers = self.get_container_by_number(cntr_no)
                
                if containers:
                    container_findings.append({
                        "cntr_no": cntr_no,
                        "found": True,
                        "version_count": len(containers),
                        "current_status": containers[0].status.value if hasattr(containers[0].status, 'value') else containers[0].status,
                        "vessel_id": containers[0].vessel_id,
                        "duplication_analysis": dup_analysis
                    })
            
            if container_findings:
                correlation_result["findings"]["containers"] = container_findings
        
        # Query vessels
        if identifiers['vessels']:
            vessel_findings = []
            for vessel_name in identifiers['vessels']:
                vessel = self.get_vessel_by_name(vessel_name)
                if vessel:
                    # Check for vessel advice conflicts
                    advice_conflict = self.detect_vessel_advice_conflict(vessel_name)
                    
                    vessel_findings.append({
                        "vessel_name": vessel_name,
                        "found": True,
                        "vessel_id": vessel.vessel_id,
                        "imo_no": vessel.imo_no,
                        "operator": vessel.operator_name,
                        "advice_conflict": advice_conflict
                    })
            
            if vessel_findings:
                correlation_result["findings"]["vessels"] = vessel_findings
        
        # Query EDI messages
        if identifiers['edi_refs']:
            edi_findings = []
            for ref in identifiers['edi_refs']:
                edi_analysis = self.analyze_edi_error(ref)
                if edi_analysis.get("found"):
                    edi_findings.append(edi_analysis)
            
            if edi_findings:
                correlation_result["findings"]["edi_messages"] = edi_findings
        
        # Get EDI errors in time window
        edi_errors = self.get_edi_errors((start_time, end_time), limit=20)
        if edi_errors:
            correlation_result["findings"]["edi_errors_in_window"] = [
                {
                    "message_ref": e.message_ref,
                    "type": e.message_type.value if hasattr(e.message_type, 'value') else e.message_type,
                    "error": e.error_text,
                    "sent_at": e.sent_at.isoformat()
                }
                for e in edi_errors
            ]
        
        # Query API events
        if identifiers['correlation_ids']:
            api_findings = []
            for corr_id in identifiers['correlation_ids']:
                events = self.get_api_events_by_correlation(corr_id)
                if events:
                    api_findings.append({
                        "correlation_id": corr_id,
                        "event_count": len(events),
                        "events": [
                            {
                                "event_type": e.event_type.value if hasattr(e.event_type, 'value') else e.event_type,
                                "source_system": e.source_system,
                                "http_status": e.http_status,
                                "timestamp": e.event_ts.isoformat()
                            }
                            for e in events
                        ]
                    })
            
            if api_findings:
                correlation_result["findings"]["api_events_by_correlation"] = api_findings
        
        # Detect API event cascades
        cascades = self.detect_api_event_cascade(start_time, end_time)
        if cascades:
            correlation_result["findings"]["api_event_cascades"] = cascades
        
        return correlation_result
