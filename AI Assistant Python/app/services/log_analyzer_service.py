"""
Log Analyzer Service for Root Cause Analysis
Parses log files, extracts error patterns, and correlates with incidents
"""
from sqlalchemy.orm import Session
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple
import re
import json
import logging
from ..models.database import SystemLog, RootCauseAnalysis

logger = logging.getLogger(__name__)


class LogEntry:
    """Structured log entry"""
    def __init__(self, timestamp: datetime, level: str, message: str, 
                 source_file: str = "", service: str = "", 
                 stack_trace: str = None, error_code: str = None):
        self.timestamp = timestamp
        self.level = level
        self.message = message
        self.source_file = source_file
        self.service = service
        self.stack_trace = stack_trace
        self.error_code = error_code
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp.isoformat(),
            "level": self.level,
            "message": self.message,
            "source_file": self.source_file,
            "service": self.service,
            "stack_trace": self.stack_trace,
            "error_code": self.error_code
        }


class ErrorPattern:
    """Detected error pattern"""
    def __init__(self, pattern_type: str, description: str, 
                 occurrences: int, first_seen: datetime, last_seen: datetime,
                 related_logs: List[LogEntry] = None):
        self.pattern_type = pattern_type
        self.description = description
        self.occurrences = occurrences
        self.first_seen = first_seen
        self.last_seen = last_seen
        self.related_logs = related_logs or []
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "pattern_type": self.pattern_type,
            "description": self.description,
            "occurrences": self.occurrences,
            "first_seen": self.first_seen.isoformat(),
            "last_seen": self.last_seen.isoformat(),
            "duration_seconds": (self.last_seen - self.first_seen).total_seconds()
        }


class RootCauseHypothesis:
    """Root cause hypothesis with evidence"""
    def __init__(self, description: str, confidence: float, 
                 evidence: List[str], contributing_factors: List[str] = None):
        self.description = description
        self.confidence = confidence
        self.evidence = evidence
        self.contributing_factors = contributing_factors or []
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "description": self.description,
            "confidence": self.confidence,
            "evidence": self.evidence,
            "contributing_factors": self.contributing_factors
        }


class LogAnalyzerService:
    """Service for analyzing system logs and performing root cause analysis"""
    
    def __init__(self, db: Session):
        self.db = db
    
    async def parse_log_file(self, file_content: bytes, filename: str) -> List[LogEntry]:
        """
        Parse log file and extract structured log entries
        Supports: plaintext, JSON logs, CSV
        """
        try:
            content = file_content.decode('utf-8')
            entries = []
            
            # Try JSON format first
            if filename.endswith('.json'):
                entries = self._parse_json_logs(content)
            else:
                # Try plaintext format (most common)
                entries = self._parse_plaintext_logs(content, filename)
            
            logger.info(f"Parsed {len(entries)} log entries from {filename}")
            return entries
            
        except Exception as ex:
            logger.error(f"Error parsing log file {filename}: {ex}")
            return []
    
    def _parse_json_logs(self, content: str) -> List[LogEntry]:
        """Parse JSON format logs"""
        entries = []
        
        for line in content.strip().split('\n'):
            if not line.strip():
                continue
            
            try:
                log_obj = json.loads(line)
                
                # Extract timestamp
                timestamp = self._extract_timestamp(
                    log_obj.get('timestamp') or log_obj.get('time') or log_obj.get('@timestamp')
                )
                
                # Extract level
                level = log_obj.get('level') or log_obj.get('severity') or 'INFO'
                level = level.upper()
                
                # Extract message
                message = log_obj.get('message') or log_obj.get('msg') or str(log_obj)
                
                # Extract optional fields
                service = log_obj.get('service') or log_obj.get('logger') or ''
                error_code = log_obj.get('error_code') or log_obj.get('code') or None
                stack_trace = log_obj.get('stack_trace') or log_obj.get('exception') or None
                
                if timestamp and message:
                    entries.append(LogEntry(
                        timestamp=timestamp,
                        level=level,
                        message=message,
                        source_file='json_log',
                        service=service,
                        stack_trace=stack_trace,
                        error_code=error_code
                    ))
            except json.JSONDecodeError:
                continue
        
        return entries
    
    def _parse_plaintext_logs(self, content: str, filename: str) -> List[LogEntry]:
        """Parse plaintext logs (most common format)"""
        entries = []
        
        # Common log patterns
        patterns = [
            # Pattern 1: [2024-10-19 14:30:15] ERROR: Message
            r'\[(\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2})\]\s+(\w+):\s+(.+)',
            # Pattern 2: 2024-10-19 14:30:15 ERROR Message
            r'(\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2})\s+(\w+)\s+(.+)',
            # Pattern 3: [ERROR] [2024-10-19 14:30:15] Message
            r'\[(\w+)\]\s+\[(\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2})\]\s+(.+)',
            # Pattern 4: ERROR 2024-10-19 14:30:15 - Message
            r'(\w+)\s+(\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2})\s+-\s+(.+)',
        ]
        
        for line in content.split('\n'):
            if not line.strip():
                continue
            
            matched = False
            for pattern in patterns:
                match = re.match(pattern, line)
                if match:
                    groups = match.groups()
                    
                    # Determine timestamp and level position
                    if len(groups) >= 3:
                        # Check if first group is timestamp or level
                        if re.match(r'\d{4}-\d{2}-\d{2}', groups[0]):
                            timestamp_str, level, message = groups[0], groups[1], groups[2]
                        else:
                            level, timestamp_str, message = groups[0], groups[1], groups[2]
                        
                        timestamp = self._extract_timestamp(timestamp_str)
                        if timestamp:
                            entries.append(LogEntry(
                                timestamp=timestamp,
                                level=level.upper(),
                                message=message.strip(),
                                source_file=filename,
                                service=''
                            ))
                            matched = True
                            break
            
            # If no pattern matched but contains ERROR/WARN, still capture it
            if not matched and any(keyword in line.upper() for keyword in ['ERROR', 'WARN', 'CRITICAL', 'FATAL']):
                # Use current time as fallback
                level = 'ERROR' if 'ERROR' in line.upper() else 'WARN'
                entries.append(LogEntry(
                    timestamp=datetime.now(),
                    level=level,
                    message=line.strip(),
                    source_file=filename,
                    service=''
                ))
        
        return entries
    
    def _extract_timestamp(self, timestamp_str: str) -> Optional[datetime]:
        """Extract datetime from various timestamp formats"""
        if not timestamp_str:
            return None
        
        formats = [
            '%Y-%m-%d %H:%M:%S',
            '%Y-%m-%dT%H:%M:%S',
            '%Y-%m-%d %H:%M:%S.%f',
            '%Y-%m-%dT%H:%M:%S.%f',
            '%d/%m/%Y %H:%M:%S',
            '%m/%d/%Y %H:%M:%S',
        ]
        
        for fmt in formats:
            try:
                return datetime.strptime(timestamp_str, fmt)
            except ValueError:
                continue
        
        return None
    
    def save_logs_to_db(self, logs: List[LogEntry], incident_id: str = None) -> int:
        """Save parsed logs to database"""
        saved_count = 0
        
        for log in logs:
            try:
                db_log = SystemLog(
                    incident_id=incident_id,
                    timestamp=log.timestamp,
                    level=log.level,
                    source_file=log.source_file,
                    service=log.service,
                    message=log.message,
                    stack_trace=log.stack_trace,
                    error_code=log.error_code
                )
                self.db.add(db_log)
                saved_count += 1
            except Exception as ex:
                logger.error(f"Error saving log: {ex}")
        
        self.db.commit()
        logger.info(f"Saved {saved_count} logs to database")
        return saved_count
    
    def find_logs_around_time(self, 
                               incident_time: datetime, 
                               window_minutes: int = 120,
                               level_filter: List[str] = None) -> List[SystemLog]:
        """Find logs within time window of incident"""
        start_time = incident_time - timedelta(minutes=window_minutes)
        end_time = incident_time + timedelta(minutes=window_minutes)
        
        query = self.db.query(SystemLog).filter(
            SystemLog.timestamp >= start_time,
            SystemLog.timestamp <= end_time
        )
        
        if level_filter:
            query = query.filter(SystemLog.level.in_(level_filter))
        
        logs = query.order_by(SystemLog.timestamp.asc()).all()
        logger.info(f"Found {len(logs)} logs around {incident_time} (±{window_minutes} min)")
        return logs
    
    def detect_error_patterns(self, logs: List[SystemLog]) -> List[ErrorPattern]:
        """Detect error patterns in logs"""
        patterns = []
        
        # Group errors by message similarity
        error_groups = {}
        
        for log in logs:
            if log.level not in ['ERROR', 'CRITICAL', 'FATAL']:
                continue
            
            # Extract error key (first 50 chars, normalized)
            error_key = self._normalize_error_message(log.message)[:50]
            
            if error_key not in error_groups:
                error_groups[error_key] = []
            error_groups[error_key].append(log)
        
        # Create patterns from groups
        for error_key, group_logs in error_groups.items():
            if len(group_logs) >= 2:  # Pattern needs at least 2 occurrences
                pattern = ErrorPattern(
                    pattern_type="REPEATED_ERROR",
                    description=group_logs[0].message[:200],
                    occurrences=len(group_logs),
                    first_seen=min(log.timestamp for log in group_logs),
                    last_seen=max(log.timestamp for log in group_logs)
                )
                patterns.append(pattern)
        
        logger.info(f"Detected {len(patterns)} error patterns")
        return patterns
    
    def _normalize_error_message(self, message: str) -> str:
        """Normalize error message for pattern matching"""
        # Remove numbers, IDs, timestamps
        normalized = re.sub(r'\d+', 'N', message)
        normalized = re.sub(r'[a-f0-9-]{32,}', 'ID', normalized)  # UUIDs
        normalized = normalized.lower().strip()
        return normalized
    
    def detect_error_cascade(self, logs: List[SystemLog]) -> List[Dict[str, Any]]:
        """Detect error cascades (A → B → C)"""
        cascade = []
        
        # Sort by timestamp
        sorted_logs = sorted(logs, key=lambda x: x.timestamp)
        
        # Look for error propagation (errors within 5 seconds of each other)
        for i, log in enumerate(sorted_logs):
            if log.level in ['ERROR', 'CRITICAL', 'FATAL']:
                cascade_item = {
                    "timestamp": log.timestamp.isoformat(),
                    "level": log.level,
                    "message": log.message[:150],
                    "service": log.service or "Unknown"
                }
                
                # Check if related to previous error (within 5 seconds)
                if i > 0:
                    time_diff = (log.timestamp - sorted_logs[i-1].timestamp).total_seconds()
                    if time_diff <= 5 and sorted_logs[i-1].level in ['ERROR', 'CRITICAL', 'FATAL']:
                        cascade_item["cascade_from"] = i - 1
                
                cascade.append(cascade_item)
        
        return cascade
    
    def extract_root_cause_candidates(self, 
                                       logs: List[SystemLog],
                                       incident_description: str) -> List[RootCauseHypothesis]:
        """Generate root cause hypotheses"""
        hypotheses = []
        
        # Find first critical error
        critical_errors = [log for log in logs if log.level in ['ERROR', 'CRITICAL', 'FATAL']]
        
        if not critical_errors:
            return hypotheses
        
        # Sort by timestamp
        critical_errors.sort(key=lambda x: x.timestamp)
        first_error = critical_errors[0]
        
        # Hypothesis 1: First error is root cause
        evidence = [
            f"First error occurred at {first_error.timestamp.strftime('%H:%M:%S')}",
            f"Error message: {first_error.message[:200]}",
            f"Total errors in cascade: {len(critical_errors)}"
        ]
        
        # Check for common root causes
        message_lower = first_error.message.lower()
        
        if any(keyword in message_lower for keyword in ['connection', 'timeout', 'pool']):
            root_cause = "Database connection pool exhaustion or timeout"
            confidence = 0.85
            evidence.append("Multiple connection-related errors detected")
        elif any(keyword in message_lower for keyword in ['memory', 'heap', 'oom']):
            root_cause = "Memory exhaustion (Out of Memory)"
            confidence = 0.90
            evidence.append("Memory-related errors detected")
        elif any(keyword in message_lower for keyword in ['network', 'unreachable', 'refused']):
            root_cause = "Network connectivity issue"
            confidence = 0.80
            evidence.append("Network-related errors detected")
        elif any(keyword in message_lower for keyword in ['deadlock', 'lock timeout']):
            root_cause = "Database deadlock or lock contention"
            confidence = 0.88
            evidence.append("Lock-related errors detected")
        else:
            root_cause = first_error.message[:200]
            confidence = 0.60
        
        hypotheses.append(RootCauseHypothesis(
            description=root_cause,
            confidence=confidence,
            evidence=evidence,
            contributing_factors=self._identify_contributing_factors(logs)
        ))
        
        return hypotheses
    
    def _identify_contributing_factors(self, logs: List[SystemLog]) -> List[str]:
        """Identify contributing factors from logs"""
        factors = []
        
        # Check for warnings before errors
        warnings = [log for log in logs if log.level == 'WARN']
        if warnings:
            factors.append(f"System issued {len(warnings)} warnings before failure")
        
        # Check for high error rate
        errors = [log for log in logs if log.level in ['ERROR', 'CRITICAL', 'FATAL']]
        if len(errors) > 10:
            time_span = (logs[-1].timestamp - logs[0].timestamp).total_seconds()
            if time_span > 0:
                rate = len(errors) / time_span
                if rate > 1:  # More than 1 error per second
                    factors.append(f"High error rate: {rate:.1f} errors/second")
        
        # Check for specific patterns
        messages = ' '.join([log.message.lower() for log in logs[:50]])
        
        if 'batch' in messages or 'scheduled' in messages:
            factors.append("Incident coincides with batch job execution")
        
        if 'spike' in messages or 'load' in messages:
            factors.append("System load spike detected")
        
        return factors
    
    def build_timeline(self, logs: List[SystemLog], 
                       incident_start: datetime,
                       incident_end: datetime = None) -> List[Dict[str, Any]]:
        """Build timeline of events"""
        timeline = []
        
        # Sort logs by timestamp
        sorted_logs = sorted(logs, key=lambda x: x.timestamp)
        
        # Add key events to timeline
        for log in sorted_logs:
            if log.level in ['ERROR', 'CRITICAL', 'FATAL', 'WARN']:
                timeline.append({
                    "timestamp": log.timestamp.isoformat(),
                    "type": log.level,
                    "message": log.message[:150],
                    "service": log.service or "Unknown",
                    "is_critical": log.level in ['ERROR', 'CRITICAL', 'FATAL']
                })
        
        # Limit timeline to most relevant entries
        return timeline[:50]  # Top 50 events
