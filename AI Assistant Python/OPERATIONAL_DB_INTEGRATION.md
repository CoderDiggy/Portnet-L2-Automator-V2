# Operational Database Integration - Complete

## 🎯 Overview

The AI Duty Officer Assistant now integrates with the **PORTNET operational database** (MySQL) to correlate incidents with actual container, vessel, EDI, and API event data. This transforms the RCA system from log-only analysis to **full operational data correlation**.

---

## 📊 Architecture: Dual Database System

### Database 1: AI Assistant DB (SQLite)
**Purpose**: Store AI-related data (knowledge base, training data, RCA results)

**Tables**:
- `knowledge_base` - SOPs and documentation
- `training_data` - Historical incident cases
- `resolution_step` - Resolution step tracking
- `system_logs` - Uploaded log files
- `root_cause_analyses` - RCA results

### Database 2: Operational DB (MySQL - PORTNET)
**Purpose**: Real-time operational data from port/terminal operations

**Tables**:
- `vessel` - Vessel reference data (20 vessels)
- `container` - Container tracking with versioning (21 container records)
- `edi_message` - EDI messages (COPARN, COARRI, CODECO, IFTMCS, IFTMIN - 20 messages)
- `api_event` - API operational events (gate, load, discharge - 20 events)
- `vessel_advice` - Vessel arrival advice lifecycle (7 advices)
- `berth_application` - Berth planning (4 applications)

---

## 🔧 Technical Implementation

### 1. Database Models (`app/models/database.py`)

Added operational database models:
```python
class Vessel(Base):
    vessel_id, imo_no, vessel_name, call_sign, operator_name,
    flag_state, capacity_teu, loa_m, beam_m, draft_m,
    last_port, next_port

class Container(Base):
    container_id, cntr_no, iso_code, size_type, status,
    origin_port, tranship_port, destination_port,
    hazard_class, vessel_id, eta_ts, etd_ts, last_free_day
    # Composite PK: (cntr_no, created_at) for versioning

class EDIMessage(Base):
    edi_id, container_id, vessel_id, message_type, direction,
    status, message_ref, sender, receiver, sent_at, ack_at,
    error_text, raw_text

class APIEvent(Base):
    api_id, container_id, vessel_id, event_type, source_system,
    http_status, correlation_id, event_ts, payload_json

class VesselAdvice(Base):
    vessel_advice_no, vessel_name, system_vessel_name,
    effective_start_datetime, effective_end_datetime

class BerthApplication(Base):
    application_no, vessel_advice_no, vessel_close_datetime,
    deleted, berthing_status
```

### 2. Dual Database Configuration (`app/database.py`)

```python
# AI Assistant Database (SQLite)
AI_DATABASE_URL = "sqlite:///./duty_officer_assistant.db"
ai_engine = create_engine(AI_DATABASE_URL)
AISessionLocal = sessionmaker(bind=ai_engine)

# Operational Database (MySQL)
OPS_DATABASE_URL = "mysql+pymysql://root:@localhost/appdb"
ops_engine = create_engine(OPS_DATABASE_URL, pool_pre_ping=True)
OpsSessionLocal = sessionmaker(bind=ops_engine)

# Dependency injection functions
def get_ai_db(): ...
def get_ops_db(): ...
def get_both_dbs(): ...  # Returns {"ai": ai_db, "ops": ops_db}
```

### 3. Operational Data Service (`app/services/operational_data_service.py`)

**Key Methods**:

#### Entity Extraction
```python
extract_identifiers(text) -> Dict[str, List[str]]
```
- Extracts: Container numbers, vessel names, error codes, EDI refs, correlation IDs, IMO numbers
- Uses regex patterns:
  - Containers: `[A-Z]{4}\d{7}` (e.g., CMAU0000020)
  - Vessels: `(?:MV|MS|MT)\s+[A-Z][a-z]+` (e.g., MV Lion City 07)
  - Error codes: `[A-Z]+_(?:ERR|ERROR|WARN)_\d+` (e.g., VESSEL_ERR_4)
  - EDI refs: `REF-[A-Z]+-\d+` (e.g., REF-IFT-0007)

#### Container Analysis
```python
detect_container_duplicates(cntr_no) -> Dict[str, Any]
```
- Checks for composite PK duplication issues
- Detects: Rapid duplicate inserts (< 5 seconds apart), data inconsistencies
- Example output for CMAU0000020:
  ```json
  {
    "has_duplicates": true,
    "count": 2,
    "issue_type": "rapid_duplicate_insert",
    "root_cause": "Multiple inserts within 1.0s - likely race condition or double-submit"
  }
  ```

#### Vessel Analysis
```python
detect_vessel_advice_conflict(vessel_name) -> Dict[str, Any]
```
- Detects VESSEL_ERR_4: Active advice already exists
- Identifies uniqueness constraint violations
- Example output for MV Lion City 07:
  ```json
  {
    "has_conflict": true,
    "error_type": "VESSEL_ERR_4",
    "active_advice_no": 1000010960,
    "root_cause": "Cannot create new advice - vessel already has active advice",
    "solution": "Expire existing advice by setting effective_end_datetime"
  }
  ```

#### EDI Analysis
```python
analyze_edi_error(message_ref) -> Dict[str, Any]
```
- Analyzes EDI ERROR status messages
- Determines root cause from error_text:
  - "Segment missing" → EDI structure incomplete
  - "Validation" → Invalid data format
  - "Timeout" → Message too large or system overload

#### API Event Analysis
```python
detect_api_event_cascade(start, end, cascade_window_seconds=10)
```
- Finds cascading API failures (4xx/5xx responses within 10 seconds)
- Useful for identifying system-wide outages or error propagation

#### Main Correlation Method
```python
correlate_incident(incident_description, incident_time, search_window_hours)
```
- Orchestrates all correlation queries
- Returns comprehensive findings:
  ```json
  {
    "identifiers": {...},
    "time_window": {...},
    "findings": {
      "containers": [...],
      "vessels": [...],
      "edi_messages": [...],
      "edi_errors_in_window": [...],
      "api_events_by_correlation": [...],
      "api_event_cascades": [...]
    }
  }
  ```

### 4. Enhanced RCA Route (`simple_main.py`)

**Changes**:
1. Updated dependency: `dbs: Dict[str, Session] = Depends(get_both_dbs)`
2. Initialize both services: `log_analyzer` + `ops_service`
3. Call `ops_service.correlate_incident()` to query operational data
4. **Enhance hypotheses** with operational findings:
   - Container duplicates → Insert confidence 0.95 hypothesis with DB evidence
   - Vessel advice conflicts → Insert confidence 0.98 hypothesis with solution
   - EDI errors → Insert confidence 0.90 hypothesis with root cause

**Example Enhancement**:
```python
if container["duplication_analysis"]["has_duplicates"]:
    dup = container["duplication_analysis"]
    hypotheses.insert(0, RootCauseHypothesis(
        description=f"Container {cntr_no} duplication: {dup['issue_type']}",
        confidence=0.95,
        evidence=[f"Database shows {dup['count']} records"],
        contributing_factors=["Composite PK", "Race condition"]
    ))
```

### 5. Database Status Page

**Route**: `/database-status`

Displays:
- ✅ **AI Database**: Knowledge base entries, training data, RCA count
- ✅ **Operational Database**: Vessels, containers, EDI messages, API events count
- ⚠️ Graceful degradation: Shows warning if ops DB unavailable

---

## 🔍 Real-World Correlation Examples

### Example 1: Container Duplication (Test Case #1)
**Incident**: "Customer on PORTNET is seeing 2 identical containers information for CMAU0000020"

**Correlation Process**:
1. Extract identifier: `CMAU0000020`
2. Query operational DB: Find 2 records with timestamps 1 second apart
3. Detect duplication: `rapid_duplicate_insert`
4. **Enhanced Root Cause**: "Container CMAU0000020 duplication detected: Multiple inserts within 1.0s - likely race condition or double-submit"
5. **Confidence**: 0.95
6. **Evidence**: "Database shows 2 records for CMAU0000020"
7. **Solution**: "Review application code for concurrent insert handling, add idempotency checks"

### Example 2: Vessel Advice Conflict (Test Case #2)
**Incident**: "VESSEL_ERR_4 when creating vessel advice for MV Lion City 07"

**Correlation Process**:
1. Extract identifier: `MV Lion City 07`
2. Query operational DB: Find active advice #1000010960
3. Detect conflict: `VESSEL_ERR_4` - uniqueness constraint violation
4. **Enhanced Root Cause**: "VESSEL_ERR_4: Cannot create new advice - vessel 'MV Lion City 07' already has active advice #1000010960"
5. **Confidence**: 0.98
6. **Evidence**: "Active vessel advice #1000010960 exists since 2025-10-01T00:00:00"
7. **Solution**: "Expire the existing advice by setting effective_end_datetime before creating new advice"

### Example 3: EDI Error Cascade
**Incident**: "Multiple EDI parsing failures at 12:25"

**Correlation Process**:
1. Query EDI errors in time window (10:25-14:25)
2. Find: REF-IFT-0007 with status ERROR, error_text: "Segment missing"
3. Analyze: Related to container MSCU0000007
4. **Enhanced Root Cause**: "EDI IFTMIN error: EDI message structure incomplete - required segment not found"
5. **Evidence**: "Message REF-IFT-0007: Segment missing"
6. **Solution**: "Verify sender's EDI message template and segment ordering"

---

## 📝 Configuration

### `.env` File
```properties
# AI Assistant Database (SQLite)
AI_DATABASE_URL=sqlite:///./duty_officer_assistant.db

# Operational Database (MySQL)
OPS_DATABASE_URL=mysql+pymysql://root:@localhost/appdb

# Legacy compatibility
DATABASE_URL=sqlite:///./duty_officer_assistant.db

# Azure OpenAI
AZURE_OPENAI_API_KEY=...
AZURE_OPENAI_ENDPOINT=...
```

### MySQL Setup
1. Create database: `CREATE DATABASE appdb;`
2. Import schema: `mysql -u root appdb < db.sql`
3. Verify data: 20 vessels, 21 containers, 20 EDI messages, 20 API events

---

## 🚀 Benefits

### Before Integration
- ❌ Log analysis only
- ❌ No operational data context
- ❌ Generic root cause hypotheses
- ❌ Cannot verify incident details
- ❌ Test cases reference data that doesn't exist in system

### After Integration
- ✅ **Log + Operational Data correlation**
- ✅ **Real container/vessel/EDI context**
- ✅ **Database-backed root cause with 95%+ confidence**
- ✅ **Verifies incident details against actual records**
- ✅ **Test cases now reference real operational data**
- ✅ **Detects data integrity issues** (duplicates, constraint violations)
- ✅ **Traces distributed operations** (correlation IDs, API cascades)

---

## 🧪 Testing the Integration

### Test Scenario 1: Container Duplication
1. Go to `/rca`
2. Enter description: "Customer on PORTNET is seeing 2 identical containers information for CMAU0000020"
3. Set time: 2025-10-19 12:00
4. **Expected Result**: 
   - RCA detects `CMAU0000020` identifier
   - Queries ops DB, finds 2 records
   - Hypothesis: "Container duplication detected: rapid_duplicate_insert"
   - Confidence: 0.95

### Test Scenario 2: Vessel Advice Conflict
1. Go to `/rca`
2. Enter description: "VESSEL_ERR_4 when creating vessel advice for MV Lion City 07"
3. Set time: 2025-10-19 12:00
4. **Expected Result**:
   - RCA detects `MV Lion City 07` and `VESSEL_ERR_4`
   - Queries ops DB, finds active advice #1000010960
   - Hypothesis: "Active advice already exists - uniqueness constraint violation"
   - Confidence: 0.98
   - Solution: "Expire existing advice first"

### Test Scenario 3: EDI Error
1. Go to `/rca`
2. Enter description: "EDI message REF-IFT-0007 parsing failed"
3. Set time: 2025-10-04 12:25
4. **Expected Result**:
   - RCA detects `REF-IFT-0007`
   - Queries ops DB, finds ERROR status with "Segment missing"
   - Hypothesis: "EDI structure incomplete - required segment not found"
   - Confidence: 0.90

### Check Database Status
Visit `/database-status` to verify:
- ✅ AI Database: Connected with X knowledge entries, Y training data
- ✅ Operational Database: Connected with 20 vessels, 21 containers, 20 EDI, 20 API events

---

## 🔮 Future Enhancements

1. **Container Lifecycle Tracking**: Query all status changes for a container
2. **Vessel Schedule Analysis**: Check ETA/ETD consistency with actual berth times
3. **EDI Flow Visualization**: Map complete EDI message sequences (COPARN → COARRI → CODECO)
4. **API Performance Metrics**: Analyze response times and failure rates by source_system
5. **Predictive Analytics**: Use historical patterns to predict incident likelihood
6. **Auto-Remediation**: Trigger database fixes (e.g., expire conflicting vessel advice)

---

## 📚 Related Files

- `app/models/database.py` - All database models (AI + Ops)
- `app/database.py` - Dual database configuration
- `app/services/operational_data_service.py` - Ops data queries (580 lines)
- `simple_main.py` - Enhanced RCA route with correlation
- `app/templates/database_status.html` - Database status UI
- `app/templates/rca_results.html` - RCA results with ops data
- `db.sql` - MySQL schema for operational database
- `SCHEMA_OVERVIEW.md` - Detailed schema documentation
- `.env` - Database connection strings

---

## ✅ Completion Status

**7 of 9 tasks completed:**
- [x] Database schema with operational models
- [x] LogAnalyzerService for log parsing
- [x] RCA form page (rca.html)
- [x] RCA routes in simple_main.py
- [x] Navigation menu updates
- [x] RCA results template
- [x] **Operational database integration** ← Just completed
- [ ] RCA history page
- [ ] End-to-end testing

**Next Steps**: Create RCA history page, then perform full end-to-end testing with real log files and operational data scenarios.
