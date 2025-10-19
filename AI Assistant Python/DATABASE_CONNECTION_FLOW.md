# 📊 Database Connection Flow - Where OpenAI Services Get Data

## The Database File

### 📁 Database File Name:
```
duty_officer_assistant.db
```

### 📍 Location:
```
C:\Users\TanJy\Downloads\Portnet-L2-Automator\AI Assistant Python\duty_officer_assistant.db
```

### 🔧 Configuration:
Defined in `.env` file:
```properties
DATABASE_URL=sqlite:///./duty_officer_assistant.db
```

---

## How OpenAI Services Access the Database

### Flow Diagram:

```
┌─────────────────────────────────────────────────────────────┐
│  1. User makes RCA request                                  │
│     http://localhost:8002/rca/analyze                       │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│  2. simple_main.py - RCA Route Handler                      │
│     @app.post("/rca/analyze")                               │
│     async def analyze_rca(db: Session = Depends(get_db))    │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│  3. app/database.py - Database Session Factory              │
│                                                              │
│     def get_db():                                           │
│         db = SessionLocal()  ← Creates session              │
│         yield db             ← Provides to route            │
│         db.close()           ← Cleans up                    │
│                                                              │
│     SessionLocal = sessionmaker(bind=engine)                │
│                                                              │
│     engine = create_engine(                                 │
│         "sqlite:///./duty_officer_assistant.db"             │
│     )                                                        │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│  4. Service Layer - Receives DB Session                     │
│                                                              │
│     training_service = TrainingDataService(db)              │
│     ├─ __init__(self, db: Session)                          │
│     │   self.db = db  ← Stores the session                 │
│     └─ find_relevant_examples_async()                       │
│         └─ self.db.query(TrainingData)... ← Uses session    │
│                                                              │
│     kb_service = KnowledgeBaseService(db)                   │
│     ├─ __init__(self, db: Session)                          │
│     │   self.db = db  ← Stores the session                 │
│     └─ search_knowledge()                                   │
│         └─ self.db.query(KnowledgeBase)... ← Uses session   │
│                                                              │
│     openai_service = OpenAIService()                        │
│     └─ Uses training_service and kb_service                 │
│         └─ They query the database                          │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│  5. SQLite Database File                                    │
│     📁 duty_officer_assistant.db                            │
│                                                              │
│     Tables:                                                 │
│     ├─ training_data (323 rows) ← AI searches here         │
│     ├─ knowledge_base (152 rows) ← AI searches here        │
│     ├─ vessel (20 rows)                                     │
│     ├─ container (12 rows)                                  │
│     ├─ edi_message (0 rows)                                 │
│     ├─ api_event (0 rows)                                   │
│     ├─ root_cause_analyses (4 rows)                         │
│     └─ system_logs (0 rows)                                 │
└─────────────────────────────────────────────────────────────┘
```

---

## Key Files Involved

### 1. `.env` - Database Configuration
```properties
DATABASE_URL=sqlite:///./duty_officer_assistant.db
```
**Purpose**: Tells the app which database file to use

### 2. `app/database.py` - Database Engine
```python
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./duty_officer_assistant.db")
engine = create_engine(DATABASE_URL, ...)
SessionLocal = sessionmaker(bind=engine)
```
**Purpose**: Creates the SQLAlchemy engine pointing to the database file

### 3. `simple_main.py` - Route Handler
```python
@app.post("/rca/analyze")
async def analyze_rca(db: Session = Depends(get_db)):
    training_service = TrainingDataService(db)  # Pass db session
    similar_incidents = await training_service.find_relevant_examples_async(...)
```
**Purpose**: Gets database session and passes to services

### 4. `app/services/training_data_service.py` - Data Access
```python
class TrainingDataService:
    def __init__(self, db: Session):
        self.db = db  # Store the session
    
    async def find_relevant_examples_async(self, query: str):
        # Query the database
        all_training = self.db.query(TrainingData).filter(
            TrainingData.is_validated == 1
        ).all()
```
**Purpose**: Executes SQL queries against the database

### 5. `app/services/openai_service.py` - AI Service
```python
def analyze_incident(self, incident_description: str, db=None):
    # Get training data from database
    if db:
        td_service = TrainingDataService(db)
        training_examples = await td_service.find_relevant_examples_async(...)
```
**Purpose**: Uses other services that query the database

---

## Database Query Flow for RCA

### When you submit an RCA:

```
User enters: "Container timestamp showing wrong timezone"
       ↓
1. simple_main.py gets db session from get_db()
       ↓
2. TrainingDataService(db) created
       ↓
3. find_relevant_examples_async() called
       ↓
4. SQL Query executed:
   SELECT * FROM training_data 
   WHERE is_validated = 1
       ↓
5. 323 training records retrieved from duty_officer_assistant.db
       ↓
6. Semantic search finds similar incidents
       ↓
7. Returns: Similar incident about "timezone drift eventTime UTC+0 vs UTC+8"
       ↓
8. Root cause generated from matching incident's solution
       ↓
9. Results shown with 85% confidence
```

---

## To Verify Database Contents

### Option 1: Use SQLite Browser
1. Download: https://sqlitebrowser.org/
2. Open: `duty_officer_assistant.db`
3. Browse tables: training_data, knowledge_base, etc.

### Option 2: Python Script
```python
from app.database import SessionLocal
from app.models.database import TrainingData, KnowledgeBase

db = SessionLocal()

# Count records
print(f"Training Data: {db.query(TrainingData).count()}")
print(f"Knowledge Base: {db.query(KnowledgeBase).count()}")
print(f"Validated Training: {db.query(TrainingData).filter(TrainingData.is_validated==1).count()}")

db.close()
```

### Option 3: Use check_database_setup.py
```bash
python check_database_setup.py
```

---

## Summary

### 🎯 Database File:
- **Name**: `duty_officer_assistant.db`
- **Location**: Same folder as `simple_main.py`
- **Type**: SQLite (single file database)

### 🔄 How OpenAI Services Access It:
1. Route gets `db` session from `get_db()` dependency
2. Creates service instances: `TrainingDataService(db)`, `KnowledgeBaseService(db)`
3. Services execute queries: `self.db.query(TrainingData).all()`
4. SQLAlchemy reads from: `duty_officer_assistant.db` file
5. Returns: 323 training cases, 152 KB entries

### ✅ What's Inside:
- 323 validated training examples (past incidents + solutions)
- 152 knowledge base entries (SOPs, procedures)
- 20 vessels, 12 containers (operational data)
- All in ONE SQLite file!

---

**The OpenAI services access the database through the `db` session parameter passed from `simple_main.py`, which connects to `duty_officer_assistant.db` SQLite file!**
