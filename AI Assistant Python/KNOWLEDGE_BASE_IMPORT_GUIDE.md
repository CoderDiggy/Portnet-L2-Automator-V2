# Knowledge Base CSV Import Guide

## Overview

This guide explains how to replace your existing knowledge base database with data from your `KnowledgeBase.csv` file. Your CSV contains incident case resolutions that will be imported into the SQLite database used by your AI Assistant.

## Your Problem Statement

Your application is an **AI Duty Officer Assistant** that:
- Analyzes incidents using Azure OpenAI
- Provides resolution plans based on a knowledge base
- Learns from training data
- Uses SQLite database to store knowledge entries

You discovered that your **knowledge base database is missing some data**, and you have the complete data in `KnowledgeBase.csv` (1127 entries).

## CSV Structure

Your CSV has the following columns:
- **Case Title**: The incident title (e.g., "CNTR:Container Range Error")
- **Module**: The category (e.g., "Container Report", "Container Booking")
- **Overview**: Problem description
- **Resolution**: Step-by-step resolution procedures
- **Verification**: How to verify the fix worked

## Import Scripts

Two scripts have been created for you:

### 1. `import_knowledge_csv.py` (Interactive - Recommended)
- **Features**:
  - Prompts for confirmation before deleting existing data
  - Shows detailed progress and statistics
  - Displays sample of imported entries
  - Best for manual, careful imports

### 2. `import_knowledge_quick.py` (Non-Interactive)
- **Features**:
  - No prompts - runs immediately
  - Faster execution
  - Good for automation or batch processing

## How to Use

### Step 1: Backup Your Current Database (Optional but Recommended)

Before running the import, backup your current database:

```powershell
# In PowerShell
Copy-Item "duty_officer_assistant.db" "duty_officer_assistant.db.backup"
```

### Step 2: Run the Import Script

#### Option A: Interactive Import (Recommended)
```powershell
python import_knowledge_csv.py
```

You'll see:
```
About to delete X existing knowledge base entries.
Are you sure you want to continue? (yes/no):
```

Type `yes` and press Enter to proceed.

#### Option B: Quick Import (No Prompts)
```powershell
python import_knowledge_quick.py
```

This runs immediately without confirmation.

### Step 3: Verify the Import

The script will show:
- Number of entries imported
- Sample of imported data
- Any errors encountered

Example output:
```
======================================================================
Import completed successfully!
  Total entries in CSV: 1127
  Successfully imported: 1127
  Skipped: 0
======================================================================
```

## What the Script Does

1. **Creates Database Tables**: Ensures the knowledge_base table exists
2. **Clears Old Data**: Deletes existing knowledge base entries (optional)
3. **Reads CSV**: Parses your KnowledgeBase.csv file
4. **Processes Each Entry**:
   - Combines Overview, Resolution, and Verification into structured content
   - Extracts keywords automatically
   - Determines priority (High/Medium/Low) based on error keywords
   - Sets appropriate type (Solution/Procedure/FAQ)
   - Uses Module as the category
5. **Imports to Database**: Inserts all entries into SQLite database

## Data Transformation

Each CSV row is transformed as follows:

```
CSV Row:
  Case Title: "CNTR:Container Range Error"
  Module: "Container Report"
  Overview: "Problem description..."
  Resolution: "1. Step one\n2. Step two..."
  Verification: "1. Verify step..."

Database Entry:
  title: "CNTR:Container Range Error"
  category: "Container Report"
  type: "Solution"
  priority: 3 (High - because contains "error")
  content: 
    ## Problem Overview
    Problem description...
    
    ## Resolution Steps
    1. Step one
    2. Step two...
    
    ## Verification
    1. Verify step...
  keywords: "container, range, error, overlap, serial, ..." (auto-extracted)
  status: "Active"
  source: "CSV Import"
```

## Priority Assignment

The script automatically assigns priority based on keywords:

- **High Priority (3)**: Contains "error", "failed", "critical", "timeout", "rejected", "500"
- **Medium Priority (2)**: Contains "discrepancy", "mismatch", "issue", "problem"
- **Default (2)**: Everything else gets Medium priority

## Type Assignment

- **Solution**: Most entries (default)
- **Procedure**: If title contains "procedure" or "how to"
- **FAQ**: If title contains "?"

## Troubleshooting

### Error: "CSV file not found"
**Solution**: Make sure `KnowledgeBase.csv` is in the same directory as the script.

### Error: "Missing required columns"
**Solution**: Ensure your CSV has all columns: Case Title, Module, Overview, Resolution, Verification

### Database is locked
**Solution**: Close any applications using the database (including your web app) before running the import.

### Import shows 0 entries
**Solution**: Check that your CSV has data and the first row contains column headers.

## After Import

Once the import is complete:

1. **Start your application**:
   ```powershell
   python simple_main.py
   ```
   or
   ```powershell
   .\run_website.bat
   ```

2. **Test the knowledge base**:
   - Go to http://localhost:8000
   - Submit an incident analysis
   - The AI should now use your updated knowledge base for recommendations

## Verifying the Data

To verify data was imported correctly, you can:

1. **Check database directly**:
   ```powershell
   python -c "from app.database import SessionLocal; from app.models.database import KnowledgeBase; db = SessionLocal(); print(f'Total entries: {db.query(KnowledgeBase).count()}'); db.close()"
   ```

2. **Use the web interface**:
   - Navigate to the knowledge base section
   - Search for specific cases
   - Verify the content is correct

## Re-running the Import

If you need to re-import (e.g., after updating the CSV):

1. Simply run the script again
2. It will clear the old data and import fresh data
3. Your database will be completely replaced with CSV data

## Preserving Usefulness Counts

⚠️ **Warning**: The current scripts will reset all usefulness counts and view counts to 0.

If you want to preserve these statistics, you would need to:
1. Export the current usefulness data before import
2. Modify the script to merge the data instead of replacing
3. This is more complex and not included in the basic scripts

## Summary

✅ **You now have two scripts to replace your knowledge base:**
- `import_knowledge_csv.py` - Interactive with confirmations
- `import_knowledge_quick.py` - Quick automated import

✅ **Your 1127 CSV entries will be:**
- Properly structured with sections
- Automatically categorized
- Keyword-indexed for better search
- Priority-ranked for relevance

✅ **Your AI Assistant will:**
- Use the complete knowledge base
- Provide better recommendations
- Have all missing data restored

Run the import script, restart your application, and your knowledge base will be fully updated! 🚀
