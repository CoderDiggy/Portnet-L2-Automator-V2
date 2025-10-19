#!/usr/bin/env python3
"""
CSV to SQLite Converter for Knowledge Base
This script converts KnowledgeBase.csv to SQLite database using the existing schema
"""

import pandas as pd
import sqlite3
from datetime import datetime
import os
import sys
from pathlib import Path

# Add the app directory to Python path so we can import the database models
sys.path.append(os.path.join(os.path.dirname(__file__), 'app'))

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from app.models.database import Base, KnowledgeBase
from app.database import engine

def convert_csv_to_sqlite():
    """Convert KnowledgeBase.csv to SQLite database"""
    
    print("🔄 Starting CSV to SQLite conversion...")
    
    # Check if CSV file exists
    csv_file = "KnowledgeBase.csv"
    if not os.path.exists(csv_file):
        print(f"❌ Error: {csv_file} not found in current directory")
        return False
    
    try:
        # Read CSV file with encoding detection
        print(f"📖 Reading {csv_file}...")
        
        # Try different encodings
        encodings = ['utf-8', 'utf-16', 'iso-8859-1', 'cp1252', 'latin1']
        df = None
        
        for encoding in encodings:
            try:
                print(f"🔄 Trying encoding: {encoding}")
                df = pd.read_csv(csv_file, encoding=encoding)
                print(f"✅ Successfully read with {encoding} encoding")
                break
            except UnicodeDecodeError:
                continue
        
        if df is None:
            print("❌ Could not read CSV with any standard encoding")
            return False
            
        print(f"✅ Found {len(df)} records in CSV")
        
        # Display CSV structure
        print("\n📊 CSV Structure:")
        print(f"Columns: {list(df.columns)}")
        print(f"First few rows:")
        print(df.head(2))
        
        # Create database tables if they don't exist
        print("\n🗄️ Creating database tables...")
        Base.metadata.create_all(bind=engine)
        
        # Create session
        Session = sessionmaker(bind=engine)
        session = Session()
        
        # Clear existing knowledge base entries (optional - comment out if you want to keep existing data)
        print("\n🧹 Clearing existing knowledge base entries...")
        session.query(KnowledgeBase).delete()
        session.commit()
        
        # Convert and insert records
        print("\n💾 Converting records to database...")
        success_count = 0
        error_count = 0
        
        for index, row in df.iterrows():
            try:
                # Extract data from CSV row - map CSV columns to database fields
                title = str(row.get('Case Title', '')).strip()
                overview = str(row.get('Overview', '')).strip()
                resolution = str(row.get('Resolution', '')).strip()
                verification = str(row.get('Verification', '')).strip()
                module = str(row.get('Module', 'General')).strip()
                
                # Combine overview, resolution, and verification into content
                content_parts = []
                if overview and overview.lower() not in ['nan', 'none', '']:
                    content_parts.append(f"Overview:\n{overview}")
                if resolution and resolution.lower() not in ['nan', 'none', '']:
                    content_parts.append(f"Resolution:\n{resolution}")
                if verification and verification.lower() not in ['nan', 'none', '']:
                    content_parts.append(f"Verification:\n{verification}")
                
                content = '\n\n'.join(content_parts)
                
                # Map other fields
                category = module if module and module.lower() not in ['nan', 'none', ''] else 'General'
                doc_type = 'case'  # This appears to be case documentation
                tags = module.lower().replace(' ', '_') if module else ''
                keywords = title.lower()  # Use title words as keywords
                source = 'CSV Import - Knowledge Base'
                
                # Skip empty records
                if not title or title.lower() in ['nan', 'none', '']:
                    error_count += 1
                    print(f"⚠️ Skipping row {index + 1}: Empty title")
                    continue
                
                if not content or content.lower() in ['nan', 'none', '']:
                    error_count += 1
                    print(f"⚠️ Skipping row {index + 1}: Empty content")
                    continue
                
                # Create KnowledgeBase entry
                kb_entry = KnowledgeBase(
                    title=title,
                    content=content,
                    category=category,
                    type=doc_type,
                    tags=tags,
                    keywords=keywords,
                    source=source,
                    usefulness_count=0,  # Start with 0
                    created_at=datetime.now(),
                    updated_at=datetime.now()
                )
                
                session.add(kb_entry)
                success_count += 1
                
                if success_count % 10 == 0:
                    print(f"✅ Processed {success_count} records...")
                
            except Exception as ex:
                error_count += 1
                print(f"❌ Error processing row {index + 1}: {str(ex)}")
                continue
        
        # Commit all changes
        session.commit()
        
        # Display results
        print(f"\n🎉 Conversion completed!")
        print(f"✅ Successfully imported: {success_count} records")
        print(f"❌ Errors encountered: {error_count} records")
        
        # Verify the import
        total_in_db = session.query(KnowledgeBase).count()
        print(f"📊 Total records now in database: {total_in_db}")
        
        # Show some sample entries
        print(f"\n📋 Sample entries from database:")
        sample_entries = session.query(KnowledgeBase).limit(3).all()
        for i, entry in enumerate(sample_entries, 1):
            print(f"\n{i}. Title: {entry.title[:60]}...")
            print(f"   Category: {entry.category}")
            print(f"   Content: {entry.content[:100]}...")
        
        session.close()
        
        return True
        
    except Exception as ex:
        print(f"❌ Fatal error during conversion: {str(ex)}")
        return False

def verify_database():
    """Verify the database contents"""
    try:
        print("\n🔍 Verifying database contents...")
        
        Session = sessionmaker(bind=engine)
        session = Session()
        
        # Count records by category
        print("\n📊 Records by category:")
        categories = session.query(KnowledgeBase.category, 
                                 session.query(KnowledgeBase).filter_by(category=KnowledgeBase.category).count().label('count'))\
                           .group_by(KnowledgeBase.category).all()
        
        for category in session.query(KnowledgeBase.category).distinct().all():
            count = session.query(KnowledgeBase).filter_by(category=category[0]).count()
            print(f"  - {category[0]}: {count} records")
        
        session.close()
        
    except Exception as ex:
        print(f"❌ Error verifying database: {str(ex)}")

if __name__ == "__main__":
    print("=" * 60)
    print("🗃️  CSV to SQLite Knowledge Base Converter")
    print("=" * 60)
    
    # Change to the script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    print(f"📂 Working directory: {os.getcwd()}")
    
    # Run conversion
    success = convert_csv_to_sqlite()
    
    if success:
        verify_database()
        print("\n🎉 Conversion completed successfully!")
        print("💡 You can now use the web interface to view the knowledge base at http://localhost:8002/knowledge")
    else:
        print("\n❌ Conversion failed. Please check the error messages above.")
    
    print("\n" + "=" * 60)