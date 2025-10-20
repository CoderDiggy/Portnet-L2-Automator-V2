"""
Script to create the solution_feedback table in duty_officer_assistant.db
"""
import sqlite3
from datetime import datetime

# Connect to the database
conn = sqlite3.connect('duty_officer_assistant.db')
cursor = conn.cursor()

# Create the solution_feedback table
create_table_sql = """
CREATE TABLE IF NOT EXISTS solution_feedback (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    incident_description TEXT NOT NULL,
    solution_description TEXT NOT NULL,
    solution_order INTEGER DEFAULT 1,
    solution_type VARCHAR(50) DEFAULT 'Resolution',
    source_type VARCHAR(50) DEFAULT '',
    knowledge_base_id INTEGER,
    training_data_id INTEGER,
    rca_id INTEGER,
    usefulness_count INTEGER DEFAULT 1,
    marked_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    user_identifier VARCHAR(255) DEFAULT '',
    FOREIGN KEY (knowledge_base_id) REFERENCES knowledge_base(id),
    FOREIGN KEY (training_data_id) REFERENCES training_data(id),
    FOREIGN KEY (rca_id) REFERENCES root_cause_analyses(id)
);
"""

try:
    cursor.execute(create_table_sql)
    conn.commit()
    print("✓ solution_feedback table created successfully!")
    
    # Create indexes for better query performance
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_solution_feedback_kb 
        ON solution_feedback(knowledge_base_id);
    """)
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_solution_feedback_td 
        ON solution_feedback(training_data_id);
    """)
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_solution_feedback_rca 
        ON solution_feedback(rca_id);
    """)
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_solution_feedback_source 
        ON solution_feedback(source_type);
    """)
    conn.commit()
    print("✓ Indexes created successfully!")
    
    # Verify the table was created
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='solution_feedback';")
    result = cursor.fetchone()
    if result:
        print(f"✓ Table verified: {result[0]}")
        
        # Show table schema
        cursor.execute("PRAGMA table_info(solution_feedback);")
        columns = cursor.fetchall()
        print("\nTable Schema:")
        for col in columns:
            print(f"  - {col[1]} ({col[2]})")
    else:
        print("✗ Table creation verification failed!")
        
except sqlite3.Error as e:
    print(f"✗ Error creating table: {e}")
finally:
    conn.close()
    print("\nDatabase connection closed.")
