"""
Script to query solution_feedback table to see what solutions users found useful
"""
import sqlite3
from datetime import datetime

# Connect to the database
conn = sqlite3.connect('duty_officer_assistant.db')
cursor = conn.cursor()

try:
    print("\n" + "="*80)
    print("SOLUTION FEEDBACK SUMMARY")
    print("="*80 + "\n")
    
    # Get total feedback count
    cursor.execute("SELECT COUNT(*) FROM solution_feedback")
    total = cursor.fetchone()[0]
    print(f"Total feedback entries: {total}\n")
    
    if total > 0:
        # Get feedback grouped by source type
        print("Feedback by Source Type:")
        print("-" * 80)
        cursor.execute("""
            SELECT source_type, COUNT(*), SUM(usefulness_count)
            FROM solution_feedback
            GROUP BY source_type
            ORDER BY SUM(usefulness_count) DESC
        """)
        for row in cursor.fetchall():
            print(f"  {row[0]:<20} | Entries: {row[1]:<5} | Total Useful Marks: {row[2]}")
        
        # Get most useful solutions
        print("\n\nMost Useful Solutions (Top 10):")
        print("-" * 80)
        cursor.execute("""
            SELECT 
                solution_description,
                source_type,
                usefulness_count,
                marked_at
            FROM solution_feedback
            ORDER BY usefulness_count DESC
            LIMIT 10
        """)
        for i, row in enumerate(cursor.fetchall(), 1):
            print(f"\n{i}. [{row[1]}] Usefulness: {row[2]}")
            print(f"   Solution: {row[0][:100]}{'...' if len(row[0]) > 100 else ''}")
            print(f"   Last marked: {row[3]}")
        
        # Get recent feedback
        print("\n\nRecent Feedback (Last 10):")
        print("-" * 80)
        cursor.execute("""
            SELECT 
                incident_description,
                solution_description,
                source_type,
                usefulness_count,
                marked_at
            FROM solution_feedback
            ORDER BY marked_at DESC
            LIMIT 10
        """)
        for i, row in enumerate(cursor.fetchall(), 1):
            print(f"\n{i}. Problem: {row[0][:80]}{'...' if len(row[0]) > 80 else ''}")
            print(f"   Solution: {row[1][:80]}{'...' if len(row[1]) > 80 else ''}")
            print(f"   Source: {row[2]} | Usefulness: {row[3]} | Marked: {row[4]}")
    else:
        print("No feedback data yet. Use the 'Useful' button when analyzing incidents!")
        
except sqlite3.Error as e:
    print(f"Error querying database: {e}")
finally:
    conn.close()
    print("\n" + "="*80 + "\n")
