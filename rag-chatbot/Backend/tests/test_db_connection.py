"""
Test script to verify PostgreSQL database connection.
"""
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'app'))

from db.session import engine, SessionLocal
from db.models import Base
from sqlalchemy import text


def test_connection():
    """Test database connection and create tables."""
    print("="*60)
    print("TESTING POSTGRESQL CONNECTION")
    print("="*60)
    
    try:
        # Test 1: Connection test
        print("\n✓ Step 1: Testing database connection...")
        with engine.connect() as connection:
            result = connection.execute(text("SELECT version();"))
            version = result.fetchone()[0]
            print(f"✓ Connected successfully!")
            print(f"  PostgreSQL version: {version[:50]}...")
        
        # Test 2: Check database name
        print("\n✓ Step 2: Checking database name...")
        with engine.connect() as connection:
            result = connection.execute(text("SELECT current_database();"))
            db_name = result.fetchone()[0]
            print(f"✓ Connected to database: {db_name}")
        
        # Test 3: Create tables
        print("\n✓ Step 3: Creating database tables...")
        Base.metadata.create_all(bind=engine)
        print("✓ Tables created successfully!")
        
        # Test 4: List tables
        print("\n✓ Step 4: Listing created tables...")
        with engine.connect() as connection:
            result = connection.execute(text("""
                SELECT table_name 
                FROM information_schema.tables 
                WHERE table_schema = 'public'
                ORDER BY table_name;
            """))
            tables = result.fetchall()
            for table in tables:
                print(f"  - {table[0]}")
        
        # Test 5: Session test
        print("\n✓ Step 5: Testing session creation...")
        db = SessionLocal()
        db.close()
        print("✓ Session works!")
        
        print("\n" + "="*60)
        print("✅ ALL TESTS PASSED! DATABASE IS READY!")
        print("="*60)
        
        return True
        
    except Exception as e:
        print(f"\n❌ ERROR: {str(e)}")
        print("\nPlease check:")
        print("  1. PostgreSQL is running")
        print("  2. Database 'rag_project' exists")
        print("  3. Username/password in .env are correct")
        return False


if __name__ == "__main__":
    success = test_connection()
    sys.exit(0 if success else 1)
