import psycopg2
from psycopg2 import pool
from contextlib import contextmanager
from utils.config_loader import GLOBAL_CONFIG
from utils.logger import get_logger

logger = get_logger(__name__)

class DatabaseManager:
    """Manages PostgreSQL connection pooling for the entire application."""
    
    _instance = None

    def __new__(cls):
        """Singleton pattern to ensure only one connection pool exists."""
        if cls._instance is None:
            cls._instance = super(DatabaseManager, cls).__new__(cls)
            cls._instance._initialize_pool()
        return cls._instance

    def _initialize_pool(self):
        """Initializes the connection pool using config credentials."""
        # Note: If your config is nested under 'retriever', adjust the get() path
        db_config = GLOBAL_CONFIG.get("database", {})
        
        try:
            # Create a pool with a minimum of 1 and maximum of 20 connections
            self.pool = psycopg2.pool.SimpleConnectionPool(
                1, 20,
                dbname=db_config.get("dbname", "rag_database"),
                user=db_config.get("user", "postgres"),
                password=db_config.get("password", "mysecretpassword"),
                host=db_config.get("host", "db"), # 'db' for Docker network
                port=db_config.get("port", 5432)
            )
            logger.info("Database connection pool initialized successfully.")
        except Exception as e:
            logger.error(f"Failed to initialize database pool: {e}")
            raise

    @contextmanager
    def get_connection(self):
        """
        Yields a database connection from the pool and ensures it is safely released.
        Usage: with db_manager.get_connection() as conn: ...
        """
        conn = self.pool.getconn()
        try:
            yield conn
        finally:
            self.pool.putconn(conn)