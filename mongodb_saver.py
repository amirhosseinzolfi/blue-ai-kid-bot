"""
Custom MongoDB checkpoint saver for LangGraph
"""
from typing import Any, Dict, Optional
import json
from datetime import datetime

from pymongo import MongoClient
from pymongo.collection import Collection
from pymongo.database import Database
from pymongo.errors import ConnectionFailure, ServerSelectionTimeoutError

from langgraph.checkpoint import Checkpoint
import logging

logger = logging.getLogger(__name__)

class MongoDBSaver(Checkpoint):
    """A MongoDB-based checkpoint saver for LangGraph."""
    
    def __init__(self, collection: Collection):
        """Initialize with MongoDB collection."""
        self.collection = collection
        
    @classmethod
    def from_conn_string(cls, conn_string: str, db_name: str, collection_name: str):
        """Create a MongoDB saver from a connection string."""
        try:
            client = MongoClient(conn_string, serverSelectionTimeoutMS=5000)
            # Verify the connection works
            client.admin.command('ping')
            logger.info("Successfully connected to MongoDB for checkpointing")
            db: Database = client[db_name]
            collection: Collection = db[collection_name]
            return cls(collection)
        except (ConnectionFailure, ServerSelectionTimeoutError) as e:
            logger.error(f"Failed to connect to MongoDB for checkpoints: {e}")
            raise
            
    def get(self, key: str, config: Optional[Dict] = None) -> Optional[Dict[str, Any]]:
        """Get checkpoint for key."""
        thread_id = self._get_thread_id(config)
        query = {"key": key}
        if thread_id:
            query["thread_id"] = thread_id
            
        result = self.collection.find_one(query)
        if result:
            # Convert BSON to dict and remove MongoDB _id
            checkpoint_data = result.get("checkpoint_data")
            if isinstance(checkpoint_data, str):
                # If stored as JSON string, parse it
                return json.loads(checkpoint_data)
            return checkpoint_data
        return None
    
    def put(self, key: str, state: Dict[str, Any], config: Optional[Dict] = None) -> None:
        """Save checkpoint for key."""
        thread_id = self._get_thread_id(config)
        query = {"key": key}
        if thread_id:
            query["thread_id"] = thread_id
            
        # Store data with timestamp
        update_data = {
            "$set": {
                "checkpoint_data": state,
                "updated_at": datetime.now()
            }
        }
        
        # Use upsert to create if doesn't exist
        self.collection.update_one(query, update_data, upsert=True)
    
    def delete(self, key: str = None, config: Optional[Dict] = None) -> None:
        """Delete checkpoint(s)."""
        thread_id = self._get_thread_id(config)
        
        query = {}
        if key:
            query["key"] = key
        if thread_id:
            query["thread_id"] = thread_id
            
        # If both key and thread_id are None, we're deleting everything
        # Add confirmation check in that case
        if not query:
            logger.warning("Deleting ALL checkpoints from MongoDB collection")
            
        self.collection.delete_many(query)
    
    def _get_thread_id(self, config: Optional[Dict]) -> Optional[str]:
        """Extract thread_id from config."""
        if not config:
            return None
        
        try:
            return config.get("configurable", {}).get("thread_id")
        except (AttributeError, TypeError):
            return None
    
    def __enter__(self):
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        pass
