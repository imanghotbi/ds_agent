import logging
import sys
import os
from logging.handlers import RotatingFileHandler
from ds_agent.config import settings

# Global record factory to ensure 'session_id' always exists on log records
# This affects ALL loggers in the current process.
old_factory = logging.getLogRecordFactory()

def record_factory(*args, **kwargs):
    record = old_factory(*args, **kwargs)
    
    # Default value
    record.session_id = "system"
    
    try:
        import chainlit as cl
        # Try to get session info from Chainlit context
        # Use get_context() as it's the safest way to check if we are in a task
        from chainlit.context import get_context
        ctx = get_context()
        if ctx and hasattr(ctx, "session") and ctx.session:
            session = ctx.session
            session_id = session.id[:8] if session.id else "no-id"
            user_id = "anon"
            if session.user and hasattr(session.user, "identifier") and session.user.identifier:
                user_id = session.user.identifier
            record.session_id = f"{user_id}:{session_id}"
    except Exception:
        # This handles cases where we are outside a Chainlit task context
        pass
        
    return record

logging.setLogRecordFactory(record_factory)

def setup_logger():
    # 1. Get Config
    log_level_str = settings.log_level
    log_file = settings.log_file_path
    max_bytes = settings.log_max_bytes
    backup_count = settings.log_backup_count
    log_level = getattr(logging, log_level_str, logging.INFO)

    # 2. Formatter
    # The [%(session_id)s] part depends on the record_factory above.
    fmt_string = "%(asctime)s - %(levelname)-8s - [%(session_id)s] - %(filename)s:%(lineno)d - %(message)s"
    formatter = logging.Formatter(fmt=fmt_string, datefmt="%Y-%m-%d %H:%M:%S")

    # 3. Configure the Root Logger
    # This is a broad stroke to ensure all logs (even from libs) use our format if they hit the root.
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    
    # Update or add handlers to root
    if not root_logger.handlers:
        console = logging.StreamHandler(sys.stdout)
        root_logger.addHandler(console)
    
    for handler in root_logger.handlers:
        handler.setFormatter(formatter)

    # 4. Create/Get the named Logger for the application
    logger = logging.getLogger("ds_agent")
    logger.setLevel(log_level)
    logger.propagate = True # Allow it to propagate to root so root's handlers handle it

    # Ensure no duplicate handlers on the named logger if we use propagation
    if logger.hasHandlers():
        logger.handlers.clear()

    # 5. Add File Handler specifically for our app (optional, root could handle this too)
    log_dir = os.path.dirname(log_file)
    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir)

    file_handler = RotatingFileHandler(
        log_file, 
        maxBytes=max_bytes, 
        backupCount=backup_count,
        encoding='utf-8'
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    return logger

# Initialize the singleton logger
logger = setup_logger()
logger.info("Logging system fully initialized with global session tracking.")
