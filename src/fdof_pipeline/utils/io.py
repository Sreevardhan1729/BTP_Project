from pathlib import Path
import logging

def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p

def get_logger(name: str = "fdof") -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        fmt = logging.Formatter("[%(levelname)s] %(message)s")
        ch.setFormatter(fmt)
        logger.addHandler(ch)
    return logger

