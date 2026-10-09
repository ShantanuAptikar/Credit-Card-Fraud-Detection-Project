from typing import Any, Dict, Optional
from fastapi import HTTPException

class AppException(HTTPException):
    """
    Base exception for the Fraud Detection API.
    Supports a 'message' argument for consistency with the project's response schema.
    """
    def __init__(
        self,
        status_code: int,
        message: str,
        headers: Optional[Dict[str, Any]] = None
    ) -> None:
        super().__init__(status_code=status_code, detail=message, headers=headers)
        self.message = message
