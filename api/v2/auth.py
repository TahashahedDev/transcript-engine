"""FastAPI dependency that validates Supabase JWT tokens."""

from __future__ import annotations

import uuid

import jwt
from fastapi import HTTPException, Security
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from api.v2.config import v2_settings

_bearer = HTTPBearer(auto_error=True)


async def get_current_user_id(
    credentials: HTTPAuthorizationCredentials = Security(_bearer),
) -> uuid.UUID:
    """Extract and validate a Supabase JWT; return the authenticated user's UUID."""
    if not v2_settings.supabase_jwt_secret:
        raise HTTPException(status_code=503, detail="Authentication not configured")

    try:
        payload = jwt.decode(
            credentials.credentials,
            v2_settings.supabase_jwt_secret,
            algorithms=["HS256"],
            audience="authenticated",
        )
    except jwt.ExpiredSignatureError as exc:
        raise HTTPException(status_code=401, detail="Token has expired") from exc
    except jwt.InvalidTokenError as exc:
        raise HTTPException(status_code=401, detail=f"Invalid token: {exc}") from exc

    sub = payload.get("sub")
    if not sub:
        raise HTTPException(status_code=401, detail="Token is missing the sub claim")

    try:
        return uuid.UUID(str(sub))
    except ValueError as exc:
        raise HTTPException(status_code=401, detail="Token sub claim is not a valid UUID") from exc
