"""Unit tests for the v2 JWT authentication dependency."""

from __future__ import annotations

import time
import uuid

import jwt
import pytest
from fastapi import HTTPException
from fastapi.security import HTTPAuthorizationCredentials

_SECRET = "test-jwt-secret-for-unit-tests-only"
_USER_ID = uuid.uuid4()


def _make_token(
    sub: object = None,
    audience: str = "authenticated",
    secret: str = _SECRET,
    exp_offset: int = 3600,
) -> str:
    payload: dict[str, object] = {
        "aud": audience,
        "iat": int(time.time()),
        "exp": int(time.time()) + exp_offset,
        "role": "authenticated",
    }
    if sub is not None:
        payload["sub"] = sub
    return jwt.encode(payload, secret, algorithm="HS256")


async def _call_dep(token: str, secret: str = _SECRET) -> uuid.UUID:
    """Call get_current_user_id with a patched JWT secret."""
    from unittest.mock import patch

    from api.v2.auth import get_current_user_id
    from api.v2.config import v2_settings

    creds = HTTPAuthorizationCredentials(scheme="Bearer", credentials=token)
    with patch.object(v2_settings, "supabase_jwt_secret", secret):
        return await get_current_user_id(credentials=creds)


async def test_valid_token_returns_user_id() -> None:
    token = _make_token(sub=str(_USER_ID))
    result = await _call_dep(token)
    assert result == _USER_ID


async def test_expired_token_raises_401() -> None:
    token = _make_token(sub=str(_USER_ID), exp_offset=-1)
    with pytest.raises(HTTPException) as exc_info:
        await _call_dep(token)
    assert exc_info.value.status_code == 401
    assert "expired" in exc_info.value.detail.lower()


async def test_wrong_signature_raises_401() -> None:
    token = _make_token(sub=str(_USER_ID), secret="wrong-secret")
    with pytest.raises(HTTPException) as exc_info:
        await _call_dep(token)
    assert exc_info.value.status_code == 401


async def test_wrong_audience_raises_401() -> None:
    token = _make_token(sub=str(_USER_ID), audience="anon")
    with pytest.raises(HTTPException) as exc_info:
        await _call_dep(token)
    assert exc_info.value.status_code == 401


async def test_missing_sub_raises_401() -> None:
    # sub is intentionally omitted
    token = _make_token()
    with pytest.raises(HTTPException) as exc_info:
        await _call_dep(token)
    assert exc_info.value.status_code == 401
    assert "sub" in exc_info.value.detail.lower()


async def test_non_uuid_sub_raises_401() -> None:
    token = _make_token(sub="not-a-uuid")
    with pytest.raises(HTTPException) as exc_info:
        await _call_dep(token)
    assert exc_info.value.status_code == 401
    assert "UUID" in exc_info.value.detail


async def test_unconfigured_secret_raises_503() -> None:
    token = _make_token(sub=str(_USER_ID))
    with pytest.raises(HTTPException) as exc_info:
        await _call_dep(token, secret="")
    assert exc_info.value.status_code == 503
