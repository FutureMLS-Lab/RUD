import hashlib
import os
import secrets
from collections import defaultdict
from dataclasses import dataclass, field
from enum import StrEnum
from time import monotonic

from fastapi import Depends, HTTPException, Security
from fastapi.security import APIKeyHeader


class ApiRole(StrEnum):
    ADMIN = "admin"
    USER = "user"


@dataclass(frozen=True)
class ApiPrincipal:
    key_id: str
    role: ApiRole


@dataclass(frozen=True)
class CreatedApiKey:
    key_id: str
    role: ApiRole
    api_key: str


@dataclass
class RateLimitBucket:
    timestamps: list[float] = field(default_factory=list)


USER_SUBMIT_LIMIT = int(os.environ["KERNEL_EVALUATOR_USER_SUBMIT_LIMIT"])
USER_SUBMIT_WINDOW_S = float(os.environ["KERNEL_EVALUATOR_USER_SUBMIT_WINDOW_S"])
USER_IN_FLIGHT_SUBMIT_LIMIT = int(os.environ["KERNEL_EVALUATOR_USER_IN_FLIGHT_SUBMIT_LIMIT"])
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)
submission_rate_limits: defaultdict[str, RateLimitBucket] = defaultdict(RateLimitBucket)


def digest_api_key(api_key: str) -> str:
    return hashlib.sha256(api_key.encode("utf-8")).hexdigest()


def create_api_key(role: ApiRole) -> CreatedApiKey:
    from kernel_evaluator.db.ops import create_api_key_record

    api_key = f"tkg_{secrets.token_urlsafe(32)}"
    key_hash = digest_api_key(api_key)
    record = create_api_key_record(key_hash=key_hash, role=str(role))
    return CreatedApiKey(key_id=record.key_id, role=ApiRole(record.role), api_key=api_key)


def require_api_key(api_key: str | None = Security(api_key_header)) -> ApiPrincipal:
    from kernel_evaluator.db.ops import get_active_api_key_by_hash, mark_api_key_used

    if api_key is None:
        raise HTTPException(status_code=401, detail="missing api key")
    record = get_active_api_key_by_hash(digest_api_key(api_key))
    if record is None:
        raise HTTPException(status_code=401, detail="invalid api key")
    mark_api_key_used(record.key_id)
    return ApiPrincipal(key_id=record.key_id, role=ApiRole(record.role))


def require_admin_key(principal: ApiPrincipal = Depends(require_api_key)) -> ApiPrincipal:
    if principal.role != ApiRole.ADMIN:
        raise HTTPException(status_code=403, detail="admin api key required")
    return principal


def check_submission_rate_limit(principal: ApiPrincipal, now: float | None = None) -> None:
    if principal.role == ApiRole.ADMIN:
        return
    current_time = monotonic() if now is None else now
    window_start = current_time - USER_SUBMIT_WINDOW_S
    bucket = submission_rate_limits[principal.key_id]
    bucket.timestamps = [timestamp for timestamp in bucket.timestamps if timestamp > window_start]
    if len(bucket.timestamps) >= USER_SUBMIT_LIMIT:
        raise HTTPException(status_code=429, detail="submission rate limit exceeded")
    bucket.timestamps.append(current_time)


def check_in_flight_submission_limit(principal: ApiPrincipal, in_flight_jobs: int) -> None:
    if principal.role == ApiRole.ADMIN:
        return
    if in_flight_jobs >= USER_IN_FLIGHT_SUBMIT_LIMIT:
        raise HTTPException(status_code=429, detail="in-flight submission limit exceeded")


def require_submission_key(principal: ApiPrincipal = Depends(require_api_key)) -> ApiPrincipal:
    check_submission_rate_limit(principal)
    return principal
