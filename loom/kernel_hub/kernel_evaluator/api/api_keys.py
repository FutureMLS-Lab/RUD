from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from kernel_evaluator.db.ops import revoke_api_key_record
from kernel_evaluator.services.api_keys import ApiPrincipal, ApiRole, create_api_key, require_admin_key

api_keys_router = APIRouter(prefix="/api-keys", tags=["api-keys"])


class CreateApiKeyRequest(BaseModel):
    role: ApiRole


class CreateApiKeyResponse(BaseModel):
    key_id: str
    role: ApiRole
    api_key: str


@api_keys_router.post("")
def create_api_key_route(
    req: CreateApiKeyRequest,
    _principal: ApiPrincipal = Depends(require_admin_key),
) -> CreateApiKeyResponse:
    created = create_api_key(req.role)
    return CreateApiKeyResponse(key_id=created.key_id, role=created.role, api_key=created.api_key)


@api_keys_router.post("/{key_id}/revoke")
def revoke_api_key_route(key_id: str, _principal: ApiPrincipal = Depends(require_admin_key)):
    success = revoke_api_key_record(key_id)
    if success is False:
        raise HTTPException(404, "API key not found")
    return {"status": "revoked"}
