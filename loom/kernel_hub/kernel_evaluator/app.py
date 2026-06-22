from fastapi import FastAPI

from kernel_evaluator.api.api_keys import api_keys_router
from kernel_evaluator.api.evaluation import evaluation_router
from kernel_evaluator.api.kernels import kernels_router
from kernel_evaluator.api.scaffold import scaffold_router

app = FastAPI()
app.include_router(api_keys_router)
app.include_router(evaluation_router)
app.include_router(kernels_router)
app.include_router(scaffold_router)
