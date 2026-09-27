import uvicorn
from fastapi import FastAPI
from app.auth.routes import auth, user
from app.classification.routes import classify_router
from app.xai.routes import xai_router
from app.experiments.routes import experiment_router
from fastapi.middleware.cors import CORSMiddleware
from app.config import origins

app = FastAPI(
    title="Skin Disease Classifier with XAI",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth.auth_router, prefix="/api/auth", tags=["Auth"])
app.include_router(user.user_router, prefix="/api/users", tags=["Users"])
app.include_router(classify_router, prefix="/api/classify", tags=["Classification"])
app.include_router(xai_router, prefix="/api/xai", tags=["XAI"])
app.include_router(experiment_router, prefix="/api/diagnostics", tags=["Diagnostics research"])
app.include_router(experiment_router, prefix="/api/experiments", tags=["Experiments"])


if __name__ == '__main__':
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
