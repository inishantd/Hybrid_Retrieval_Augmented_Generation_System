from pathlib import Path
from pydantic import Field, model_validator, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    #Paths
    BASE_DIR: Path = Path(__file__).resolve().parent.parent
    DATA_DIR: Path = BASE_DIR / "data"
    CHROMA_STORAGE_DIR: Path = DATA_DIR / "chroma_db"
    SPARSE_INDEX_PATH: Path = DATA_DIR / "sparse_index.pkl"


    #Models
    EMBEDDING_MODEL_NAME: str = "all-MiniLM-L6-v2"
    RERANKER_MODEL_NAME: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    GENERATION_MODEL_NAME: str = "openai/gpt-oss-20b"
    JUDGE_MODEL_NAME: str = "openai/gpt-oss-20b"

    #Retrival/RAG
    CHUNK_SIZE: int = Field(default=1500, gt = 0)
    CHUNK_OVERLAP: int = Field(default=300, ge=0)
    RETRIEVAL_TOP_K: int = Field(default=10, gt=0)
    RERANK_TOP_N: int = Field(default=5, gt=0)

    #Security 
    GROQ_API_KEY: str


    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8"
    )

    #Cross-field Validation 
    @model_validator(mode="after")
    def validate_rag_parameters(self):

        if self.CHUNK_OVERLAP >= self.CHUNK_SIZE:
            raise ValueError(
                "CHUNK_OVERLAP must be smaller than CHUNK_SIZE"
            )

        if self.RERANK_TOP_N > self.RETRIEVAL_TOP_K:
            raise ValueError(
                "RERANK_TOP_N cannot be greater than RETRIEVAL_TOP_K"
            )

        return self

    #API Key validator
    @field_validator("GROQ_API_KEY")
    @classmethod
    def validate_groq_api_key(cls, value: str) -> str:
        if not value or not value.strip():
            raise ValueError("GROQ_API_KEY is missing or empty")

        return value.strip()
config = Settings()