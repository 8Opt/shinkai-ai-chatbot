from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_ignore_empty=True, extra="ignore"
    )
    APP: str
    HOST: str
    PORT: int

    ENVIRONMENT: str

    GOOGLE_API_KEY: str
    HUGGINGFACE_API_KEY: str
    GROQ_API_KEY: str


settings = Settings()
