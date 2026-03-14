from pydantic import BaseModel


class SignInResponse(BaseModel):
    access_token: str
    type: str
