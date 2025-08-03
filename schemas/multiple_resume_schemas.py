from pydantic import BaseModel
from typing import Optional


class MultipleResumeUploadRequest(BaseModel):
    """
    Request model for multiple resume upload with user details.
    """

    user_id: str  # Required - the user ID who is uploading the resumes
    username: str  # Required - the username/name of the user

    # Optional parameters for bulk processing
    max_concurrent: Optional[int] = 10  # Maximum concurrent threads
    batch_size: Optional[int] = 20  # Database batch size
