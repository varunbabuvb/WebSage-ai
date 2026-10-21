from pydantic import BaseModel, Field
class AskRequest(BaseModel):
    question : str = Field(min_length = 1 ,max_length= 1000,description="The question to be asked", example="what is this page summary?")
    page_content : str = Field(min_length = 1,max_length=100000 ,description="web page text scrapped from the web url")
    page_url : str = Field(min_length = 1,max_length = 2048 ,description = "url of the page given by user",example = "https://www.google.com/")
class Citation(BaseModel) :
    chunk_id : str 
    excerpt : str 
class AskResponse(BaseModel) :
    answer = str 
    citation : list[Citation]
    answerable : bool 
    