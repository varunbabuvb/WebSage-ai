from pydantic import BaseModel,Field
from enum  import Enum

class AskRequest(BaseModel) :
    source_url : str = Field(description = 'this is the url given by the user ',example = "https://www.google.com/",min_length = 1,max_length = 1024)
    url_content  : str = Field(description = 'this is the text extracted from the source_url ',min_length = 1,max_length = 100000)
    question : str =Field(description = "the question asked by the user which need to be answered from the page content ",min_length = 1, max_length = 1024)

class Status(str,Enum):
    answered = 'answered'
    out_of_scope = 'out_of_scope'
    insufficient_evidence = 'insufficient_evidence'



class Citation (BaseModel) :
    source_url : str = Field(description= 'the url from which this is retrived')
    excerpt  : str=Field(description = 'the retieved text from the source_url')

class AskResponse(BaseModel):
    answer : str = Field(description =' the answer given by the agent ')
    status : Status
    citations : list[Citation]

