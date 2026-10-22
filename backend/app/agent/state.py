from typing import TypedDict , Literal
from pydantic import BaseModel, Field
class RelevanceClass(BaseModel):
    relevance : Literal['relevant','out_of_scope'] = Field(description='determine if the question is related to the page_content or not ')

class AgentState(TypedDict) :
    question : str = Field(description = 'question asked by the user',min_length = 1,max_length = 1024)
    url_content : str = Field(description = 'the url page content ',min_length=1)
    source_url : str = Field(description = 'the url given by the user')
    evidence : list[str]
    answer : str 
    status : str 
    citations : list[dict[str,str]]
    relevance : RelevanceClass
