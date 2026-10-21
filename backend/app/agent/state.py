from typing import TypedDict
class AgentSate(TypedDict):
    question : str 
    page_text : str 
    page_url : str 
    evidence : list[dict[str,str]]
    answer : str 
    citations : list[dict[str,str]]
    answerable : bool 
    error : str | None 
    