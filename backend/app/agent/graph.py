import os 
from dotenv import load_dotenv 
from langchain_google_genai import ChatGoogleGenerativeAI 
from app.agent.state import AgentState
from pathlib import Path 
from langchain_core.messages import SystemMessage ,HumanMessage
from langgraph.graph import StateGraph,START,END 

base_dir = Path(__file__).resolve().parents[2]
load_dotenv(dotenv_path=base_dir/".env" )

llm = ChatGoogleGenerativeAI(google_api_key = os.getenv('GOOGLE_API_KEY'),model = 'gemini-2.5-flash')

graph = StateGraph(AgentState)

def generate_answer(state : AgentState) -> dict :
    print('calling generate_answer function ...')
    question = state['question']
    page_content = state['url_content']
    messages = [
        SystemMessage(content = '''Answer questions using only the supplied webpage content if  
        if the question is unrelayted to the page ,say it is out of scope .
        Do not use general knowledge to fill gaps .''')
        ,HumanMessage(
            content = f"Webpage content: {page_content} user question : {question}"
        )
    ]
    response = llm.invoke(messages)
    return {'answer': response.content}
def check_relevance(state : AgentState) -> str:
    print('calling check_relevance function  ...')
    prompt = '''analyse both the page content and the question asked by the user
    and classify the question as one of the two 1. relevant 2.out_of_scope 
    classify as relevant if the question and page content are talking about a similar context 
    classify as out_of_scope if there is no relation in between page content and the question  
    you need to give only one word answer either relevant or out_of_scope '''
    question = state['question']
    page_content = state['url_content']
    messages = [
        SystemMessage(content = prompt)
        ,HumanMessage(content = f'classify the question {question} based on {page_content}')

    ]
    response= llm.invoke(messages)
    relevance = response.content.strip().lower()
    return relevance 

def out_of_scope(state: AgentState)->dict :
    print('calling out_of_scope function ...')
    answer = ' This question is outside the scope of the current webpage.Please ask a question relevant to the websites content .'
    status = 'out_of_scope'
    return {
        'answer' : answer ,
        'status' : status 
    }
    

import re

def find_relevant_passage(state: AgentState) -> dict:
    print('Calling find_relevant_passage function ... ')
    question = state["question"]
    page_content = state["url_content"]

    passages = [
        paragraph.strip()
        for paragraph in page_content.split("\n")
        if paragraph.strip()
    ]

    question_words = set(re.findall(r"\b\w+\b", question.lower()))

    scored_passages = []

    for passage in passages:
        passage_words = set(
            re.findall(r"\b\w+\b", passage.lower())
        )

        common_words = question_words & passage_words
        score = len(common_words)

        if score > 0:
            scored_passages.append((score, passage))

    scored_passages.sort(key=lambda item: item[0], reverse=True)

    evidence = [
        passage for score, passage in scored_passages[:3]
    ]

    return {"evidence": evidence}
graph.add_node('generate_answer',generate_answer)
graph.add_node('out of scope',out_of_scope)
graph.add_node('find relevant passage',find_relevant_passage)
graph.add_conditional_edges(START,check_relevance,
                            {'relevant':'find relevant passage','out_of_scope':'out of scope'})

graph.add_edge('find relevant passage','generate_answer')
graph.add_edge('generate_answer',END)
graph.add_edge('out of scope',END)

graph = graph.compile()


input_state = {
    "question": "What does WebSage AI do?",
    "url_content": (
        "WebSage AI summarizes webpages and answers questions.\n"
        "It provides citations to support its answers.\n"
        "The application was built using React and Python."
    ),
    "source_url": "https://example.com"
}

print(find_relevant_passage(input_state))
output_state = graph.invoke(input_state)

print(output_state)




