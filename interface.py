from langchain_community.vectorstores import Chroma
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_openai import OpenAI, ChatOpenAI, OpenAIEmbeddings

import streamlit as st
import os
import openai
from openai import OpenAI
import time
import hmac
import time


os.environ["OPENAI_API_KEY"] = st.secrets["openai_api_key"]
#openai.api_key  = os.environ['OPENAI_API_KEY']




def get_completion_from_messages(messages, client, model="gpt-3.5-turbo-1106", temperature=0.1):
	"""
	Input takes a n array of messages that specify a set of prompts and associated roles
	for example:
	messages =  [  
		{'role':'system', 'content':'You are an assistant that speaks like Shakespeare.'},    
		{'role':'user', 'content':'tell me a joke'},   
		{'role':'assistant', 'content':'Why did the chicken cross the road'},   
		{'role':'user', 'content':'I don\'t know'}  
	]

	Output is a message that has a format of: [{"role": "user", "content": prompt}].
	One needs to use the parse_response function to extract the actual 
	model output message from the output.
	"""

	response = client.chat.completions.create(
    	messages=messages,
    	model=model,
    	temperature=temperature
	)

	return response.choices[0].message 


system_message = """
You are a friendly assistant that helps people who are browsing a website with information on scar treatments.
You are polite, provide extensive accurate answers, and point the user to the right location for more information.
"""

template = """
You are a friendly assistant that helps people who are browsing a website with information on scar treatments.
You are polite, provide extensive accurate answers, and point the user to the right location for more information.

You have to answer a question that you find below, but only using information in the context below.
Do not use any other information and make sure your answer is almost an exact copy of the relevant text in the context.
The provided context in split in different chunks of information delimited by triple '#', and at the end of each
piece of context you find a urls where the info is retrieved from. You are allowed to combine information from
different parts of the context into one consistent and complete answer.

If the question is completely unrelated to the treatment of scars, do NOT make up an answer but instead reply with:
'Sorry, this information can not be found on the website.'

If you give an answer, end your answer by stating on which website this info can be found, which is given at the end of each piece of context.
Make sure to give the entire link, starting with 'https:'
You are also allowed to give multiple URLs.
The very last sentence of your reply should always be: 'Feel free to ask any follow-up questions related to the topic above, or submit a new question on a different topicby clicking on the button "New question" in the Menu on the left.'

After providing your first answer, you are allowed to answer any follow-up questions related to the initial question.
If you can not find the answer to any follow-up of the user in the provided context, because it does not relate enough to the initial question, reply with:
'Sorry, I can not find the answer to that question in relation to your initial question, you can submit a new question by clicking on the button "New question" in the Menu on the left.'
Do NOT use 'Sorry, this information can not be found on the website.' after you answered the initial question!

Question: {question}
Context: {context}
"""	

def get_context_from_db(
    query,
    vectorstore,
    n_retrieve=3
):
    querybase = vectorstore.as_retriever(search_type="mmr", search_kwargs={"k":n_retrieve, "lambda_mult":0.6})
    res = querybase.get_relevant_documents(query)
    
    final_answer = ""
    for i in res:
        summary = "###\n"+ i.page_content + "\n This info was retrieved from: " + i.metadata["source"] + "\n###\n"
        final_answer+=summary
        
    return final_answer




def get_response(
	prompt_template,
	question,
	context,
	model,
	api_key,
	temp = 0.1,
):
	prompt = PromptTemplate.from_template(template)
	llm = ChatOpenAI(model=model,
                       temperature=temp,
                       openai_api_key=api_key)
	llm_chain = LLMChain(prompt=prompt, llm=llm)
	answer = llm_chain.invoke({"question":question,"context":context})
	return answer["text"]

def main():

	def check_password():
		"""Returns `True` if the user had the correct password."""

		def password_entered():
			"""Checks whether a password entered by the user is correct."""
			if hmac.compare_digest(st.session_state["password"], st.secrets["password"]):
				st.session_state["password_correct"] = True
				del st.session_state["password"]  # Don't store the password.
			else:
				st.session_state["password_correct"] = False

		# Return True if the passward is validated.
		if st.session_state.get("password_correct", False):
			return True

		# Show input for password.
		st.text_input(
			"Password", type="password", on_change=password_entered, key="password"
		)
		if "password_correct" in st.session_state:
			st.error("😕 Password incorrect")
		return False


	if not check_password():
		st.stop()  # Do not continue if check_password is not True.
	
	
	st.set_page_config(layout="wide") 
	
	
	st.sidebar.title("Menu")
	
	
	
	def new_question():
		st.session_state.new_question = True
		st.session_state.question = ''
		st.session_state.context = ''
		st.session_state.messages = []
	
	def existing_question():
		st.session_state.new_question = False
	
	def add_user_input():
		st.session_state.messages.append({'role':'user', 'content':st.session_state.chat_input})
		
	
	if "question" not in st.session_state.keys():
		st.session_state.question = ''
	
	if "context" not in st.session_state.keys():
		st.session_state.context = ''
	
	if "new_question" not in st.session_state.keys():
		st.session_state.new_question = True
	
	if "embedding_function" not in st.session_state.keys():
		st.session_state.embedding_function = OpenAIEmbeddings(openai_api_key=st.secrets["openai_api_key"])
	
	if "vector_store" not in st.session_state.keys():
		st.session_state.vectorstore = Chroma(persist_directory="./chroma_db",
			embedding_function = st.session_state.embedding_function)
	
	if "client" not in st.session_state.keys():
		st.session_state.client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

	
	if "messages" not in st.session_state.keys():
		st.session_state.messages = [  
			#{'role':'system', 'content':system_message},
		] 
		
	st.sidebar.button("New question", on_click=new_question)
	
	st.sidebar.selectbox(
	   "Which model do you want to use",
	   ("gpt-3.5-turbo-1106", "gpt-4-1106-preview"),
	   index=0,
	   key="model_value",
	   on_change=new_question,
	)
	
	st.title("[HowToTreatScars](https://howtotreatscars.com/)")
	st.image('./banner2.jpg')
	

	if st.session_state.new_question:	
		st.text_input('Ask your question:',key='question', on_change=existing_question)
		
		
	else:
		if len(st.session_state.messages) == 0:
			st.session_state.context = get_context_from_db(
				st.session_state.question,
				st.session_state.vectorstore,
				n_retrieve=3
			)
			question_prompt = PromptTemplate.from_template(template).format(
				question = st.session_state.question,
				context = st.session_state.context)
			st.session_state.messages.append({'role':'system', 'content':question_prompt})
		# answer = get_response(
# 			prompt_template = template,
# 			question = st.session_state.question,
# 			context = st.session_state.context,
# 			model=st.session_state.model_value,
# 			api_key = st.secrets["openai_api_key"],
# 			temp = 0.1,
# 		)
		answer = get_completion_from_messages(st.session_state.messages, st.session_state.client, model=st.session_state.model_value, temperature=0.1)
		st.session_state.messages.append({'role':answer.role, 'content':answer.content})
	
		with st.container():
			for idx, message in enumerate(st.session_state.messages):
				if message["role"]=="system": continue
				else: 
					with st.chat_message(message["role"]): st.write(message["content"])
		st.chat_input("Enter your message", key="chat_input", on_submit = add_user_input)


		
	
	
	
	
	#st.session_state.history.append(st.subheader(f"Powered by ChatGPT version: {st.session_state.model_value}",help=f"{st.session_state.model_value} is a large languange model (LLM) hosted by OpenAI, that can understand as well as generate natural language or code: [link for more info](https://platform.openai.com/docs/models)"))
	
	
	
			
	#st.chat_input("Enter your message", on_submit=submit_question, key="chat_input")

	
	
	
	

if __name__ == '__main__':
	main()