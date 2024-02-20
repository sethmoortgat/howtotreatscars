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
from copy import deepcopy

__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

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


template_after_summary = """
You are a friendly assistant that helps people who are browsing a website with information on scar treatments.
You are polite, provide extensive accurate answers, and point the user to the right location for more information.

You have to answer a question that you find below, but only using information that is given to you as 'context'.
Do not use any other information and make sure your answer is almost an exact copy of the relevant text in the context.
The provided context in split in different chunks of information delimited by triple '#', and at the end of each
piece of context you find a urls where the info is retrieved from. You are allowed to combine information from
different parts of the context into one consistent and complete answer.

If the question is completely unrelated to the treatment of scars, do NOT make up an answer but instead reply with:
'Sorry, this information can not be found on the website.'
If the user did not really ask a question at all, remain polite and answer adequately, but focus on reminding the user you are here
to answer any question related to the website 'howtotreatscars.com'.

If you give an answer, end your answer by stating on which website this info can be found, which is given at the end of each piece of context.
Make sure to give the entire link, starting with 'https:'
You are also allowed to give multiple URLs.

There is already a history of the conversation, which is given below between triple quotes:

history of the conversation:
'''
{history}
'''
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
If the user did not really ask a question at all, remain polite and answer adequately, but focus on reminding the user you are here
to answer any question related to the website 'howtotreatscars.com'.

If you give an answer, end your answer by stating on which website this info can be found, which is given at the end of each piece of context.
Make sure to give the entire link, starting with 'https:'
You are also allowed to give multiple URLs.

Question: {question}
Context: {context}
"""	


follow_up_template = """
A new question has been asked, you can find below some additional context that might be helpful.
If the question is related to some of the previous questions of the user, you can use previous context as well.
However, if the questions seems somewhat unrelated, please prioritise this new context to answer it.

Additional Context: {context}
"""	

summarize_template = """
Your task is to summarize the history of a given conversation, where a website vistor has been asking questions
about a website and the website owner is providing answers based on context retrieved from that website.
You have to give an extensive summary, making sure the different questions that were posed are not lost, 
but also omitting any context or information that was not useful to the questions and answers provived.

The conversation is given below between triple hashtags. It contains parts mentioned by the visitor (called 'user'),
parts mentioned by the website owner (called 'assistant') and general pieces of context (referred to as 'system')

Conversation: ###
{conversation}
###
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
	
	
	def summarize_chat_history(
		chat_history,
		summarization_template,
	):
		concatenated_message = ""
		# concatenate message history:
		for entry in chat_history:
			concatenated_message += entry["role"]
			concatenated_message += ":\n"
			concatenated_message += entry["content"]
			concatenated_message += ":\n\n"
		full_prompt = PromptTemplate.from_template(summarization_template).format(
				conversation = concatenated_message)
		answer = get_completion_from_messages([{'role':'user', 'content':full_prompt}], 
				st.session_state.client, model="gpt-3.5-turbo-1106", temperature=0.1)
		return answer
		
	
	def new_question():
		st.session_state.new_question = True
		st.session_state.question = ''
		st.session_state.context = ''
		st.session_state.messages = []
		st.session_state.chat_history = []
		st.session_state.n_questions = 0
	
	def existing_question():
		st.session_state.new_question = False
	
	def add_user_input():
		st.session_state.messages.append({'role':'user', 'content':st.session_state.chat_input})
		st.session_state.chat_history.append({'role':'user', 'content':st.session_state.chat_input})
		
		
	if "n_questions" not in st.session_state.keys():
		st.session_state.n_questions = 0
	
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
		st.session_state.messages = [] 
	
	if "chat_history" not in st.session_state.keys():
		st.session_state.chat_history = [] 
	
	st.markdown(
    """
	<style>
	button {
		height: auto;
		padding-top: 15px !important;
		padding-bottom: 15px !important;
	}
	</style>
	""",
		unsafe_allow_html=True,
	)
	
	#st.sidebar.title("Menu")
	st.sidebar.image('./logo.png')	
	st.sidebar.button("New question", on_click=new_question, type="primary")
	
	st.sidebar.selectbox(
	   "Which model do you want to use",
	   ("gpt-3.5-turbo-1106", "gpt-4-1106-preview"),
	   index=0,
	   key="model_value",
	   on_change=new_question,
	)
	
	# st.title("[HowToTreatScars](https://howtotreatscars.com/)")
	#st.image('./banner2.jpg')
	
	info_text = """
	This is a chatbot powered by ChatGPT. Even though its accuracy and relevance have been thoroughly tested,
	the responses that are provided should always be validated against the true content of [howtotreatscars.com](https://howtotreatscars.com/).
	"""
	st.info(info_text,icon="ℹ️")

	if st.session_state.new_question:	
		label = 'Ask your question:'
		st.text_input(label, key='question', on_change=existing_question)
		st.components.v1.html(
			f"""
			<script>
				var elems = window.parent.document.querySelectorAll('div[class*="stTextInput"] p');
				var elem = Array.from(elems).find(x => x.innerText == '{label}');
				elem.style.fontSize = '20px'; // the fontsize you want to set it to
			</script>
			"""
		)
		
		
	else:
		
		if len(st.session_state.messages) == 0:
			st.session_state.messages.append({'role':'user', 'content':st.session_state.question})
			st.session_state.chat_history.append({'role':'user', 'content':st.session_state.question})
			st.session_state.context = get_context_from_db(
				st.session_state.question,
				st.session_state.vectorstore,
				n_retrieve=3
			)
			question_prompt = PromptTemplate.from_template(template).format(
				question = st.session_state.question,
				context = st.session_state.context)
			st.session_state.messages.append({'role':'system', 'content':question_prompt})
			st.session_state.chat_history.append({'role':'system', 'content':question_prompt})
		
		else:
			last_question = st.session_state.messages[-1]["content"]
			follow_up_context = get_context_from_db(
				last_question,
				st.session_state.vectorstore,
				n_retrieve=2
			)
			follow_up_question_prompt = PromptTemplate.from_template(follow_up_template).format(
				context = follow_up_context)
			st.session_state.messages.append({'role':'system', 'content':follow_up_question_prompt})
			st.session_state.chat_history.append({'role':'system', 'content':follow_up_question_prompt})
			
		answer = get_completion_from_messages(st.session_state.messages, st.session_state.client, model=st.session_state.model_value, temperature=0.1)
		st.session_state.messages.append({'role':answer.role, 'content':answer.content})
		st.session_state.chat_history.append({'role':answer.role, 'content':answer.content})
		
		st.session_state.n_questions += 1
		# in case already more than 3 questions got asked, summarize history and start with new system prompt
		if st.session_state.n_questions%3 == 0:
			summary = summarize_chat_history(st.session_state.messages,summarize_template)
			summary_of_history_template = PromptTemplate.from_template(template_after_summary).format(
				history = summary.content)
			st.session_state.messages = [{'role':"system", 'content':summary_of_history_template}]
		
			
		# show the chat history on screen
		with st.container():
			for idx, message in enumerate(st.session_state.chat_history):
				if message["role"]=="system": continue
				else: 
					if message["role"]=="assistant":icon="./icon_self.png"
					else: icon="./user.png"
					with st.chat_message(message["role"],avatar=icon): st.write(message["content"])
		
		# show the bar to enter a new question
		st.chat_input("Enter your next question...", key="chat_input", on_submit = add_user_input)


		
	
	
	
	
	#st.session_state.history.append(st.subheader(f"Powered by ChatGPT version: {st.session_state.model_value}",help=f"{st.session_state.model_value} is a large languange model (LLM) hosted by OpenAI, that can understand as well as generate natural language or code: [link for more info](https://platform.openai.com/docs/models)"))
	
	
	
			
	#st.chat_input("Enter your message", on_submit=submit_question, key="chat_input")

	
	
	
	

if __name__ == '__main__':
	main()