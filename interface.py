from langchain_community.vectorstores import Chroma
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_openai import OpenAI, ChatOpenAI

import streamlit as st
import os
import openai
import time
import hmac


os.environ["OPENAI_API_KEY"] = st.secrets["openai_api_key"]
openai.api_key  = os.environ['OPENAI_API_KEY']



def get_completion_from_messages(messages, model="gpt-3.5-turbo-1106", temperature=0):
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

	response = openai.ChatCompletion.create(
		model=model,
		messages=messages,
		temperature=temperature, # this is the degree of randomness of the model's output
	)

	# This is how you can get the number of tokens from the response
	# token_dict = {
	# 	'prompt_tokens':response['usage']['prompt_tokens'],
	# 	'completion_tokens':response['usage']['completion_tokens'],
	# 	'total_tokens':response['usage']['total_tokens'],
	# 	}
	#     
	return response.choices[0].message # , token_dict



	

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
	
	def show_conversation():
		with st.session_state.conversation_box:
			for message in st.session_state.messages:
				if message["role"]=="system": continue
				else: 
					with st.chat_message(message["role"]): st.write(message["content"])
	
	
	def new_question():
		st.session_state.new_question = True
	
	def existing_question():
		st.session_state.new_question = False
	
	if "new_question" not in st.session_state.keys():
		st.session_state.new_question = True
	
	if "vector_store" not in st.session_state.keys():
		st.session_state.vectorstore = Chroma(persist_directory="./chroma_db")
		
	st.sidebar.button("New question", on_click=new_question)
	
	st.sidebar.selectbox(
	   "Which model do you want to use",
	   ("gpt-3.5-turbo-1106", "gpt-4", "gpt-4-1106-preview"),
	   index=0,
	   key="model_value",
	   on_change=new_question,
	)
	
	st.title("[HowToTreatScars](https://howtotreatscars.com/)")
	st.image('./banner2.jpg')
	
	
	if st.session_state.new_question:	
		st.button("New question YES", on_click=existing_question)
		
	else:
		st.chat_input("Enter your message", on_submit=new_question, key="chat_input")

		
	
	
	
	
	#st.session_state.history.append(st.subheader(f"Powered by ChatGPT version: {st.session_state.model_value}",help=f"{st.session_state.model_value} is a large languange model (LLM) hosted by OpenAI, that can understand as well as generate natural language or code: [link for more info](https://platform.openai.com/docs/models)"))
	
	
	
			
	#st.chat_input("Enter your message", on_submit=submit_question, key="chat_input")

	
	
	
	

if __name__ == '__main__':
	main()