
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate

import streamlit as st
import openai
import os
import hmac

lang_dict = {
	"🇧🇪  Nederlands":"NL",
	"🇬🇧  English":"EN"
}

template_text_system = """ You are a friendly assistant that helps people who are browsing a website with information on scar treatments.
You are polite, provide extensive accurate answers, and point the user to the right location for more information.
You always answer in the same language as the original question.

You have to answer a question that you find below, but only using information in the context below.
Do not use any other information and make sure your answer is almost an exact copy of the relevant text in the context.
The provided context is split in different chunks of information delimited by triple '#', and at the end of each
piece of context you find a urls where the info is retrieved from. You are allowed to combine information from
different parts of the context into one consistent and complete answer.

If the question is completely unrelated to the treatment of scars, do NOT make up an answer but instead reply with:
'Sorry, this information can not be found on the website.'. If however you can not find an exact answer in the context, 
but you find some related information, you can still give a reply acknowleding that it might not exactly answer their question,
but more info might be available on the website.

If you give an answer, end your answer by stating on which website this info can be found, which is given at the end of each piece of context.
Make sure to give the entire link, starting with 'https:'
You are also allowed to give multiple URLs.
Add the URL in the following form: "You can read more about <topic_the_question_was_about> on: https://..."
You can use the context of the entire chat history to answer any follow-up questions
"""

context_template_text = "The following context has been added to the conversation: {context}"

def get_context(
	query,
    vectorstore,
    n_chunks=3,
    filters = None,
):
	chunks = vectorstore.max_marginal_relevance_search(
		query,
		k=n_chunks,
		filter=filters,
	)

	context = ""
	for _chunk in chunks:
		summary = "###\n"+ _chunk.page_content + "\n This info was retrieved from: " + _chunk.metadata["url"] + "\n###\n"
		context+=summary
	
	return context


def get_llm_response(
	messages,
	llm,
):
	response = llm.invoke(messages)
	return response.content

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
		
	def change_language():
		st.session_state.language = lang_dict[st.session_state.new_language]
	
	if "language" not in st.session_state.keys():
		st.session_state.language = 'NL'
	
	if "question" not in st.session_state.keys():
		st.session_state.question = ''
	
	if "context" not in st.session_state.keys():
		st.session_state.context = ''
	
	if "new_question" not in st.session_state.keys():
		st.session_state.new_question = True
	
	if "embedding_function" not in st.session_state.keys():
		st.session_state.embedding_function = OpenAIEmbeddings(
			openai_api_key=st.secrets["openai_api_key"],
			model="text-embedding-3-large",
		)
	
	if "vector_store" not in st.session_state.keys():
		st.session_state.vectorstore = Chroma(
			collection_name="myscarspecialist",
    		embedding_function=st.session_state.embedding_function,
    		persist_directory="./myscarspecialist_chroma_db",
    	)
	
	if "client" not in st.session_state.keys():
		st.session_state.client = ChatOpenAI(
			model="gpt-4o",
			temperature=0.2,
			max_tokens=None,
			timeout=None,
			max_retries=2,
			openai_api_key=st.secrets["openai_api_key"]
		)

	if "messages" not in st.session_state.keys():
		st.session_state.messages = [] 
	
	if "chat_history" not in st.session_state.keys():
		st.session_state.chat_history = [] 
	
	st.set_page_config(layout="wide") 
	

	info_text = """This is a chatbot powered by ChatGPT. Even though its accuracy and relevance have been thoroughly tested,
the responses that are provided should always be validated against the true content of [myscarspecialist.com](https://myscarspecialist.com/).
"""
	col1, col2 = st.columns([2,15])
	
	with col1:
		st.selectbox(
			label="",
			options=(lang_dict.keys()),
			key="new_language",
			on_change=change_language,
		)
	
	with col2:
		st.info(info_text,icon="ℹ️")
	

	if st.session_state.new_question:	
		label = 'Ask your question:'
		#st.markdown(f"## {label}")
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
			# First initialise the system prompt
			system_prompt = template_text_system
			st.session_state.messages.append({'role':'system', 'content':system_prompt})
			st.session_state.chat_history.append({'role':'system', 'content':system_prompt})
			
			# Then add the context prompt
			with st.spinner('Browsing website...'):
				st.session_state.context = get_context(
					st.session_state.question,
					st.session_state.vectorstore,
					n_chunks=3,
				)
			context_prompt = context_template_text.format(context = st.session_state.context)
			st.session_state.messages.append({'role':'system', 'content':context_prompt})
			st.session_state.chat_history.append({'role':'system', 'content':context_prompt})
			
			# Finally add the user question
			st.session_state.messages.append({'role':'user', 'content':st.session_state.question})
			st.session_state.chat_history.append({'role':'user', 'content':st.session_state.question})
		
		else:
			last_question = st.session_state.messages[-1]["content"]
			with st.spinner('Browsing website...'):
				follow_up_context = get_context(
					last_question,
					st.session_state.vectorstore,
					n_chunks=2,
				)
			context_prompt = context_template_text.format(context = follow_up_context)
			st.session_state.messages.append({'role':'system', 'content':context_prompt})
			st.session_state.chat_history.append({'role':'system', 'content':context_prompt})

		with st.spinner('Composing answer...'):	
			answer = get_llm_response(
				st.session_state.messages,
				st.session_state.client,
			)
		st.session_state.messages.append({'role':'assistant', 'content':answer})
		st.session_state.chat_history.append({'role':'assistant', 'content':answer})

		
			
		# show the chat history on screen
		with st.container():
			for idx, message in enumerate(st.session_state.chat_history):
				if message["role"]=="system": continue
				else: 
					if message["role"]=="assistant":icon="./icon_self.png"
					else: icon="./user.png"
					with st.chat_message(message["role"],avatar=icon): st.write(message["content"])
		
		# show the bar to enter a new question
		col1, col2, col3 = st.columns([5,0.2, 1])

		with col1:
			st.chat_input("Enter your follow-up question...", key="chat_input", on_submit = add_user_input)
		
		with col2:
			st.markdown('##### or')
		
		with col3:
			st.button("Submit new question", on_click=new_question, type="primary")
		
		


		
	
	
	
	
	#st.session_state.history.append(st.subheader(f"Powered by ChatGPT version: {st.session_state.model_value}",help=f"{st.session_state.model_value} is a large languange model (LLM) hosted by OpenAI, that can understand as well as generate natural language or code: [link for more info](https://platform.openai.com/docs/models)"))
	
	
	
			
	#st.chat_input("Enter your message", on_submit=submit_question, key="chat_input")

	
	
	
	

if __name__ == '__main__':
	main()