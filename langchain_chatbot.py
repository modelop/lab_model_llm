import glob
import os

import weaviate
import guardrails as gd
import openai

from langchain import OpenAI, LlamaCpp, PromptTemplate
from langchain.chains import ConversationalRetrievalChain
from langchain.document_loaders import DirectoryLoader
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.indexes import VectorstoreIndexCreator
from langchain.prompts import SystemMessagePromptTemplate, load_prompt, HumanMessagePromptTemplate, ChatPromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Weaviate

OPEN_AI_CONVERSATION: ConversationalRetrievalChain
LLAMA_CONVERSATION: ConversationalRetrievalChain
GUARDRAILS: [gd.guard]


#
# This is the model initialization function.  This will be called at model load time by the modelop runtime.  In this
# instance it will load up two chains.  The first chain is for openai, and will utilize it's API remote access point
# for all LLM interactions, and use a larger chunked document store than the local option.  It will also load a second
# chain utilizing a local copy of llama2 to execute the requests, along with a smaller chunked document store to reduce
# the overall processing.  Both will be available for use by the scoring engine.

# modelop.init
def init() -> None:
	global OPEN_AI_CONVERSATION, LLAMA_CONVERSATION, GUARDRAILS

	print("Loading prompts....")
	system_prompt = load_prompt("system_prompt.json")
	user_prompt = load_prompt("user_prompt.json")
	messages = [
		SystemMessagePromptTemplate(prompt=system_prompt),
		HumanMessagePromptTemplate(prompt=user_prompt)
	]
	qa_prompt = ChatPromptTemplate.from_messages(messages)

	print("Connecting to weaviate....", flush=True)
	db_client = weaviate.Client(url='http://weaviate:8092')
	print("Instantiating OpenAI LLM.....", flush=True)
	open_ai_llm = OpenAI(temperature=0.2)
	print("Fetching Hugging Face OpenAI Embedding Model....", flush=True)
	open_ai_embedding_model = HuggingFaceInstructEmbeddings(model_name='hkunlp/instructor-xl')
	open_ai_vectorstore = Weaviate(db_client, index_name='Model_Governance_Docs_Instructor_Embedding',
								   embedding=open_ai_embedding_model, by_text=False,
								   text_key='text')
	print("Instantiating OpenAI Conversational Chain....", flush=True)
	OPEN_AI_CONVERSATION = ConversationalRetrievalChain.from_llm(open_ai_llm, open_ai_vectorstore.as_retriever(),
																 return_source_documents=True, verbose=True,
																 combine_docs_chain_kwargs={'prompt': qa_prompt})

	print("Instantiating LLamaCPP LLM....", flush=True)
	llama_files = glob.glob('./llama-2-7b-chat*.bin')
	if not llama_files:
		raise Exception('Could not find any llama2 binary files, initialization failed!')

	llama_llm = LlamaCpp(model_path=os.path.abspath(llama_files[0]),
						 input={"temperature": 0.2, "max_length": 1000, "top_p": 1}, n_ctx=2048,
						 verbose=True)
	print("Loading Llama2 Embedding Model....", flush=True)
	llama_embedding_model = HuggingFaceInstructEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
	llama_vectorstore = Weaviate(db_client, index_name='Model_Governance_Docs_MiniLM_Embedding',
								 embedding=llama_embedding_model, by_text=False,
								 text_key='text')
	print("Instantiating Llama2 Conversational Chaing....", flush=True)
	LLAMA_CONVERSATION = ConversationalRetrievalChain.from_llm(llama_llm, llama_vectorstore.as_retriever(),
															   return_source_documents=True, verbose=True,
															   combine_docs_chain_kwargs={'prompt': qa_prompt})
	print("Applying guard rails....")
	rail_files = glob.glob('./*.rail')
	GUARDRAILS = []
	for rail_file in rail_files:
		GUARDRAILS.append(gd.Guard.from_rail(rail_file))


#
# This function implements the scoring method that the modelop runtime will call with each scoring request.  In this
# case those requests are a chatbot interface.  The client should pass in three different fields in their request:
#
# model - Either openai or llama2 indicating which LLM to utilize for responses
# question - The question that is being asked
# chat_history - Either an empty array ([]) or the last returned chat_history from the response of this method
#
# The function will return a response with several fields set:
# answer - The answer to the given question as determined by the llm and corresponding documents
# chat_history - The history that should be passed back in to continue the conversation
# source_documents - The documents referenced to result in this response
#

# modelop.score
def score(request: dict) -> dict:
	global OPEN_AI_CONVERSATION, LLAMA_CONVERSATION, GUARDRAILS

	result = {}
	query = {"question": request["question"],
			 "chat_history": request.get("chat_history", [])}
	if request.get('model', 'openai') == 'openai':
		qa_result = OPEN_AI_CONVERSATION(query)
	elif request.get('model', 'llama2') == 'llama2':
		qa_result = LLAMA_CONVERSATION(query)
	else:
		qa_result = {"answer": "I don't know"}

	validated_response = {"answer": qa_result["answer"]}
	for guard_rail in GUARDRAILS:
		raw_llm_response, validated_response = guard_rail(openai.Completion.create,
											 prompt_params=validated_response,
											 engine="text-davinci-003",
											 max_tokens=2048,
											 temperature=0)

	query["chat_history"].append((query["question"], 'Answer: ' + validated_response['answer']))
	result["answer"] = validated_response
	result["chat_history"] = query["chat_history"]

	return result


#
# This method implements a local ability to chat with either model.  This allows for testing of the model without having
# to deploy it onto a modelop runtime.  Instead, the main function can be called and this method will loop endlessly on
# providing a chat interface to the metrics() function
#
def chat_with_documents(model: str) -> None:
	chat_history = []
	while True:
		print(">>>", end=" ")
		query = input("")
		result = score({
			"model": model,
			"question": query,
			"chat_history": chat_history})
		print(result["answer"])
		chat_history = result["chat_history"]


#
# Generate the prompt files utilized for querying the llm
#
def generate_prompt_files():
	template = """
	Use the following pieces of information to answer the user's question.
	If you don't know the answer, just say that you don't know, don't try to make up an answer.
	Context: {context}
	Question: {question}
	Only return the helpful answer below and nothing else.
	Helpful answer:	"""

	prompt_template = PromptTemplate.from_template(template=template)
	prompt_template.save("system_prompt.json")
	template = """
	Question:'''{question}'''
	"""
	prompt_template = PromptTemplate.from_template(template=template)
	prompt_template.save("user_prompt.json")



#
# This method is utilized to load documents into the vector store.  It chunks these documents into two different
# indexes.  One is smaller chunks to be utlilized for a local llm instance where resources are more constrained, and
# another is with larger chunks with more overlap when used with larger llms such as chatgpt-4.  This should be run
# once to prepare a vector store with the documents and would not be called as a generic part of deploying this model.
#
def load_documents() -> None:
	loader = DirectoryLoader('./documents', glob='**/*.pdf', show_progress=True)
	documents = loader.load()
	db_client = weaviate.Client(url='http://weaviate:8092')
	splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
	embeddings = HuggingFaceInstructEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
	vectorstore_kwargs = {"client": db_client, "index_name": "Model_Governance_Docs_MiniLM_Embedding"}
	VectorstoreIndexCreator(embedding=embeddings, vectorstore_cls=Weaviate,
							text_splitter=splitter, vectorstore_kwargs=vectorstore_kwargs).from_documents(documents)

	splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=300)
	embeddings = HuggingFaceInstructEmbeddings(model_name='hkunlp/instructor-xl')
	vectorstore_kwargs = {"client": db_client, "index_name": "Model_Governance_Docs_Instructor_Embedding"}
	VectorstoreIndexCreator(embedding=embeddings, vectorstore_cls=Weaviate,
							text_splitter=splitter, vectorstore_kwargs=vectorstore_kwargs).from_documents(documents)


#
# A main entry point that allows for running a model outside of the modelop runtime.  In this case it will either allow
# you to vectorize and embed documents for queries in a vector store, or begin an interactive chat session with the
# chosen llm.  This can be used for debugging and testing.
#
def main():
	print("1 - Chat with Loaded Documents")
	print("2 - Load Documents into Vector Database")
	print("3 - Create Prompt Templates")
	choice = input(">>> ")
	if choice == "2":
		load_documents()
	elif choice == "3":
		generate_prompt_files()
	else:
		init()
		print("1 - Use local llama2 instance")
		print("2 - Use OpenAI Chat-GPT 4")
		choice = input(">>> ")
		if choice == "1":
			model = "llama2"
		else:
			model = "openai"
		chat_with_documents(model)


if __name__ == '__main__':
	main()

