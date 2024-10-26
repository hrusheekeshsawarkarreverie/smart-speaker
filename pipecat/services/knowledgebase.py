import asyncio
import json
import requests
import aiohttp
import time
from typing import List
from requests import RequestException
from pipecat.frames.frames import (
    Frame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    LLMMessagesFrame,
    LLMResponseEndFrame,
    LLMResponseStartFrame,
    TextFrame,
    VisionImageRawFrame,
    ErrorFrame,
    AudioRawFrame,
)
from pipecat.processors.aggregators.openai_llm_context import (
    OpenAILLMContext,
    OpenAILLMContextFrame,
)
try:
    from openai import AsyncOpenAI, AsyncStream, BadRequestError, OpenAI
    from openai.types.chat import (
        ChatCompletionChunk,
        ChatCompletionFunctionMessageParam,
        ChatCompletionMessageParam,
        ChatCompletionToolParam,
    )
except ModuleNotFoundError as e:
    logger.error(f"Exception: {e}")
    logger.error(
        "In order to use OpenAI, you need to `pip install pipecat-ai[openai]`. Also, set `OPENAI_API_KEY` environment variable."
    )
    raise Exception(f"Missing module: {e}")
from loguru import logger
import random
import string
from typing import AsyncGenerator
import http.client
import json
from pipecat.services.ai_services import LLMService
from pipecat.processors.frame_processor import FrameDirection
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
from qdrant_client.http.models import PointStruct
from typing import List
from qdrant_client import QdrantClient, AsyncQdrantClient, models
from openai import OpenAI
import numpy as np
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import OpenAIEmbeddings
from langchain.vectorstores import Qdrant
from langchain.callbacks import StdOutCallbackHandler
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_history_aware_retriever,create_retrieval_chain
from langchain_core.prompts import MessagesPlaceholder
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory,InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.messages import HumanMessage

# from langchain_groq import ChatGroq



class ReverieKnowledgeBase(LLMService):
    def __init__(
        self,
        # aiohttp_session: aiohttp.ClientSession,
        api_key: str,
        model: str,
        namespace: str,
        tgt_lan: str,
        nmt_flag: str,
        nmt_provider: str,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._openai_client = OpenAI(api_key=api_key)
        self.openai_key = api_key
        self._save_bot_context = kwargs.get("save_bot_context")
        logger.debug(f"{api_key}, {model}, {namespace}, {tgt_lan}")
        self._Aqdrant_client = AsyncQdrantClient(
            url="http://172.18.0.4", 
            # https=True,
            port=6333,
            grpc_port=6334,
            prefer_grpc=True
            )
        self._qdrant_client = QdrantClient(            
            url="http://172.18.0.4", 
            # https=True,
            port=6333,
            grpc_port=6334,
            prefer_grpc=True
            )
        # self._embedding_model = OpenAIEmbeddings(api_key=api_key,model="text-embedding-3-large",dimensions=1536)
        self._embedding_model = OpenAIEmbeddings(api_key=api_key)
        self.config = {"configurable": {"session_id": "123"}}
        self.store ={}
        self.collection_name = namespace

        self.vectorstore = Qdrant(client=self._qdrant_client,
                            async_client=self._Aqdrant_client,
                            collection_name=self.collection_name,
                            embeddings=self._embedding_model,
                            content_payload_key="content"
                            # vector_name="content"
                            )
        self.retriever = self.vectorstore.as_retriever()
        logger.debug("Retriever set")
        self.handler =  StdOutCallbackHandler()
        self.llm = ChatOpenAI(temperature=0.0,
                        model="gpt-4o",
                        max_tokens=512)
        self.contextualize_q_system_prompt = (
            "Given a chat history and the latest user question "
            "which might reference context in the chat history, "
            "formulate a standalone question which can be understood "
            "without the chat history. Do NOT answer the question, "
            "just reformulate it if needed and otherwise return it as is."
        )
        self.contextualize_q_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", self.contextualize_q_system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )
        self.history_aware_retriever = create_history_aware_retriever(
            self.llm, self.retriever, self.contextualize_q_prompt
        )
        self.system_prompt = (
            "You are an assistant for question-answering tasks, your name is Reva. "
            "Use the following pieces of retrieved context to answer "
            "the question. If you don't know the answer, say that you "
            "don't know. Use three sentences maximum and keep the "
            "answer concise."
            "\n\n"
            "{context}"
        )
        self.qa_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", self.system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )
        self.question_answer_chain = create_stuff_documents_chain(self.llm, self.qa_prompt)

        self.rag_chain = create_retrieval_chain(self.history_aware_retriever, self.question_answer_chain)

        self.conversational_rag_chain = RunnableWithMessageHistory(
            self.rag_chain,
            lambda sid: self.get_session_history(sid),
            input_messages_key="input",
            history_messages_key="chat_history",
            output_messages_key="answer",
        ) 
        self._completion = ""
        # self.groq_client = ChatGroq(temperature=0, groq_api_key="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", model_name="llama3-8b-8192")

    def get_session_history(self,session_id: str) -> BaseChatMessageHistory:
        if session_id not in self.store:
            # self.store[session_id] = ChatMessageHistory()
            self.store[session_id] = InMemoryChatMessageHistory()
        return self.store[session_id]

    # Function to generate completion from prompt
    def generate_completion(self,prompt):
        completion = self._openai_client.chat.completions.create(
            model="gpt-3.5-turbo",
            # stream=True,
            messages=[
                {"role": "system", "content": "You are researcher, your task is to help in answering a question."},
                {"role": "user", "content": prompt}
            ]
            # messages=messages,
        )
        return completion.choices[0].message.content


    def embed_query(self,Question):
        return self._embedding_model.embed_query(Question)
        # return np.array(list(self._embedding_model.embed_query([Question])))

    async def generate_rag(self,Question,context_messages):
        query_embeddings_time = time.time()
        query_embeddings = self.embed_query(Question)
        logger.info("query_embeddings Time consuming: {:.4f}s".format(time.time() - query_embeddings_time))
        logger.debug(f"query_embeddings: {type(query_embeddings)}, len: {len(query_embeddings)}")

        # # collection_name = '8849fc29-f47e-4fd4-a564-569048f65202'
        # collection_name = '53f0f281-ab6a-4af1-96bd-a39734964960'
        all_text = ""

        # # # Retrieve all hits and concatenate texts into a single prompt
        # # for query_embedding in query_embeddings:
        # #     # query_vector: List[np.ndarray] = list(query_embedding)
        # collection_exist=False
        collection_exist = await self._Aqdrant_client.collection_exists(self.collection_name)
        print(collection_exist)
        # if collection_exist:
        query_vector=query_embeddings
        Retrieval_time = time.time()
        hits = await self._Aqdrant_client.search(
            collection_name=self.collection_name,
            query_vector=query_vector,
            limit=8
        )
        logger.info("Retrieval_time Time consuming: {:.4f}s".format(time.time() - Retrieval_time))
        logger.debug(f"hits: {hits}")
        for hit in hits:
            try:
                text = hit.payload["content"]
            except:
                text = hit.payload["text"]
            all_text += text + "\n"
        print(all_text)
        # # Generate completion using all texts as a single prompt
        # prompt = f"Given the following context:\n{all_text}\n\nAnd the message history:\n{context_messages}\n\nAnswer the following question\n\nQuestion: {Question}?"
        prompt = f"Given the following context:\n{all_text}\n\nAnswer the following question\n\nQuestion: {Question}?"
        # completionn = self.generate_completion(prompt)
        flag=0
        completionn_time = time.time()
        stream = self._openai_client.chat.completions.create(
            model="gpt-3.5-turbo",
            stream=True,
            messages=[
                {"role": "system", "content": self.system_prompt}]+
                [{"role": m["role"], "content": m["content"]} for m in context_messages if m["content"] is not None]+
                [{"role": "user", "content": prompt}]
            
            # messages=messages,
        )
        for chunk in stream:
            if chunk.choices[0].delta.content is not None:
                print(chunk.choices[0].delta.content, end="")
                if flag ==0:
                    logger.info("completionn Time consuming: {:.4f}s".format(time.time()-completionn_time))
                    flag=1
                completionn= chunk.choices[0].delta.content
                await self.push_frame(LLMResponseStartFrame())
                await self.push_frame(TextFrame(completionn))
                await self.push_frame(LLMResponseEndFrame())
        # return completion.choices[0].message.content
        # completionn = completionn.choices[0].message.content
        logger.info("completionn Time consuming: {:.4f}s".format(time.time()-completionn_time))        


    async def langchain_rag(self,Question,session_id):
        response_time = time.time()
        flag =0
        async for chunk in self.conversational_rag_chain.astream(
                {"input": f"{Question}",
                 "chat_history":session_id
                 },
                config={"configurable": {"session_id": session_id}
                },
            ):
            logger.debug("----------")
            try:
                chunk = chunk["answer"]
                # print(chunk, end="|", flush=True)
                if flag==0:
                    logger.info("first response Time consuming: {:.4f}s".format(time.time() - response_time))
                    flag=1

                self._completion=chunk
                logger.debug(f"completion: {self._completion}")
                if self._completion != "":
                    await self.push_frame(LLMResponseStartFrame())
                    await self.push_frame(TextFrame(self._completion))
                    await self.push_frame(LLMResponseEndFrame())
                    self._completion=""
                    # flag=0
            except:
                pass


    # client = QdrantClient("http://172.18.0.4:6333", https=True)
    async def generate_response(self,context_messages,Question):
        # logger.debug(f"in generate response: {Question}")
        

        # await self.generate_rag(Question,context_messages)


        OPENAI_API_KEY=self.openai_key

        session_id="abc123"
 

        logger.debug(f"context messages: {context_messages}")
        logger.debug(f"context messages type: {type(context_messages)}")
        message_history_time = time.time()

        message_history = self.get_session_history(session_id).messages
        print("\nMessage History:")
        for message in message_history:
            role = "User" if isinstance(message, HumanMessage) else "Bot"
            print(f"{role}: {message.content}")
        print("\n")
        logger.info("get message history Time consuming: {:.4f}s".format(time.time() - message_history_time))


        await self.langchain_rag(Question,session_id)
           



    async def _process_context(self, context: OpenAILLMContext):
        await self.push_frame(LLMFullResponseStartFrame())
        try:
            logger.debug(f"Generating chat: {context.get_messages_json()}")
            
            logger.debug(f"context: {context}")
            # get the last object from the context
            user_message = context.get_messages()[-1]
            # messages: List[ChatCompletionMessageParam] = context.get_messages()
            # get context messages 
            context_messages = context.get_messages()
            self._save_bot_context(context_messages)

            # get the content from the user message
            user_message_content = user_message["content"]

            # log the messages
            logger.debug(f"User Input: [{user_message_content}]")

            await self.start_ttfb_metrics()

            # call stream dify response
            await self.generate_response(context_messages,user_message_content) ##trigger function

        except Exception as e:
            logger.exception(f"{self} exception: {e}")
        finally:
            await self.push_frame(LLMFullResponseEndFrame())

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        context = None

        if isinstance(frame, OpenAILLMContextFrame):
            context: OpenAILLMContext = frame.context
        elif isinstance(frame, LLMMessagesFrame):
            context = OpenAILLMContext.from_messages(frame.messages)
        elif isinstance(frame, VisionImageRawFrame):
            context = OpenAILLMContext.from_image_frame(frame)
        else:
            await self.push_frame(frame, direction)

        if context:
            await self._process_context(context)