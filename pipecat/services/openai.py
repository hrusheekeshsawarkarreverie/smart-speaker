import aiohttp
import base64
import io
import json
import os
from typing import AsyncGenerator, List, Literal, Optional, Union

from loguru import logger
from PIL import Image
# from litellm import completion
from pipecat.services.nmt import NMTService
from pipecat.frames.frames import (
    AudioRawFrame,
    ErrorFrame,
    Frame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    LLMMessagesFrame,
    LLMResponseEndFrame,
    LLMResponseStartFrame,
    TextFrame,
    URLImageRawFrame,
    VisionImageRawFrame,
)
from pipecat.processors.aggregators.openai_llm_context import (
    OpenAILLMContext,
    OpenAILLMContextFrame,
)
from pipecat.processors.frame_processor import FrameDirection
from pipecat.services.ai_services import ImageGenService, LLMService, TTSService
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import hashlib
import os
import time
from dotenv import load_dotenv

load_dotenv()

try:
    from openai import AsyncOpenAI, AsyncStream, BadRequestError, OpenAI,AsyncAzureOpenAI
    import redis
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

from qdrant_client import QdrantClient

# Load the SentenceTransformer model globally to avoid latency
_embedder = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

# Initialize the sentence transformer model
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

# _qdrant_client = QdrantClient(host=os.getenv('QDRANT_HOST', 'qdrant'), port=int(os.getenv('QDRANT_PORT', 6333)))

# set the qdrant client
if os.getenv("VECTOR_STORE", "redis").lower() == "qdrant":
    try:
        qdrant_host = os.getenv('QDRANT_HOST', 'qdrant')
        qdrant_port = int(os.getenv('QDRANT_PORT', 6333))
        _qdrant_client = QdrantClient(host=qdrant_host, port=qdrant_port)
        logger.debug(f"Qdrant client initialized with host: {qdrant_host}, port: {qdrant_port}")
        
        # Optional: Verify connection
        _qdrant_client.get_collections()
        logger.debug("Qdrant client connection verified.")
    except Exception as e:
        logger.error(f"Failed to initialize Qdrant client: {e}", exc_info=True)
        _qdrant_client = None
else:
    _qdrant_client = None

# semantic text prefix for caching
semantic_response_prefix = "semantic-response"
semantic_vector_prefix = "semantic-vector"

class OpenAIUnhandledFunctionException(Exception):
    pass


class CacheService:
    def __init__(self, redis_host="localhost", redis_port=6379, redis_db=0):
        self._redis_host = redis_host
        self._redis_port = redis_port
        self._redis_db = redis_db
        self._vector_store = os.getenv("VECTOR_STORE", "redis").lower()
        self._redis_client = self.create_redis_client()
        self._qdrant_client = _qdrant_client
        self._collection_name = 'cached_responses'
        self._embedder = _embedder
        self._cache_type = os.getenv("cache_type", "").lower()
        self._is_semantic_caching_enabled = self._cache_type == "semantic"
        self._is_text_caching_enabled = self._cache_type == "text"

        if self._vector_store == "qdrant":
            try:
                self._qdrant_client.get_collection(self._collection_name)
            except Exception as e:
                logger.info(f"Collection {self._collection_name} not found. Creating a new one.")
                self._qdrant_client.create_collection(
                    collection_name=self._collection_name,
                    vectors_config={
                        "size": 384,
                        "distance": "Cosine"
                    }
                )

    def create_redis_client(self):
        return redis.StrictRedis(host=self._redis_host, port=self._redis_port, db=self._redis_db)

    def generate_hash_key(self, conversation_context: str) -> str:
        return hashlib.md5(conversation_context.encode()).hexdigest()

    def compute_cosine_similarity(self, vec1, vec2):
        return cosine_similarity([vec1], [vec2])[0][0]

    def store_response_in_redis(self, conversation_context: str, response: str):
        hash_key = self.generate_hash_key(conversation_context)
        response_key = f"{semantic_response_prefix}:{hash_key}.txt"
        if not self._redis_client.exists(response_key):
            self._redis_client.set(response_key, response)

    def store_embedding_in_redis(self, conversation_context: str, response: str):
        hash_key = self.generate_hash_key(conversation_context)
        key = f"{semantic_vector_prefix}:{hash_key}"
        if not self._redis_client.exists(key):
            embedding = self.generate_embedding(conversation_context)
            self._redis_client.set(key, embedding.tobytes())
        self.store_response_in_redis(conversation_context, response)
        
    def store_embedding_in_qdrant(self, conversation_context: str, response: str):
        try:
            hash_key = self.generate_hash_key(conversation_context)
            logger.debug(f"Storing embedding in Qdrant for context: {conversation_context}")
            
            existing_data = self._qdrant_client.retrieve(
                collection_name=self._collection_name,
                ids=[hash_key]
            )
            logger.debug(f"Existing data: {existing_data}")
            
            if not existing_data:
                embedding = self.generate_embedding(conversation_context)
                logger.debug(f"Generated embedding: {embedding}")
                
                self._qdrant_client.upsert(
                    collection_name=self._collection_name,
                    points=[
                        {
                            'id': hash_key,
                            'vector': embedding,
                            'payload': {'conversation_context': conversation_context}
                        }
                    ]
                )
                self.store_response_in_redis(conversation_context, response)
            else:
                logger.debug(f"Data already present for hash_key: {hash_key}")
        except Exception as e:
            logger.error(f"Error in store_embedding_in_qdrant: {e}", exc_info=True)

    def generate_embedding_for_redis(self, text: str) -> np.ndarray:
        try:
            return self._embedder.encode([text])[0]
        except Exception as e:
            logger.error(f"Error generating embedding for Redis: {e}")
            return np.array([], dtype=np.float32)

    def generate_embedding_for_qdrant(self, text: str) -> np.ndarray:
        try:
            embedding = self._embedder.encode([text])[0]
            embedding = np.array(embedding, dtype=np.float32)
            return embedding
        except Exception as e:
            logger.error(f"Error generating embedding: {e}")
            return np.array([], dtype=np.float32)

    def generate_embedding(self, text: str) -> np.ndarray:
        try:
            if self._vector_store == "redis":
                embedding = self.generate_embedding_for_redis(text)
            elif self._vector_store == "qdrant":
                embedding = self.generate_embedding_for_qdrant(text)
            else:
                logger.error(f"Unknown vector store: {self._vector_store}")
                return np.array([], dtype=np.float32)
            return embedding
        except Exception as e:
            logger.error(f"Error generating embedding: {e}")
            return np.array([], dtype=np.float32)

    def get_cached_response_from_redis(self, key_hash: str) -> Optional[str]:
        key_hash = key_hash.replace("-", "")
        response_key = f"{semantic_response_prefix}:{key_hash}.txt"
        cached_response = self._redis_client.get(response_key)
        logger.debug(f"Cached response: {cached_response}")
        if cached_response:
            return cached_response.decode('utf-8')
        return None

    def fetch_redis_cached_response(self, sentence: str):
        try:
            prefix_pattern = f"{semantic_vector_prefix}:*"
            stored_keys = self._redis_client.keys(prefix_pattern)
            new_embedding = self.generate_embedding(sentence)
            if new_embedding.size == 0:
                logger.error("Generated embedding is empty.")
                return None
            for key in stored_keys:
                stored_embedding = np.frombuffer(self._redis_client.get(key), dtype=np.float32)
                if stored_embedding.size == 0:
                    logger.error(f"Stored embedding for key {key} is empty.")
                    continue
                similarity = self.compute_cosine_similarity(stored_embedding, new_embedding)
                if similarity > 0.9:
                    key_hash = key.decode('utf-8').split(":", 1)[1]
                    logger.debug(f"Key hash: {key_hash}")
                    return self.get_cached_response_from_redis(key_hash)
        except Exception as e:
            logger.error(f"Error in fetch_redis_cached_response: {e}")
        return None

    def get_semantic_cached_response(self, context: OpenAILLMContext) -> Optional[str]:
        try:
            messages = context.get_messages()
            if len(messages) < 1:
                return None
            conversation_context = self.get_conversation_context(messages)
            logger.debug(f"Vector store: {self._vector_store}")
            if self._vector_store == "redis":
                cached_response = self.fetch_redis_cached_response(conversation_context)
            elif self._vector_store == "qdrant":
                cached_response = self.fetch_qdrant_cached_response(conversation_context)
            else:
                logger.error(f"Unknown vector store: {self._vector_store}")
                cached_response = None
            logger.debug(f"cached response {cached_response}")
            return cached_response
        except Exception as e:
            logger.error(f"Error in get_semantic_cached_response: {e}")
        return None

    def get_conversation_context(self, messages: List[dict]) -> str:
        if len(messages) > 1 and messages[-2]["role"] == "assistant":
            return messages[-2]["content"] + " " + messages[-1]["content"]
        return messages[-1]["content"]

    def fetch_qdrant_cached_response(self, conversation_context: str) -> Optional[str]:
        try:
            logger.debug(f"Conversation context: {conversation_context}")
            search_result = self._qdrant_client.search(
                collection_name=self._collection_name,
                query_vector=self.generate_embedding(conversation_context),
                limit=1
            )
            logger.debug(f"Search result: {search_result}")
            if search_result and len(search_result) > 0:
                top_result = search_result[0]
                similarity_score = top_result.score
                if similarity_score >= 0.95:
                    key_hash = top_result.id
                    logger.debug(f"Key hash: {key_hash}")
                    search_result = self.get_cached_response_from_redis(key_hash)
                    logger.debug(f"Search result: {search_result}")
                    return search_result
        except Exception as e:
            logger.error(f"Error in fetch_qdrant_cached_response: {e}")
        return None

    def get_text_cached_response(self, context: OpenAILLMContext) -> Optional[str]:
        try:
            messages = context.get_messages()
            if len(messages) < 1:
                return None
            conversation_context = self.get_conversation_context(messages)
            logger.debug(f"Conversation context: {conversation_context}")
            hash_key = self.generate_hash_key(conversation_context)
            response_key = f"{hash_key}.txt"
            logger.debug(f"Response key: {response_key}")
            cached_response = self._redis_client.get(response_key)
            if cached_response:
                logger.debug("Serving response from cache")
                return cached_response.decode('utf-8')
        except Exception as e:
            logger.error(f"Error in get_text_cached_response: {e}")
        return None

    def get_cached_response(self, context: OpenAILLMContext) -> Optional[str]:
        if self._is_semantic_caching_enabled:
            text_response = self.get_text_cached_response(context)
            if text_response:
                return text_response
            return self.get_semantic_cached_response(context)
        elif self._is_text_caching_enabled:
            return self.get_text_cached_response(context)
        return None

    def store_embedding(self, conversation_context: str, response: str):
        if self._vector_store == "redis":
            self.store_embedding_in_redis(conversation_context, response)
        elif self._vector_store == "qdrant":
            self.store_embedding_in_qdrant(conversation_context, response)
        else:
            logger.error(f"Unknown vector store: {self._vector_store}")

    def cache_response(self, context: OpenAILLMContext, response: str):
        if not self._is_semantic_caching_enabled and not self._is_text_caching_enabled:
            return
        messages = context.get_messages()
        if len(messages) < 1:
            return
        conversation_context = self.get_conversation_context(messages)
        logger.debug(f"Caching response for context: {conversation_context}")
        logger.debug(f"Response: {response}")
        if self._is_semantic_caching_enabled:
            logger.debug(f"Storing new embedding for sentence: {conversation_context} with prefix: llm")
            self.store_embedding(conversation_context, response)
        elif self._is_text_caching_enabled:
            hash_key = self.generate_hash_key(conversation_context)
            self._redis_client.set(f"{hash_key}.txt", response)

class BaseOpenAILLMService(LLMService):
    """This is the base for all services that use the AsyncOpenAI client.

    This service consumes OpenAILLMContextFrame frames, which contain a reference
    to an OpenAILLMContext frame. The OpenAILLMContext object defines the context
    sent to the LLM for a completion. This includes user, assistant and system messages
    as well as tool choices and the tool, which is used if requesting function
    calls from the LLM.
    """

    def __init__(
            self,
            *,
            model: str,
            api_key=None,
            base_url=None,
            tgt_lan="",
            nmt_flag=False,
            nmt_provider="",
            redis_host: str = os.getenv('REDIS_HOST', 'redis'),
            redis_port: int = int(os.getenv('REDIS_PORT', 6379)),
            redis_db: int = 0,
            **kwargs
            ):
        super().__init__(**kwargs)
        self._model: str = model
        self._client = self.create_client(api_key=api_key, base_url=base_url, **kwargs)
        self._save_bot_context = kwargs.get("save_bot_context")
        self._tgt_lan = tgt_lan
        self._src_lan = "en"
        self._nmt_flag = nmt_flag
        self._nmt_provider = nmt_provider
        self._is_start_msg_sent = False
        self._frame = ""
        self._processed_text = ""
        self._vector_store = os.getenv("VECTOR_STORE", "redis").lower()
        self._redis_host = redis_host
        self._redis_port = redis_port
        self._redis_db = redis_db
        self._cache_type = os.getenv("cache_type", "").lower()
        self._is_semantic_caching_enabled = self._cache_type == "semantic"
        self._is_text_caching_enabled = self._cache_type == "text"
        if self._is_semantic_caching_enabled:
            self._embedder = _embedder
        self._redis_client = self.create_redis_client()
        self._qdrant_client = _qdrant_client
        self._collection_name = 'cached_responses'
        # Ensure the collection exists
        if self._vector_store == "qdrant":
            try:
                self._qdrant_client.get_collection(self._collection_name)
            except Exception as e:
                logger.info(f"Collection {self._collection_name} not found. Creating a new one.")
                self._qdrant_client.create_collection(
                collection_name=self._collection_name,
                vectors_config={
                    "size": 384,  # Adjust based on your embedding size
                    "distance": "Cosine"
                }
                )
            self._vector_store = os.getenv("VECTOR_STORE", "redis").lower()

    def create_client(self, api_key=None, base_url=None, **kwargs):
        return AsyncOpenAI(api_key=api_key, base_url=base_url)

    def create_redis_client(self):
        return redis.StrictRedis(host=self._redis_host, port=self._redis_port, db=self._redis_db)

    def can_generate_metrics(self) -> bool:
        return True

    # Utility function to compute cosine similarity between two vectors
    def compute_cosine_similarity(self, vec1, vec2):
        return cosine_similarity([vec1], [vec2])[0][0]
    
    
    def store_response_in_redis(self, conversation_context: str, response: str):
        hash_key = self.generate_hash_key(conversation_context)
        response_key = f"{semantic_response_prefix}:{hash_key}.txt"
        if not self._redis_client.exists(response_key):
            self._redis_client.set(response_key, response)
    
    # Store embeddings in Redis under a specific prefix
    def store_embedding_in_redis(self, conversation_context: str, response: str):
        hash_key = self.generate_hash_key(conversation_context)
        key = f"{semantic_vector_prefix}:{hash_key}"
        # Check if the embedding is already stored in Redis
        if not self._redis_client.exists(key):
            # Get the embedding for the conversation context
            embedding = self.generate_embedding(conversation_context)
            self._redis_client.set(key, embedding.tobytes())
        
        # store the response in redis
        self.store_response_in_redis(conversation_context, response)

        
        
    def store_embedding_in_qdrant(self, conversation_context: str, response: str):
        hash_key = self.generate_hash_key(conversation_context)
        
        # Log the conversation context and response
        logger.debug(f"Storing embedding in Qdrant for context: {conversation_context}")

        # Check if the data is already present based on the hash_key
        existing_data = self._qdrant_client.retrieve(
            collection_name=self._collection_name,
            ids=[hash_key]
        )

        if not existing_data:
            # Store the response in Qdrant
            self._qdrant_client.upsert(
            collection_name=self._collection_name,
            points=[
                {
                'id': hash_key,
                'vector': self.generate_embedding(conversation_context),
                'payload': {'conversation_context': conversation_context}
                }
            ]
            )
            
            # store the response in redis so that later we can retrieve it based on the hash key
            self.store_response_in_redis(conversation_context, response)
        
        else:
            logger.debug(f"Data already present for hash_key: {hash_key}")
            

    # Generate embeddings for the redis db
    def generate_embedding_for_redis(self, text: str) -> np.ndarray:
        try:
            # new_embedding = model.encode([text])[0]
            return self._embedder.encode([text])[0]
        except Exception as e:
            logger.error(f"Error generating embedding for Redis: {e}")
            return np.array([], dtype=np.float32)
        
    def generate_embedding_for_qdrant(self, text: str) -> np.ndarray:
        try:
            embedding = self._embedder.encode([text])[0]
            embedding = np.array(embedding, dtype=np.float32)  # Ensure embedding is a numpy array with float32 type
            return embedding
        except Exception as e:
            logger.error(f"Error generating embedding: {e}")
            return np.array([], dtype=np.float32)

    def generate_embedding(self, text: str) -> np.ndarray:
        try:
            if self._vector_store == "redis":
                embedding = self.generate_embedding_for_redis(text)
            elif self._vector_store == "qdrant":
                embedding = self.generate_embedding_for_qdrant(text)
            else:
                logger.error(f"Unknown vector store: {self._vector_store}")
                return np.array([], dtype=np.float32)
            return embedding
        except Exception as e:
            logger.error(f"Error generating embedding: {e}")
            return np.array([], dtype=np.float32)
        
    # Get the response from Redis based on the key hash
    def get_cached_response_from_redis(self, key_hash: str) -> Optional[str]:
        
        key_hash = key_hash.replace("-", "")
        
        response_key = f"{semantic_response_prefix}:{key_hash}.txt"
        cached_response = self._redis_client.get(response_key)
        
        # log the cached response
        logger.debug(f"Cached response: {cached_response}")
        
        if cached_response:
            return cached_response.decode('utf-8')
        return None
        
    # Retrieve embeddings from Redis prefix and check cosine similarity
    def fetch_redis_cached_response(self, sentence: str):
        try:
            prefix_pattern = f"{semantic_vector_prefix}:*"
            stored_keys = self._redis_client.keys(prefix_pattern)  # Get all stored keys with the prefix
            new_embedding = self.generate_embedding(sentence)  # Generate embedding for new sentence

            if new_embedding.size == 0:
                logger.error("Generated embedding is empty.")
                return None

            for key in stored_keys:
                stored_embedding = np.frombuffer(self._redis_client.get(key), dtype=np.float32)

                if stored_embedding.size == 0:
                    logger.error(f"Stored embedding for key {key} is empty.")
                    continue

                # Compute cosine similarity with stored embeddings
                similarity = self.compute_cosine_similarity(stored_embedding, new_embedding)

                if similarity > 0.9:  # Define a threshold for similarity (0.9 here)
                    key_hash = key.decode('utf-8').split(":", 1)[1]  # Extract the sentence part from the key
                    
                    logger.debug(f"Key hash: {key_hash}")
                    
                    # Return the cached response based on the key hash
                    return self.get_cached_response_from_redis(key_hash)
                    
        except Exception as e:
            logger.error(f"Error in fetch_redis_cached_response: {e}")
        return None
        
    def get_semantic_cached_response(self, context: OpenAILLMContext) -> Optional[str]:
        try:
            messages = context.get_messages()
            if len(messages) < 1:
                return None    
                
            # get the conversation context
            conversation_context = self.get_conversation_context(messages)
            
            # log the vectore store
            logger.debug(f"Vector store: {self._vector_store}")
            
            if self._vector_store == "redis":
                # get cached response based on embedding from Redis
                cached_response = self.fetch_redis_cached_response(conversation_context)
            elif self._vector_store == "qdrant":
                # get cached response based on embedding from Qdrant
                cached_response = self.fetch_qdrant_cached_response(conversation_context)
            else:
                logger.error(f"Unknown vector store: {self._vector_store}")
                cached_response = None

            logger.debug(f"cached response {cached_response}")
            
            return cached_response

        except Exception as e:
            logger.error(f"Error in get_semantic_cached_response: {e}")
        return None


    def get_conversation_context(self, messages: List[dict]) -> str:
            """Get the conversation context for caching."""
            if len(messages) > 1 and messages[-2]["role"] == "assistant":
                return messages[-2]["content"] + " " + messages[-1]["content"]
            return messages[-1]["content"]

    def generate_hash_key(self, conversation_context: str) -> str:
        return hashlib.md5(conversation_context.encode()).hexdigest()
    
    def fetch_redis_text_cached_response(self, redis_client, response_key):
        return redis_client.get(response_key)
    
    def fetch_qdrant_cached_response(self, conversation_context: str) -> Optional[str]:
        try:
            # Log the conversation context
            logger.debug(f"Conversation context: {conversation_context}")
            
            # Search for the cached response in Qdrant
            search_result = self._qdrant_client.search(
                collection_name=self._collection_name,
                query_vector=self.generate_embedding(conversation_context),
                limit=1  # We only need the top result
            )
            
            logger.debug(f"Search result: {search_result}")

            # Check if the similarity score is more than 90 percent
            if search_result and len(search_result) > 0:
                top_result = search_result[0]
                similarity_score = top_result.score  # Access the score attribute
                if similarity_score >= 0.95:
                    # get the id of the record
                    key_hash = top_result.id
                    
                    # get the response based on the id from the redis
                    logger.debug(f"Key hash: {key_hash}")
                    
                    # Return the cached response based on the key hash
                    search_result = self.get_cached_response_from_redis(key_hash)
                    
                    # Log the search result
                    logger.debug(f"Search result: {search_result}")
                    
                    return search_result
        except Exception as e:
            logger.error(f"Error in fetch_qdrant_cached_response: {e}")
        return None

    def get_text_cached_response(self, context: OpenAILLMContext) -> Optional[str]:
        """
        Retrieve a cached text response based on the provided context.
        
        Args:
            context (OpenAILLMContext): The context for which the cached response is to be retrieved.
        
        Returns:
            Optional[str]: The cached response if found, otherwise None.
        """
        try:
            # Get the list of messages from the context
            messages = context.get_messages()
            
            # If there are no messages, return None
            if len(messages) < 1:
                return None

            # Use the get_conversation_context function to get the conversation context
            conversation_context = self.get_conversation_context(messages)
                
            # Log the conversation context for debugging purposes
            logger.debug(f"Conversation context: {conversation_context}")
                            
            # Generate a hash key for the conversation context
            hash_key = self.generate_hash_key(conversation_context)
            
            # Create the response key using the hash key
            response_key = f"{hash_key}.txt"
            
            # Log the response key for debugging purposes
            logger.debug(f"Response key: {response_key}")

            # Fetch the cached response from Redis using the response key
            cached_response = self.fetch_redis_text_cached_response(self._redis_client, response_key)
            
            # If a cached response is found, log it and return the decoded response
            if cached_response:
                logger.debug("Serving response from cache")
                return cached_response.decode('utf-8')
        except Exception as e:
            # Log any exceptions that occur during the process
            logger.error(f"Error in get_text_cached_response: {e}")
        
        # Return None if no cached response is found or an error occurs
        return None

    def get_cached_response(self, context: OpenAILLMContext) -> Optional[str]:
        """
        Retrieve a cached response based on the provided context.
        This method checks if semantic caching is enabled and attempts to 
        retrieve a cached response using semantic caching first. If a 
        semantic cached response is not found, it falls back to text 
        caching. If semantic caching is not enabled, it directly attempts 
        to retrieve a cached response using text caching.
        Args:
            context (OpenAILLMContext): The context for which the cached 
            response is to be retrieved.
        Returns:
            Optional[str]: The cached response if found, otherwise None.
        """
        # Implementation here
        if self._is_semantic_caching_enabled:
            text_response = self.get_text_cached_response(context)
            if text_response:
                return text_response
            
            return self.get_semantic_cached_response(context)
        elif self._is_text_caching_enabled:
            return self.get_text_cached_response(context)
        return None
    
    def store_embedding(self, conversation_context: str, response: str):
        """
        Store the embedding of the conversation context and the response in the appropriate vector store.
        
        Args:
            conversation_context (str): The context of the conversation.
            response (str): The response to be stored.
        """
        if self._vector_store == "redis":
            # Store the embedding and response in Redis
            self.store_embedding_in_redis(conversation_context, response)
        elif self._vector_store == "qdrant":
            # Store the embedding and response in Qdrant
            self.store_embedding_in_qdrant(conversation_context, response)
        else:
            # Log an error if the vector store is unknown
            logger.error(f"Unknown vector store: {self._vector_store}")
    

    def cache_response(self, context: OpenAILLMContext, response: str):
        if not self._is_semantic_caching_enabled and not self._is_text_caching_enabled:
            return

        messages = context.get_messages()
        if len(messages) < 1:
            return

        # Use only the last user message for caching
        conversation_context = self.get_conversation_context(messages)
        
        # log that the response is being cached along with the conversation context and response
        logger.debug(f"Caching response for context: {conversation_context}")
        logger.debug(f"Response: {response}")
        
        if self._is_semantic_caching_enabled:
            logger.debug(f"Storing new embedding for sentence: {conversation_context} with prefix: llm")

            # If no similar sentence is found, store the new embedding and return None
            self.store_embedding(conversation_context, response)
                
        elif self._is_text_caching_enabled:
            hash_key = self.generate_hash_key(conversation_context)
            self._redis_client.set(f"{hash_key}.txt", response)

    async def get_chat_completions(
        self, context: OpenAILLMContext, messages: List[ChatCompletionMessageParam]
    ) -> AsyncStream[ChatCompletionChunk]:
        cached_response = self.get_cached_response(context)
        if cached_response:            
            # log the cached response
            logger.debug(f"Cached response: {cached_response}")
            
            async def cached_response_stream():
                yield ChatCompletionChunk(
                    id="cached_response",
                    object="chat.completion.chunk",
                    created=int(time.time()),
                    model=self._model,
                    choices=[{"index": 0, "delta": {"content": cached_response}}]
                )
            return cached_response_stream()

        # logger.debug(f"Cache miss for context: {context.get_messages_json()}")
        chunks = await self._client.chat.completions.create(
            model=self._model,
            stream=True,
            messages=messages,
            tools=context.tools,
            tool_choice=context.tool_choice,
            temperature=0,
            top_p=0.3
        )
        
        # log one chunk
        # logger.debug(f"Chunk: {chunks}")
        
        return chunks
        
    async def _stream_chat_completions(
        self, context: OpenAILLMContext
    ) -> AsyncStream[ChatCompletionChunk]:
        logger.debug(f"Generating chat: ... {context.get_messages_json()[-1]}")

        messages: List[ChatCompletionMessageParam] = context.get_messages()

        # save the bot context messages to the context
        self._save_bot_context(messages)

        # base64 encode any images
        for message in messages:
            if message.get("mime_type") == "image/jpeg":
                encoded_image = base64.b64encode(message["data"].getvalue()).decode(
                    "utf-8"
                )
                text = message["content"]
                message["content"] = [
                    {"type": "text", "text": text},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{encoded_image}"},
                    },
                ]
                del message["data"]
                del message["mime_type"]

        chunks = await self.get_chat_completions(context, messages)
        return chunks

    async def _process_context(self, context: OpenAILLMContext):
        function_name = ""
        arguments = ""
        tool_call_id = ""

        await self.start_ttfb_metrics()

        chunk_stream: AsyncStream[ChatCompletionChunk] = (
            await self._stream_chat_completions(context)
        )

        full_response = ""  # Initialize full_response to accumulate the response

        async for chunk in chunk_stream:
            await self.stop_ttfb_metrics()

            if chunk.choices[0].delta.tool_calls:
                tool_call = chunk.choices[0].delta.tool_calls[0]
                if tool_call.function and tool_call.function.name:
                    function_name += tool_call.function.name
                    tool_call_id = tool_call.id
                    await self.call_start_function(function_name)
                if tool_call.function and tool_call.function.arguments:
                    arguments += tool_call.function.arguments

            if self._nmt_flag == True:
                if chunk.choices[0].delta.content is not None:
                    self._frame += chunk.choices[0].delta.content

                if self._frame.strip().endswith(
                    (".", "?", "!", "|", "।")) and not self._frame.strip().endswith(
                    ("Mr,", "Mrs.", "Ms.", "Dr.")):
                    text = self._frame
                    text = text.replace("*", "")
                    logger.debug(f"consolidated: {text}")
                    translator = NMTService(text, self._tgt_lan, self._nmt_provider)
                    processed_text = await translator.translate()
                    self._processed_text = processed_text
                    logger.debug(f"processed_text: {self._processed_text}")
                    self._frame = ""
            else:
                if chunk.choices[0].delta.content is not None:
                    cleaned_text = chunk.choices[0].delta.content.replace("*", "")
                    self._processed_text = cleaned_text
                else:
                    self._processed_text = chunk.choices[0].delta.content

            if self._processed_text:
                await self.push_frame(LLMResponseStartFrame())
                await self.push_frame(TextFrame(self._processed_text))
                await self.push_frame(LLMResponseEndFrame())
                full_response += self._processed_text  # Accumulate the response
                self._processed_text = ""

        if function_name and arguments:
            if self.has_function(function_name):
                await self._handle_function_call(
                    context, tool_call_id, function_name, arguments
                )
            else:
                raise OpenAIUnhandledFunctionException(
                    f"The LLM tried to call a function named '{function_name}', but there isn't a callback registered for that function."
                )

        # Cache the full response
        self.cache_response(context, full_response)

    async def _handle_function_call(
        self, context, tool_call_id, function_name, arguments
    ):
        arguments = json.loads(arguments)
        result = await self.call_function(function_name, arguments)
        arguments = json.dumps(arguments)
        if isinstance(result, (str, dict)):
            tool_call = ChatCompletionFunctionMessageParam(
                {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "id": tool_call_id,
                            "function": {"arguments": arguments, "name": function_name},
                            "type": "function",
                        }
                    ],
                }
            )
            context.add_message(tool_call)
            if isinstance(result, dict):
                result = json.dumps(result)
            tool_result = ChatCompletionToolParam(
                {"tool_call_id": tool_call_id, "role": "tool", "content": result}
            )
            context.add_message(tool_result)
            await self._process_context(context)
        elif isinstance(result, list):
            for msg in result:
                context.add_message(msg)
            await self._process_context(context)
        elif isinstance(result, type(None)):
            pass
        else:
            raise TypeError(
                f"Unknown return type from function callback: {type(result)}"
            )

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
            await self.push_frame(LLMFullResponseStartFrame())
            await self.start_processing_metrics()
            await self._process_context(context)
            await self.stop_processing_metrics()
            await self.push_frame(LLMFullResponseEndFrame())
        # else:
        #     if self._is_start_msg_sent == False:
        #         self._is_start_msg_sent = True
        #         await self.push_frame(LLMFullResponseStartFrame())
        #         await self.start_processing_metrics()
        #         await self.push_frame(TextFrame("नमस्ते!"))
        #         await self.stop_processing_metrics()
        #         await self.push_frame(LLMFullResponseEndFrame())
        #         self._is_start_msg_sent = True


class AzureOpenAILLMService(BaseOpenAILLMService):
    """This service uses Azure OpenAI to interact with the LLM."""

    # def __init__(self, *, model: str, api_key: str, endpoint: str, **kwargs):
    #     super().__init__(model=model, api_key=api_key, base_url=endpoint, **kwargs)
    #     self._endpoint = endpoint
        
    def __init__(
        self,
        *,
        api_key: str,
        endpoint: str,
        model: str,
        api_version: str = "2023-12-01-preview", **kwargs):
        
        # log the api_key, endpoint, model and api_version
        logger.debug(f"api_key: {api_key}, endpoint: {endpoint}, model: {model}, api_version: {api_version}")
        
        # Initialize variables before calling parent __init__() because that
        # will call create_client() and we need those values there.
        self._endpoint = endpoint
        self._api_version = api_version
        # super().__init__(api_key=api_key, model=model)
        super().__init__(model=model,api_key=api_key,base_url=endpoint, **kwargs)
        self._model: str = model
        self._client = self.create_client(api_key=api_key, base_url=endpoint, **kwargs)
        self._save_bot_context = kwargs.get("save_bot_context")
        self._is_start_msg_sent = False

    # def create_client(self, api_key=None, base_url=None, **kwargs):
    #     return AsyncOpenAI(api_key=api_key, base_url=base_url)
    
    def create_client(self, api_key=None, base_url=None, **kwargs):
        
        # log the api key, azure endpoint and api version
        logger.debug(f"API Key: {api_key}")
        logger.debug(f"Azure Endpoint: {self._endpoint}")
        logger.debug(f"API Version: {self._api_version}")
        
        return AsyncAzureOpenAI(
            api_key=api_key,
            azure_endpoint=self._endpoint,
            api_version=self._api_version,
        )

    async def get_chat_completions(
        self, context: OpenAILLMContext, messages: List[ChatCompletionMessageParam]
    ) -> AsyncStream[ChatCompletionChunk]:
        chunks = await self._client.chat.completions.create(
            model=self._model,
            stream=True,
            messages=messages,
            tools=context.tools,
            tool_choice=context.tool_choice,
        )        
        return chunks

    async def _stream_chat_completions(
        self, context: OpenAILLMContext
    ) -> AsyncStream[ChatCompletionChunk]:
        logger.debug(f"Generating chat with Azure OpenAI: ... {context.get_messages_json()[-1]}")

        messages: List[ChatCompletionMessageParam] = context.get_messages()

        self._save_bot_context(messages)

        for message in messages:
            if message.get("mime_type") == "image/jpeg":
                encoded_image = base64.b64encode(message["data"].getvalue()).decode("utf-8")
                text = message["content"]
                message["content"] = [
                    {"type": "text", "text": text},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encoded_image}"}},
                ]
                del message["data"]
                del message["mime_type"]

        chunks = await self.get_chat_completions(context, messages)
        return chunks

    async def _process_context(self, context: OpenAILLMContext):
        
        function_name = ""
        arguments = ""
        tool_call_id = ""

        await self.start_ttfb_metrics()

        chunk_stream: AsyncStream[ChatCompletionChunk] = await self._stream_chat_completions(context)

        async for chunk in chunk_stream:
            await self.stop_ttfb_metrics()

            if chunk.choices and chunk.choices[0].delta.tool_calls:
                tool_call = chunk.choices[0].delta.tool_calls[0]
                if tool_call.function and tool_call.function.name:
                    function_name += tool_call.function.name
                    tool_call_id = tool_call.id
                    await self.call_start_function(function_name)
                if tool_call.function and tool_call.function.arguments:
                    arguments += tool_call.function.arguments
            elif chunk.choices and chunk.choices[0].delta.content is not None:
                await self.push_frame(LLMResponseStartFrame())
                await self.push_frame(TextFrame(chunk.choices[0].delta.content))
                await self.push_frame(LLMResponseEndFrame())

        if function_name and arguments:
            if self.has_function(function_name):
                await self._handle_function_call(context, tool_call_id, function_name, arguments)
            else:
                raise OpenAIUnhandledFunctionException(
                    f"The LLM tried to call a function named '{function_name}', but there isn't a callback registered for that function."
                )


# class LiteLLMService(LLMService):
#     """
#     This service uses LiteLLM to interact with multiple LLM providers using a consistent interface.
#     """

#     def __init__(self, *, model: str, api_key: Optional[str] = None, **kwargs):
#         super().__init__(**kwargs)
#         self._model: str = model
#         self._api_key = api_key
#         if api_key:
#             litellm.api_key = api_key
#         self._total_cost = 0

#     async def get_chat_completions(
#         self, context: OpenAILLMContext, messages: List[dict]
#     ) -> AsyncGenerator[dict, None]:
#         try:
#             response = completion(
#                 model=self._model,
#                 messages=messages,
#                 stream=True,
#             )
#             async for chunk in response:
#                 yield chunk

#             # Calculate and update the cost after the completion
#             usage = response.usage
#             if usage:
#                 cost = litellm.completion_cost(completion_response=response)
#                 self._total_cost += cost
#                 logger.info(
#                     f"Completion cost: ${cost:.6f}, Total cost: ${self._total_cost:.6f}"
#                 )
#         except Exception as e:
#             logger.error(f"Error in LiteLLM completion: {e}")
#             raise

#     async def _stream_chat_completions(
#         self, context: OpenAILLMContext
#     ) -> AsyncGenerator[dict, None]:
#         logger.debug(
#             f"Generating chat with LiteLLM: ... {context.get_messages_json()[-1]}"
#         )

#         messages: List[dict] = context.get_messages()

#         async for chunk in self.get_chat_completions(context, messages):
#             yield chunk

#     async def _process_context(self, context: OpenAILLMContext):
#         await self.push_frame(LLMResponseStartFrame())

#         full_response = ""
#         async for chunk in self._stream_chat_completions(context):
#             if "choices" in chunk and len(chunk["choices"]) > 0:
#                 content = chunk["choices"][0].get("delta", {}).get("content", "")
#                 if content:
#                     full_response += content
#                     await self.push_frame(TextFrame(content))

#         await self.push_frame(LLMResponseEndFrame())

#         # Add the full response to the context
#         context.add_message({"role": "assistant", "content": full_response})

#     async def process_frame(self, frame: Frame, direction: FrameDirection):
#         await super().process_frame(frame, direction)

#         context = None
#         if isinstance(frame, OpenAILLMContextFrame):
#             context: OpenAILLMContext = frame.context
#         elif isinstance(frame, LLMMessagesFrame):
#             context = OpenAILLMContext.from_messages(frame.messages)
#         elif isinstance(frame, VisionImageRawFrame):
#             context = OpenAILLMContext.from_image_frame(frame)
#         else:
#             await self.push_frame(frame, direction)

#         if context:
#             await self.push_frame(LLMFullResponseStartFrame())
#             await self.start_processing_metrics()
#             await self._process_context(context)
#             await self.stop_processing_metrics()
#             await self.push_frame(LLMFullResponseEndFrame())

#     def get_total_cost(self) -> float:
#         return self._total_cost


class BaseKrutrimOpenAILLMService(LLMService):
    """This is the base for all services that use the AsyncOpenAI client.

    This service consumes OpenAILLMContextFrame frames, which contain a reference
    to an OpenAILLMContext frame. The OpenAILLMContext object defines the context
    sent to the LLM for a completion. This includes user, assistant and system messages
    as well as tool choices and the tool, which is used if requesting function
    calls from the LLM.
    """

    def __init__(self, *, model: str, api_key=None, base_url=None, **kwargs):
        super().__init__(**kwargs)
        self._model: str = model
        # self.results=self.get_krutrim_chat_completions("tell me poem in kannada language")
        # self._client = self.create_client(api_key=api_key, base_url=base_url, **kwargs)

    # def get_krutrim_chat_completions(messages: List[ChatCompletionMessageParam]) -> AsyncStream[ChatCompletionChunk]:
    def get_krutrim_chat_completions(
        self, context: OpenAILLMContext, messages: List[ChatCompletionMessageParam]
    ) -> AsyncStream[ChatCompletionChunk]:
        print(f"messages: {messages}")
        openai = OpenAI(
            api_key="K5P_f+Dy9Z_lzs3.R_6XklzW",
            base_url="https://cloud.olakrutrim.com/v1",
        )

        chat_completion = openai.chat.completions.create(
            model="Krutrim-spectre-v2",
            # messages=[
            #     {"role": "system", "content": "You are a helpful assistant."},
            #     {"role": "user", "content": "{messages}"}
            # ]
            messages=messages,
            max_tokens=1024,
            # stream= False, # Optional, Defaults to false
            # temperature= 0, # Optional, Defaults to 1. Range: 0 to 2
        )
        # print(chat_completion.choices[0].message.content)
        print(chat_completion.choices[0].message.content.encode("utf-8").decode())
        # print(chat_completion)
        # return chat_completion
        return chat_completion.choices[0].message.content.encode("utf-8").decode()

    async def _stream_chat_completions_k(
        self, context: OpenAILLMContext
    ) -> AsyncStream[ChatCompletionChunk]:
        # logger.debug(f"Generating chat: ... {context.get_messages_json()[-1]}")

        messages: List[ChatCompletionMessageParam] = context.get_messages()

        #     # base64 encode any images
        #     for message in messages:
        #         if message.get("mime_type") == "image/jpeg":
        #             encoded_image = base64.b64encode(message["data"].getvalue()).decode("utf-8")
        #             text = message["content"]
        #             message["content"] = [
        #                 {"type": "text", "text": text},
        #                 {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encoded_image}"}}
        #             ]
        #             del message["data"]
        #             del message["mime_type"]

        # chunks = await self.get_chat_completions(context, messages)
        chunks = self.get_krutrim_chat_completions(context, messages)
        print(f"chunkss {chunks}")
        # print(f'chunkss f call {self.get_krutrim_chat_completions(context,messages)}')
        return chunks

    async def _process_context(self, context: OpenAILLMContext):
        function_name = ""
        arguments = ""
        tool_call_id = ""

        await self.start_ttfb_metrics()

        # chunk_stream: AsyncStream[ChatCompletionChunk] = (
        #     await self._stream_chat_completions_k(context)
        # )
        LLM_response = await self._stream_chat_completions_k(context)

        # for chunk in LLM_response:
        #         if not chunk.choices:
        #             continue
        #         content = chunk.choices[0].delta.content
        #         if content:
        await self.push_frame(LLMResponseStartFrame())
        await self.push_frame(TextFrame(LLM_response))
        await self.push_frame(LLMResponseEndFrame())

        # for chunk in chunk_stream:
        #     await self.stop_ttfb_metrics()

        #     if chunk.choices[0].delta.tool_calls:
        #         # We're streaming the LLM response to enable the fastest response times.
        #         # For text, we just yield each chunk as we receive it and count on consumers
        #         # to do whatever coalescing they need (eg. to pass full sentences to TTS)
        #         #
        #         # If the LLM is a function call, we'll do some coalescing here.
        #         # If the response contains a function name, we'll yield a frame to tell consumers
        #         # that they can start preparing to call the function with that name.
        #         # We accumulate all the arguments for the rest of the streamed response, then when
        #         # the response is done, we package up all the arguments and the function name and
        #         # yield a frame containing the function name and the arguments.

        #         tool_call = chunk.choices[0].delta.tool_calls[0]
        #         if tool_call.function and tool_call.function.name:
        #             function_name += tool_call.function.name
        #             tool_call_id = tool_call.id
        #             await self.call_start_function(function_name)
        #         if tool_call.function and tool_call.function.arguments:
        #             # Keep iterating through the response to collect all the argument fragments
        #             arguments += tool_call.function.arguments
        #     elif chunk.choices[0].delta.content:
        #         await self.push_frame(LLMResponseStartFrame())
        #         await self.push_frame(TextFrame(chunk.choices[0].delta.content))
        #         await self.push_frame(LLMResponseEndFrame())

        # if we got a function name and arguments, check to see if it's a function with
        # a registered handler. If so, run the registered callback, save the result to
        # the context, and re-prompt to get a chat answer. If we don't have a registered
        # handler, raise an exception.
        if function_name and arguments:
            if self.has_function(function_name):
                await self._handle_function_call(
                    context, tool_call_id, function_name, arguments
                )
            else:
                raise OpenAIUnhandledFunctionException(
                    f"The LLM tried to call a function named '{function_name}', but there isn't a callback registered for that function."
                )

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
            await self.push_frame(LLMFullResponseStartFrame())
            await self.start_processing_metrics()
            await self._process_context(context)
            await self.stop_processing_metrics()
            await self.push_frame(LLMFullResponseEndFrame())

    # def create_client(self, api_key=None, base_url=None, **kwargs):
    #     return OpenAI(api_key=api_key, base_url=base_url)

    # def get_chat_completions():
    #     chat_completion = self._client.chat.completions.create(
    #         model="Krutrim-spectre-v2",
    #         messages=[
    #             {"role": "system", "content": "You are a helpful assistant."},
    #             {"role": "user", "content": "Hello"}
    #         ]
    #     )
    #     print(chat_completion.choices[0].message.content)
    #     res = chat_completion.choices[0].message.content
    #     return res


class OpenAILLMService(BaseOpenAILLMService):

    def __init__(self, *, model: str = "gpt-4o", **kwargs):
        super().__init__(model=model, **kwargs)


class KrutrimLLMService(BaseKrutrimOpenAILLMService):

    def __init__(self, *, model: str = "gpt-4o", **kwargs):
        super().__init__(model=model, **kwargs)


class OpenAIImageGenService(ImageGenService):

    def __init__(
        self,
        *,
        image_size: Literal[
            "256x256", "512x512", "1024x1024", "1792x1024", "1024x1792"
        ],
        aiohttp_session: aiohttp.ClientSession,
        api_key: str,
        model: str = "dall-e-3",
    ):
        super().__init__()
        self._model = model
        self._image_size = image_size
        self._client = AsyncOpenAI(api_key=api_key)
        self._aiohttp_session = aiohttp_session

    async def run_image_gen(self, prompt: str) -> AsyncGenerator[Frame, None]:
        logger.debug(f"Generating image from prompt: {prompt}")

        image = await self._client.images.generate(
            prompt=prompt, model=self._model, n=1, size=self._image_size
        )

        image_url = image.data[0].url

        if not image_url:
            logger.error(f"{self} No image provided in response: {image}")
            yield ErrorFrame("Image generation failed")
            return

        # Load the image from the url
        async with self._aiohttp_session.get(image_url) as response:
            image_stream = io.BytesIO(await response.content.read())
            image = Image.open(image_stream)
            frame = URLImageRawFrame(
                image_url, image.tobytes(), image.size, image.format
            )
            yield frame


class OpenAITTSService(TTSService):
    """This service uses the OpenAI TTS API to generate audio from text.
    The returned audio is PCM encoded at 24kHz. When using the DailyTransport, set the sample rate in the DailyParams accordingly:
    ```
    DailyParams(
        audio_out_enabled=True,
        audio_out_sample_rate=24_000,
    )
    ```
    """

    def __init__(
        self,
        *,
        api_key: str | None = None,
        voice: Literal["alloy", "echo", "fable", "onyx", "nova", "shimmer"] = "alloy",
        model: Literal["tts-1", "tts-1-hd"] = "tts-1",
        **kwargs,
    ):
        super().__init__(**kwargs)

        self._voice = voice
        self._model = model

        self._client = AsyncOpenAI(api_key=api_key)

    def can_generate_metrics(self) -> bool:
        return True

    async def run_tts(self, text: str) -> AsyncGenerator[Frame, None]:
        logger.debug(f"Generating TTS: [{text}]")

        try:
            await self.start_ttfb_metrics()

            async with self._client.audio.speech.with_streaming_response.create(
                input=text,
                model=self._model,
                voice=self._voice,
                response_format="pcm",
            ) as r:
                if r.status_code != 200:
                    error = await r.text()
                    logger.error(
                        f"{self} error getting audio (status: {r.status_code}, error: {error})"
                    )
                    yield ErrorFrame(
                        f"Error getting audio (status: {r.status_code}, error: {error})"
                    )
                    return
                async for chunk in r.iter_bytes(8192):
                    if len(chunk) > 0:
                        await self.stop_ttfb_metrics()
                        frame = AudioRawFrame(chunk, 24_000, 1)
                        yield frame
        except BadRequestError as e:
            logger.exception(f"{self} error generating TTS: {e}")
