
import requests
import asyncio
import ssl
from dotenv import load_dotenv
from livekit.agents import AutoSubscribe, JobContext, WorkerOptions, cli, llm
from livekit.agents.voice_assistant import VoiceAssistant
from livekit.plugins import openai, silero
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.memory import ConversationBufferMemory
from langchain.chains.llm import LLMChain
from langchain_mongodb.chat_message_histories import MongoDBChatMessageHistory
import os
import aiohttp 


# For testing purposes: This disables SSL verification
ssl._create_default_https_context = ssl._create_unverified_context

# Load environment variables
load_dotenv()

# MongoDB connection details
DATABASE_NAME = "ChatBotDB"
COLLECTION_NAME = "VoiceBot"
api_key = os.getenv("OPENAI_API_KEY")

# Initialize OpenAI model for Langchain
langchain_model = ChatOpenAI(
    model_name="gpt-4o-mini-2024-07-18",
    temperature=0.7,
    openai_api_key=api_key
)

# Define the initial prompt for Langchain processing
prompt = ChatPromptTemplate.from_messages([
    ("assistant", "You are Milla, an AI companion here to support the user by suggesting activities, exercises, or quotes that you and the user can do together together to support their progress and align with their counseling goals. These can range from light-hearted or fun activities to more introspective exercises.  Begin by warmly asking the user to share any reflections, emotions, or ideas that stood out to them since their last session. Maintain a tone that is compassionate, non-judgmental, and add a touch of wit whenever it feels appropriate, fostering a safe, engaging, and uplifting environment. Keep the conversation going. If the user doesn't like or wants to discontinue with a activity you switch up with a new activity for the user."),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}")
])



def fetch_email():
    try:
        response = requests.get("https://api.supermilla.com/email/get-email")
        response.raise_for_status()  # Check if request was successful
        data = response.json()
        email = data.get("email")
        
        if email:
            print("Fetched email from Node.js API:", email)
            return email
        else:
            print("No email found")
            return None
    except requests.exceptions.RequestException as e:
        print("Error fetching email:", e)
        return None
            
async def entrypoint(ctx: JobContext):
    system_msg = llm.ChatMessage(
        role="system",
        content=(
            "You are a voice assistant created by Copublica. Reply the context without making any changes. You should not make any modifiaction in the context."
        ),
    )
    initial_ctx = llm.ChatContext()
    initial_ctx.messages.append(system_msg)

    gmail_address = fetch_email()
    print(gmail_address)
    print(f"Fetched email: {gmail_address}")
    chat_message_history = MongoDBChatMessageHistory(
        session_id=gmail_address,
        connection_string="mongodb+srv://thesupermilla:79yhZPP3pIKCsQI9@cluster.bvidkli.mongodb.net/?retryWrites=true&w=majority&appName=Cluster",
        database_name=DATABASE_NAME,
        collection_name=COLLECTION_NAME
    )
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True, chat_memory=chat_message_history)

    # Define the Langchain LLMChain
    langchain_chain = LLMChain(prompt=prompt, llm=langchain_model, memory=memory, verbose=True)

    # Callback for handling conversation and refinement
    async def before_llm(assistant: VoiceAssistant, chat_ctx: llm.ChatContext):
      ctx_msg = system_msg.copy()
      user_msg = chat_ctx.messages[-1]
      print("user", user_msg)
      langchain_response = await langchain_chain.ainvoke(
                {"input": user_msg.content}
                )
      langchain_text = langchain_response['text']
      ctx_msg.content = "Context that might help answer the user's question:"
      ctx_msg.content += f"\n\n{langchain_text}"
      chat_ctx.messages[0] = ctx_msg
      return assistant.llm.chat(chat_ctx=chat_ctx)

    await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)
    assistant = VoiceAssistant(
        vad=silero.VAD.load(),
        stt=openai.STT(),
        will_synthesize_assistant_reply= before_llm,  
        llm=openai.LLM(),
        tts=openai.TTS(voice="nova"),
        chat_ctx=initial_ctx,
    )
    assistant.start(ctx.room)

    await assistant.say("Hey, how can I help you today!", allow_interruptions=True)



if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))
