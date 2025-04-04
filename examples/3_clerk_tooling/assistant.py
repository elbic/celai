"""
Smoothy Inc Clerk Assistant with RAG
-------------------------------------

This is a simple example of an AI Assistant implemented using the Cel.ai framework.
It serves as a basic demonstration of how to get started with Cel.ai for creating intelligent assistants.

Framework: Cel.ai
License: MIT License

This script is part of the Cel.ai example series and is intended for educational purposes.

Usage:
------
Configure the required environment variables in a .env file in the root directory of the project.
The required environment variables are:
- WEBHOOK_URL: The webhook URL for the assistant, you can use ngrok to create a public URL for your local server.
- TELEGRAM_TOKEN: The Telegram bot token for the assistant. You can get this from the BotFather on Telegram.

Then run this script to see a basic AI assistant in action.

Note:
-----
Please ensure you have the Cel.ai framework installed in your Python environment prior to running this script.
"""
# LOAD ENV VARIABLES
import os
import time
from loguru import logger as log
# Load .env variables
from dotenv import load_dotenv

from cel.assistants.request_context import RequestContext
load_dotenv()


# REMOVE THIS BLOCK IF YOU ARE USING THIS SCRIPT AS A TEMPLATE
# -------------------------------------------------------------
import sys
from pathlib import Path
# Add parent directory to path
path = Path(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(str(path.parents[1]))
# -------------------------------------------------------------

# Import Cel.ai modules
from cel.connectors.telegram import TelegramConnector
from cel.gateway.message_gateway import MessageGateway, StreamMode
from cel.message_enhancers.smart_message_enhancer_openai import SmartMessageEnhancerOpenAI
from cel.assistants.macaw.macaw_assistant import MacawAssistant
from cel.prompt.prompt_template import PromptTemplate
from cel.rag.providers.markdown_rag import MarkdownRAG
from cel.assistants.function_context import FunctionContext
from cel.assistants.function_response import RequestMode
from cel.assistants.common import Param

from datetime import datetime
# date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def get_current_date():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    


# Setup prompt
prompt = """You are an AI Assistant called Jane.
Today: {date}
Keep responses short and to the point.
Don't use markdown formatting in your responses. No contestes nada que no tenga que ver con Smoothy Inc.
You work Smoothy Inc. is a company that specializes in creating smoothies in food trucks.
Available products from Smoothy Inc. are:
    - Smoothies: 
        - Small $5
        - Medium $7
        - Large $9
    - Juices
        - Small $4
        - Medium $6
        - Large $8
        
    - Smoothie bowls
        - Small $6
        - Medium $8
        - Large $10
        
    - Acai bowls  
        - Small $7
        - Medium $9
        - Large $11
    
Size: 
    - Small: 12 oz
    - Medium: 16 oz
    - Large: 20 oz
    
Smoothies can be customized with the following extra ingredients:
    - Fruits: Strawberry, Banana, Mango, Pineapple, Blueberry
    - Nuts
    - Seeds
    
    

Extra ingredients cost $1 each.

Cash payments are accepted: 10% discount on cash payments.
If the user asks for the date and time, answer with today's date and time.
When order is successful finished, answer with the total of the order and
the order details (product, size, extra ingredients).

Messages processed by the assistant: {count}
"""
# https://localhost:5004/ouath2_calendar?code=4/0AX4XfWgq
endpoint_path = "/ouath2_calendar"
endpoint_verbs = ["GET"]


# callback_middleware = CallbackMiddleware(
#     endpoints = {
#         'calendar_event': {
#             'path': endpoint_path,
#             'verbs': endpoint_verbs
#         }
#     },
# )
    

prompt_template = PromptTemplate(prompt, initial_state={
        # Today full date and time
        "date": get_current_date,
    })

# Create the assistant based on the Macaw Assistant 
# NOTE: Make sure to provide api key in the environment variable `OPENAI_API_KEY`
# add this line to your .env file: OPENAI_API_KEY=your-key
# or uncomment the next line and replace `your-key` with your OpenAI API key
# os.environ["OPENAI_API_KEY"] = "your-key.." 
ast = MacawAssistant(prompt=prompt_template, state={})



# Configure the RAG model using the MarkdownRAG provider
# by default it uses the CachedOpenAIEmbedding for text2vec
# and ChromaStore for storing the vectors
mdm = MarkdownRAG("demo", file_path="examples/3_clerk_tooling/qa.md", split_table_rows=True)
# Load from the markdown file, then slice the content, and store it.
mdm.load()
# Register the RAG model with the assistant
ast.set_rag_retrieval(mdm)





# # in-context
# @ast.callback('calendar_event')
# async def handle_calendar_event(session, ctx: RequestContext, data):
#     token = data['state']
#     async with ctx.state_manager() as state:
#         state['calendar_token'] = token
    
#     ctx.response_text(f"Calendar token saved: {token}")
#     return ctx.cancel_ai_response()


@ast.event('message')
async def handle_message(session, ctx: RequestContext):
    async with ctx.state_manager() as state:
        count = state.get("count", 0)
        count += 1
        state["count"] = count
        log.warning(f"Message count: {count}")



# Tool - Create Order
# In order to declare a function, you need to use the @ast.function decorator
# The function name should be unique and should not contain spaces
# Description should be a short description of what the function does, 
# this is very important for the assistant to understand the function
# --------------------------------------------------------------------
@ast.function('create_order', 'Customer creates an order', params=[
    Param('product', 'string', 'Product to order', required=True),
    Param('product_size', 'string', 'Product size', required=True),
    Param('date_time', 'string', 'Date and time of the order formatted: YYYY-MM-DD HH:MM:SS', required=False),
    Param('extra_ingredients', 'string', 'Extra ingredients for personalized order', required=True)
])
async def handle_create_order(session, params, ctx: FunctionContext):    
    log.debug(f"Got create_order from client:{ctx.lead.conversation_from.name}\
                command with params: {params}")

    # TODO: Implement order creation logic here
    product = params['product']
    extra_ingredients = params['extra_ingredients']
    date_time = params.get('date_time', None)
    product_size = params['product_size']
    
    log.warning(f"Order created for product: {product} with extra ingredients: {extra_ingredients}")
    log.warning(f"Order created date:", date_time)
    
    async with ctx.state_manager() as state:
        # Save the order details to the state
        state["order_detail"] = {
            "product": product,
            "product_size": product_size,
            "extra_ingredients": extra_ingredients,
            "date_time": date_time
        }
        # Save the total to the state
        state["total"] = 100
        # Save the order status to the state
        state["order_status"] = "pending"

    #TODO: Integration with the POS system ERP
    # callback_middleware.create_callback(lead, 'calendar_event', endpoint_path, endpoint_verbs)

    response_message = (
        f"Great we are preparing your order for {product} "
        f"with extra ingredients: {extra_ingredients}. "
        f"Your order will be ready in a few minutes. Su total es: $[total]"
        f"Order details: [order_detail]"
    )
    # Response back using FunctionContext. 
    # This allows you to send a response back to genAI for processing
    return FunctionContext.response_text(
        response_message
    )
# --------------------------------------------------------------------



# Create the Message Gateway - This component is the core of the assistant
# It handles the communication between the assistant and the connectors
gateway = MessageGateway(
    assistant=ast,
    host="127.0.0.1", port=5004,
    message_enhancer=SmartMessageEnhancerOpenAI()
)

# For this example, we will use the Telegram connector
conn = TelegramConnector(
    token=os.environ.get("TELEGRAM_TOKEN"), 
    stream_mode=StreamMode.FULL
)
# Register the connector with the gateway
gateway.register_connector(conn)

# Then start the gateway and begin processing messages
gateway.run(enable_ngrok=True)

# if you want to use ngrok for testing, 
# you can enable it by setting enable_ngrok=True
# NOTE: Make sure you have ngrok installed in your system
# and env variable NGROK_AUTHTOKEN set with your ngrok token
# gateway.run(enable_ngrok=True)

