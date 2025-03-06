from flask import Flask, request, jsonify
from flask_cors import CORS
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
from langchain.tools import DuckDuckGoSearchRun
from langmem import LangMem
from langchain_core.agents import AgentAction

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

# Initialize LangMem
memory = LangMem()

# Initialize OpenAI model
 # Initialize LLM and embeddings model
from langchain_ollama import OllamaEmbeddings
from langchain_openai import ChatOpenAI
# --- Initialize LLM ---
model = ChatOpenAI(
    base_url="http://localhost:15205/v1",
    model_name="gemini-1.5-flash",
    temperature=0.5,
    streaming=true,
    api_key="324"
)

embeddings = OllamaEmbeddings(model="nomic-embed-text")

# Define tools
tools = [DuckDuckGoSearchRun()]

# Define agent node
def agent_node(state):
    messages = state['messages']
    agent = create_react_agent(llm=model, tools=tools)
    response = agent.run(messages)
    return {"messages": messages + [AIMessage(content=str(response))]}

# Define memory node
def memory_node(state):
    messages = state['messages']
    memory.add_messages(messages)
    memory_response = memory.get_memory(messages)
    if memory_response:
        return {"messages": messages + [AIMessage(content=f"Memory: {memory_response}")]}
    else:
        return {"messages": messages}

# Define tool node
def tool_node(state):
    messages = state['messages']
    tool_invocation = state['tool_invocation']
    tool_name = tool_invocation.tool_name
    tool_input = tool_invocation.tool_input
    if tool_name == "duckduckgo_search":
        tool_output = DuckDuckGoSearchRun().run(tool_input)
        return {"messages": messages + [AIMessage(content=f"Tool Output: {tool_output}")]}
    else:
        return {"messages": messages}

# Define graph state
class GraphState:
    messages: list
    tool_invocation: AgentAction = None

# Build graph
graph = StateGraph(GraphState)
graph.add_node("memory", memory_node)
graph.add_node("agent", agent_node)
graph.add_node("tool", tool_node)

graph.set_entry_point("memory")
graph.add_edge("memory", "agent")
graph.add_conditional_edges(
    "agent",
    lambda state: "tool" if isinstance(state['messages'][-1], AgentAction) else END,
    {
        "tool": "tool",
        END: END
    }
)
graph.add_edge("tool", "memory")

# Compile graph
chain = graph.compile()

@app.route('/chat', methods=['POST'])
def chat():
    user_message = request.json['message']
    messages = [HumanMessage(content=user_message)]
    result = chain.run({"messages": messages})
    return jsonify({"response": result['messages'][-1].content})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=54275, debug=True)