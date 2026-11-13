"""
FinDesk V4 - Finance Document and Operations Assistant
Day 2 / Session 4 - runnable checkpoint (D2P1.py)

WHAT THIS VERSION DOES
----------------------
Adds two capabilities on top of the Day 1 router:

  A. RAG over corporate finance documents
     6 policy .txt files -> chunks -> embeddings -> ChromaDB -> retriever
     -> the retrieved context is injected into the agent's system prompt.

  B. Finance tools bound to the chat model
     expense calculator, GST calculator, budget utilisation, invoice total
     and vendor lookup (see common/tools.py). The model decides when to
     call them; ToolNode executes them.

GRAPH STRUCTURE
---------------
              START
                |
                v
        Input Processing      node 1 - clean question, seed chat history
                |
                v
             Retrieval        node 2 - similarity search (NO LLM call)
                |
                v
               Agent          node 3 - LLM with bound tools + RAG context
                |
          tools_condition     decides: tool call?  -> ToolNode -> Agent (loop)
                |                        no tool call? -> Final Output
                v
             ToolNode         executes the requested Python tools
                |  (loops back to Agent)
                v
         Final Output         node 4 - formats the last AI message
                |
                v
               END

CONCEPTS INTRODUCED
-------------------
  * Document objects, text splitting, embeddings, ChromaDB, retrievers
  * @tool decorator, tool schemas/descriptions, bind_tools()
  * ToolNode + tools_condition -> the agent execution cycle (a LOOP)
  * Provider-compatible fallback: if bind_tools() is unsupported the graph
    still answers with plain RAG (no tools)

RUN
---
    Same procedure as Day 1:  cd Day2  ->  python D2P1.py
    (create the venv and install requirements.txt once - see Day2.md §1)

COMMANDS
--------
    /retrieval   run the similarity search only (no LLM) and show chunks
    /tools       list the tools the model can call
    /steps       answer a question showing each node update
    /graph       print the graph structure
    /model       show active model configuration
    /provider    switch provider (ollama | gemini)
    /exit        quit
"""

# Allows modern Python type annotations to be evaluated lazily.
from __future__ import annotations

# sys is used to print configuration errors and exit with a status code.
import sys

# TypedDict defines the schema of the data shared between graph nodes.
from typing import TypedDict

# Annotated attaches a REDUCER to a state field; add_messages tells
# LangGraph to APPEND chat messages instead of overwriting them.
from typing import Annotated

# StrOutputParser converts an AIMessage (content may be str or list) into
# one plain string - used by the final output node.
from langchain_core.output_parsers import StrOutputParser

# SystemMessage / HumanMessage are the typed chat messages LangChain uses.
from langchain_core.messages import HumanMessage, SystemMessage

# add_messages is the standard reducer for chat histories.
from langgraph.graph.message import add_messages

# START / END mark entry and exit; StateGraph builds the workflow.
# ToolNode and tools_condition are LangGraph's prebuilt agent components.
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

# Shared modules: configuration, model factory, prompts, RAG helpers and
# the finance tools defined in common/tools.py.
from common import config, model_factory, prompts, rag
from common.tools import FINANCE_TOOLS


# ==========================================================================
# 1. STATE - data shared by every node in the V4 workflow
# ==========================================================================
class FinanceState(TypedDict):
    """State schema for the RAG + tool-calling agent.

    Fields
    ------
    question         : raw text typed by the employee
    cleaned_question : normalised text produced by input_processing
    context          : retrieved policy chunks (string) set by retrieve
    messages         : chat history. REDUCER = add_messages, so every node
                       that returns messages APPENDS to the list instead of
                       replacing it (AIMessage, ToolMessage, ... accumulate)
    final_output     : formatted reply written by final_output
    """

    question: str
    cleaned_question: str
    context: str
    messages: Annotated[list, add_messages]
    final_output: str


# Set by build_app(): True when the model accepted the tool catalogue.
# Read by the /tools command so students know whether tool calling works
# with their current provider/model combination.
TOOLS_BOUND: bool = True


# ==========================================================================
# 2. GRAPH FACTORY
# ==========================================================================
def build_app(provider: str | None = None, retriever=None):
    """Compile and return the FinDesk V4 application.

    Parameters
    ----------
    provider : str, optional
        "ollama" / "gemini"; None = value from .env.
    retriever : optional
        Pre-built retriever from rag.build_retriever(). Passing it in
        lets the CLI build the vector index ONCE at startup instead of
        re-embedding six policies on every provider switch or /retrieval
        command (embedding calls cost time and, on Gemini, quota).

    Returns
    -------
    Compiled graph (StateGraph.compile()) - supports .invoke()/.stream().
    """
    global TOOLS_BOUND
    llm = model_factory.get_llm(provider)

    # ------------------------------------------------------------------
    # BIND TOOLS TO THE MODEL
    # bind_tools() appends the tool catalogue (name, description, JSON
    # schema of arguments) to the model request. The model then answers
    # either with plain text or with structured "tool_calls".
    #
    # FALLBACK: not every local model supports tool calling. If bind_tools
    # raises, we keep a tool-free LLM - the graph still works, the model
    # simply cannot perform calculations (RAG answers continue to work).
    # ------------------------------------------------------------------
    TOOLS_BOUND = True
    try:
        agent_llm = llm.bind_tools(FINANCE_TOOLS)
    except Exception as exc:
        print(f"  [tool binding unavailable: {type(exc).__name__}; "
              f"running without tools]")
        agent_llm = llm
        TOOLS_BOUND = False

    # Prebuilt node that executes whichever tools the model requested.
    # ToolNode reads the tool_calls of the last AIMessage, runs the
    # matching Python functions and appends ToolMessage results.
    tool_node = ToolNode(FINANCE_TOOLS)

    # The retriever (policies -> chunks -> embeddings -> Chroma) is built
    # by the caller and passed in - see build_app() docstring. Building it
    # eagerly means a configuration problem surfaces at startup, not on
    # the first question.
    if retriever is None:
        try:
            retriever = rag.build_retriever(provider)
        except Exception as exc:
            raise RuntimeError(f"RAG setup failed: {exc}") from exc

    # ------------------------------------------------------------------
    # NODE 1: input processing
    # Cleans the question AND seeds the chat history with the employee's
    # message. Returning {"messages": [HumanMessage(...)]} APPENDS to the
    # (initially empty) message list thanks to the add_messages reducer.
    # ------------------------------------------------------------------
    def input_processing(state: FinanceState) -> dict:
        # Collapse repeated whitespace so the retriever gets a clean query.
        question = " ".join(state["question"].split())
        if not question:
            question = "(empty question)"
        return {
            "cleaned_question": question,
            # HumanMessage is the typed form of a user turn; the agent node
            # relies on it being present in state["messages"].
            "messages": [HumanMessage(content=question)],
        }

    # ------------------------------------------------------------------
    # NODE 2: retrieval (RAG step - NO LLM involved)
    # Similarity search compares the question's vector with every policy
    # chunk vector and returns the k most similar chunks. format_context()
    # turns them into one labelled string (or NO_CONTEXT if nothing
    # relevant was found).
    # ------------------------------------------------------------------
    def retrieve(state: FinanceState) -> dict:
        chunks = retriever.invoke(state["cleaned_question"])
        context = rag.format_context(chunks)
        return {"context": context}

    # ------------------------------------------------------------------
    # NODE 3: agent
    # Builds the model request as:
    #     [SystemMessage: rules + retrieved context]
    #   + state["messages"]        <- human turn, AI turns, tool results
    # then invokes the model. The reply (AIMessage) is APPENDED to
    # state["messages"], so on the next loop iteration the model sees the
    # ToolMessage results of any tools it just called.
    #
    # The system message is rebuilt every time (not stored in state) so
    # the context appears exactly once in the request.
    # ------------------------------------------------------------------
    def agent(state: FinanceState) -> dict:
        system_text = prompts.RAG_CONTEXT_PROMPT.format(
            context=state.get("context", prompts.NO_CONTEXT)
        )
        request = [SystemMessage(content=system_text)] + state["messages"]
        response = agent_llm.invoke(request)
        # Returning the AIMessage (with or without tool_calls) appends it.
        return {"messages": [response]}

    # ------------------------------------------------------------------
    # NODE 4: final output (presentation only - no model call)
    # Reached only when the agent answered WITHOUT tool calls.
    # StrOutputParser safely converts the AIMessage content (string or
    # list-of-parts, which some providers return) into one string.
    # ------------------------------------------------------------------
    def final_output(state: FinanceState) -> dict:
        last = state["messages"][-1]
        try:
            text = StrOutputParser().invoke(last)
        except Exception:
            # Defensive fallback: stringify content by hand.
            content = getattr(last, "content", str(last))
            text = "".join(
                part.get("text", "") if isinstance(part, dict) else str(part)
                for part in content
            ) if isinstance(content, list) else str(content)
        return {"final_output": f"FinDesk:\n{text.strip()}"}

    # ------------------------------------------------------------------
    # WIRE THE GRAPH
    # ------------------------------------------------------------------
    builder = StateGraph(FinanceState)

    builder.add_node("input_processing", input_processing)
    builder.add_node("retrieve", retrieve)
    builder.add_node("agent", agent)
    builder.add_node("tools", tool_node)          # prebuilt ToolNode
    builder.add_node("final_output", final_output)

    # Linear head: START -> clean -> retrieve -> agent.
    builder.add_edge(START, "input_processing")
    builder.add_edge("input_processing", "retrieve")
    builder.add_edge("retrieve", "agent")

    # THE AGENT LOOP - a conditional edge decides what happens after the
    # agent node runs:
    #   tools_condition returns "tools"   if the last AIMessage has
    #                                     tool_calls  -> run ToolNode
    #   tools_condition returns "__end__" otherwise    -> we redirect that
    #                                     path to final_output instead of
    #                                     ending the graph immediately.
    builder.add_conditional_edges(
        "agent",                     # decision is made after this node
        tools_condition,             # built-in: "tools" | "__end__"
        {
            "tools": "tools",        # tool call requested -> execute tools
            "__end__": "final_output",  # plain answer -> format and finish
        },
    )

    # After tools finish, ALWAYS return to the agent so the model can see
    # the tool results and either call more tools or answer the employee.
    builder.add_edge("tools", "agent")

    builder.add_edge("final_output", END)

    # compile() validates the graph (all nodes reachable, no dead ends)
    # and returns the runnable app. Whether tools were bound is exposed
    # through the module-level TOOLS_BOUND flag for the /tools command.
    return builder.compile()


# ==========================================================================
# 3. HELPERS
# ==========================================================================
def print_graph(app) -> None:
    """Print the graph structure (ASCII art when available)."""
    graph = app.get_graph()
    try:
        print(graph.draw_ascii())
    except Exception:
        # Fallback: node/edge lists - always works without grandalf.
        print("Nodes :", [n for n in graph.nodes])
        print("Edges :", [(e.source, e.target) for e in graph.edges])


def run_with_trace(app, question: str) -> None:
    """Invoke the graph with stream_mode="updates" and print every step.

    Each yielded dict is {"node_name": {keys that node changed}}. Merging
    them into a local dict mirrors what LangGraph does internally, so the
    final reply is produced without a second model call. The agent/tools
    loop is visible as repeated 'agent' and 'tools' updates.
    """
    print(f"  question: {question}")
    state: dict = {"question": question}
    for step in app.stream(state, stream_mode="updates"):
        for node_name, update in step.items():
            # Messages are printed as their type + short text to keep the
            # trace readable (full message objects are noisy).
            shown = update
            if "messages" in update:
                shown = {
                    k: [
                        f"{type(m).__name__}: "
                        f"{str(getattr(m, 'content', ''))[:80]}"
                        for m in v
                    ] if k == "messages" else v
                    for k, v in update.items()
                }
            print(f"  -> node '{node_name}' updated: {shown}")
            state.update(update)
    print(f"  {state.get('final_output', '(no output)')}\n")


def show_retrieval(retriever, question: str) -> None:
    """Run ONLY the RAG retrieval step and print the raw chunks.

    Students see exactly what the model will receive as context - no LLM
    call, no tokens spent. This is the fastest way to debug a bad answer:
    if the right chunk is missing here, the problem is the index/query,
    not the model.

    Parameters
    ----------
    retriever : the retriever built once at startup (passed in so the
        policies are not re-embedded for every /retrieval command).
    question : str - the employee's question used as the search query.
    """
    chunks = retriever.invoke(question)
    if not chunks:
        print("  (no chunks matched)\n")
        return
    print(f"  {len(chunks)} chunk(s) retrieved:\n")
    for i, chunk in enumerate(chunks, start=1):
        source = chunk.metadata.get("source", "?")
        # First 220 characters are enough to recognise the section.
        preview = chunk.page_content[:220].replace("\n", " ")
        print(f"  [{i}] {source}: {preview}...")
    print()


def choose_provider() -> str | None:
    """Model menu - same UX as Day 1 sessions."""
    print("=" * 60)
    print("FinDesk V4 - Finance Document and Operations Assistant")
    print("=" * 60)
    print(f"Active configuration: {config.describe()}")
    print(f"Embeddings           : {config.describe_embeddings()}")
    print("\nSelect model:")
    print("  1. Gemini   (cloud, requires GOOGLE_API_KEY in .env)")
    print("  2. Ollama   (local, default)")
    print(f"  3. Use .env default (currently: {config.LLM_PROVIDER})")
    choice = input("Choice [3]: ").strip()
    if choice == "1":
        return "gemini"
    if choice == "2":
        return "ollama"
    return None


BANNER = """
------------------------------------------------------------
FinDesk V4 - Finance Document and Operations Assistant
Type a finance question, or one of:
  /retrieval   - show the chunks retrieved for a question (no LLM)
  /tools       - list the tools the model can call
  /steps       - answer a question showing each node update
  /graph       - print the graph structure
  /model       - show active model configuration
  /provider    - switch provider (ollama | gemini)
  /exit        - quit
------------------------------------------------------------"""


# ==========================================================================
# 4. INTERACTIVE CLI
# ==========================================================================
def main() -> None:
    """Entry point: model menu + chat loop (same pattern as Day 1)."""
    try:
        provider = choose_provider()
        # Build the vector index ONCE here and pass it to build_app() -
        # six policies are embedded a single time per process.
        retriever = rag.build_retriever(provider)
        app = build_app(provider, retriever)
    except (RuntimeError, ValueError, ImportError) as exc:
        # Configuration / RAG setup errors get a clean message, not a
        # raw traceback.
        print(f"\nConfiguration error:\n{exc}", file=sys.stderr)
        sys.exit(1)

    print(BANNER)
    print(f"(using {config.describe()})")
    print(f"({config.describe_embeddings()})\n")

    while True:
        try:
            question = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break
        if not question:
            continue

        cmd = question.lower()
        if cmd in ("/exit", "/quit"):
            print("Goodbye!")
            break
        if cmd == "/model":
            print(f"  {config.describe()}")
            print(f"  {config.describe_embeddings()}")
            continue
        if cmd == "/graph":
            print_graph(app)
            continue
        if cmd == "/tools":
            # TOOLS_BOUND is written by build_app() - True means the model
            # accepted the tool catalogue (bind_tools succeeded).
            state = "ACTIVE" if TOOLS_BOUND else \
                "NOT available (bind_tools failed for this model)"
            print(f"  Tool calling: {state}")
            for t in FINANCE_TOOLS:
                # name + first line of the docstring = what the model sees.
                description = (t.description or "").split("\n")[0]
                print(f"  - {t.name}: {description}")
            print()
            continue
        if cmd == "/provider":
            try:
                provider = choose_provider()
                # Rebuild BOTH the retriever (new embedding model) and the
                # graph (new chat model) - they must share the provider.
                retriever = rag.build_retriever(provider)
                app = build_app(provider, retriever)
                print(f"  Switched -> {config.describe()}")
                print(f"  ({config.describe_embeddings()})")
            except (RuntimeError, ValueError, ImportError) as exc:
                print(f"  Switch failed: {exc}")
            continue
        if cmd == "/retrieval":
            raw = input("Question to retrieve for: ").strip()
            if raw:
                try:
                    show_retrieval(retriever, raw)
                except Exception as exc:
                    print(f"  [error] {type(exc).__name__}: {exc}\n")
            continue
        if cmd == "/steps":
            raw = input("Question to trace: ").strip()
            if raw:
                try:
                    run_with_trace(app, raw)
                except Exception as exc:
                    print(f"  [error] {type(exc).__name__}: {exc}\n")
                    print(config.connection_hint() + "\n")
            continue

        # ---- normal question: run the whole graph once -------------------
        try:
            # recursion_limit guards against a model that requests tools
            # forever (default 25 supersteps is plenty for this graph).
            final_state = app.invoke(
                {"question": question},
                config={"recursion_limit": 40},
            )
        except Exception as exc:
            print(f"FinDesk: [error] {type(exc).__name__}: {exc}\n")
            print(config.connection_hint() + "\n")
            continue

        print(f"{final_state['final_output']}\n")


if __name__ == "__main__":
    main()
