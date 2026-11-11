# D1P1.py
# me\Day1\D1P1.py
"""
FinDesk V1 - Basic Finance Assistant using LangChain
Day 1 / Session 1 - runnable checkpoint (D1P1.py)

WHAT THIS VERSION DOES
----------------------
1. Loads provider settings from Day1/.env via common/config.py
   (no model names or API keys are hardcoded in this file).
2. Lets the student pick the provider at startup: Gemini or Ollama,
   with the .env value as the default.
3. Builds a reusable finance prompt template (ChatPromptTemplate).
4. Connects  prompt -> chat model -> output parser  with LCEL (the `|`
   pipe operator) to form a Runnable chain.
5. Offers an optional `/structured` mode that returns a validated Pydantic
   object (FinanceAnswer) instead of free text.
6. Runs an interactive CLI chat loop.

RUN
---
    python D1P1.py

COMMANDS INSIDE THE CHAT LOOP
------------------------------
    /structured   answer using a Pydantic-structured response
    /model        show the active provider/model configuration
    /provider     switch provider (ollama / gemini) at runtime
    /exit         quit the application
"""

# Allows the use of modern Python type hints and annotations.
from __future__ import annotations

# Provides access to Python's system-related functionality.
# Here, it is used to display errors and terminate the program.
import sys

# StrOutputParser converts the model's response into a plain Python string.
from langchain_core.output_parsers import StrOutputParser

# ChatPromptTemplate helps us create reusable prompts containing
# system instructions and user messages.
from langchain_core.prompts import ChatPromptTemplate

# Import our shared modules.
# config       - contains application and provider configurations.
# model_factory - creates the required chat model (Gemini or Ollama).
# prompts      - contains reusable prompt templates and instructions.
from common import config, model_factory, prompts

# Import the Pydantic model that defines the expected structure
# of a finance-related answer.
from common.schemas import FinanceAnswer


# ==========================================================================
# BUILDING THE CHAIN
# ==========================================================================

# Define a function that creates and returns the finance prompt.
def build_prompt() -> ChatPromptTemplate:

    # Create a prompt using the predefined general finance instructions.
    # The prompt contains system instructions and a placeholder for the question.
    return prompts.finance_prompt(prompts.GENERAL_FINANCE_SYSTEM)


# Define a function that creates the complete LangChain processing pipeline.
# The provider parameter allows us to choose Gemini or Ollama.
# If no provider is supplied, the application uses the .env configuration.
def build_chain(provider: str | None = None):

    # Create the chat model using the selected provider.
    # The model_factory decides whether to create a Gemini or Ollama model.
    llm = model_factory.get_llm(provider)

    # Create the reusable finance prompt template.
    prompt = build_prompt()

    # Connect the three components using the LCEL pipe operator.
    # Step 1: prompt formats the system instructions and user question.
    # Step 2: llm sends the formatted messages to the selected AI model.
    # Step 3: StrOutputParser extracts the response as a plain string.
    #
    # The resulting chain can be executed using chain.invoke().
    return prompt | llm | StrOutputParser()


# Define a function that returns the model's answer as a Pydantic object.
# The question is supplied by the user.
def answer_structured(provider: str | None, question: str) -> FinanceAnswer:

    # Start a try block because structured output may not be supported
    # by every model or provider.
    try:

        # Create the chat model using the selected provider.
        llm = model_factory.get_llm(provider)

        # Tell LangChain that the model should return its response
        # in the structure defined by the FinanceAnswer Pydantic schema.
        structured_llm = llm.with_structured_output(FinanceAnswer)

        # Build the finance prompt containing the system instructions.
        prompt = build_prompt()

        # Format the prompt by inserting the user's question.
        # format_messages() produces the messages required by the chat model.
        #
        # invoke() sends those messages to the model and returns
        # a validated FinanceAnswer object.
        return structured_llm.invoke(prompt.format_messages(question=question))

    # Handle errors such as unsupported structured output or
    # provider/model compatibility problems.
    except Exception as exc:

        # Display the type of error that occurred.
        # type(exc).__name__ gives the exception class name.
        print(f"  [structured output unavailable: {type(exc).__name__}; "
              f"falling back to plain text]")

        # If structured output fails, execute the normal text-based chain.
        plain = build_chain(provider).invoke({"question": question})

        # Convert the plain-text response into a FinanceAnswer object.
        # category is set to general because the original response
        # was not classified using the structured output.
        # answer contains the actual model response.
        # key_points is an empty list because no separate key points
        # were generated in this fallback process.
        return FinanceAnswer(category="general", answer=plain, key_points=[])


# ==========================================================================
# PROVIDER SELECTION (menu shown at startup)
# ==========================================================================

# Define a function that displays the provider selection menu.
def choose_provider() -> str | None:

    # Print a separator to make the menu visually clear.
    print("=" * 60)

    # Display the application name.
    print("FinDesk - Finance Operations Assistant")

    # Print another separator.
    print("=" * 60)

    # Display the current provider configuration.
    # describe() is a helper function from the config module.
    print(f"Active configuration: {config.describe()}")

    # Print an empty line for better readability.
    print()

    # Display the available model provider options.
    print("Select model:")

    # Option 1: Use Gemini, which runs through Google's cloud service.
    # A valid Google API key must be configured in the .env file.
    print("  1. Gemini   (cloud, requires GOOGLE_API_KEY in .env)")

    # Option 2: Use Ollama, which runs models locally on the computer.
    print("  2. Ollama   (local, default)")

    # Option 3: Use the provider configured in the .env file.
    print("  3. Use .env default (currently: "
          f"{config.LLM_PROVIDER})")

    # Read the user's choice from the keyboard.
    # strip() removes unnecessary spaces before and after the input.
    choice = input("Choice [3]: ").strip()

    # If the user enters 1, select Gemini.
    if choice == "1":
        return "gemini"

    # If the user enters 2, select Ollama.
    if choice == "2":
        return "ollama"

    # For option 3 or an empty input, return None.
    # None tells the model factory to use the provider from .env.
    return None


# ==========================================================================
# INTERACTIVE CLI
# ==========================================================================

# Store the welcome message and available commands in a multiline string.
# This message is displayed when the application starts.
BANNER = """
------------------------------------------------------------
FinDesk - Finance Operations Assistant
Type a finance question, or one of:
  /structured  - get a structured (Pydantic) answer
  /model       - show active model configuration
  /provider    - switch provider (ollama | gemini)
  /exit        - quit
------------------------------------------------------------"""


# Main function: controls the complete application execution.
def main() -> None:

    # Handle configuration and initialization errors.
    try:

        # Display the provider selection menu and get the user's choice.
        provider = choose_provider()

        # Build the LangChain chain using the selected provider.
        # This chain is created once and reused for normal questions.
        chain = build_chain(provider)

    # Catch common errors that may occur during application initialization.
    except (RuntimeError, ValueError, ImportError) as exc:

        # Display the configuration error on the standard error stream.
        # sys.stderr is generally used for error messages.
        print(f"\nConfiguration error:\n{exc}", file=sys.stderr)

        # Terminate the application with exit status 1.
        # A non-zero exit status indicates that the program failed.
        sys.exit(1)

    # Display the welcome message and available commands.
    print(BANNER)

    # Display the currently configured provider/model information.
    print(f"(using {config.describe()})\n")

    # Start an infinite loop to continuously accept user questions.
    # The loop ends only when the user exits or an interruption occurs.
    while True:

        # Handle keyboard interruptions and end-of-input situations.
        try:

            # Read the user's question from the command line.
            # strip() removes leading and trailing spaces.
            question = input("You: ").strip()

        # EOFError occurs when there is no more input.
        # KeyboardInterrupt occurs when the user presses Ctrl+C.
        except (EOFError, KeyboardInterrupt):

            # Display a goodbye message.
            print("\nGoodbye!")

            # Exit the interactive loop.
            break

        # If the user enters an empty question, skip this iteration.
        if not question:
            continue

        # ---- special commands -------------------------------------------

        # Check whether the user wants to exit the application.
        # lower() makes the command comparison case-insensitive.
        if question.lower() in ("/exit", "/quit"):

            # Display a goodbye message.
            print("Goodbye!")

            # Stop the application.
            break

        # Check whether the user wants to see the active model configuration.
        if question.lower() == "/model":

            # Display the provider and model configuration.
            print(f"  {config.describe()}")

            # Skip the remaining operations and wait for the next input.
            continue

        # Check whether the user wants to change the model provider.
        if question.lower() == "/provider":

            # Handle errors that may occur while switching providers.
            try:

                # Display the provider selection menu again.
                # The user can select Gemini, Ollama or the .env default.
                provider = choose_provider()

                # Rebuild the chain using the newly selected provider.
                # The previous chain is replaced with this new chain.
                chain = build_chain(provider)

                # Display a message indicating that the switch was completed.
                print(f"  Switched -> {config.describe()}")

            # Handle common provider switching errors.
            except (RuntimeError, ValueError, ImportError) as exc:

                # Display the reason why the provider switch failed.
                print(f"  Switch failed: {exc}")

            # Return to the beginning of the loop and wait for another input.
            continue

        # Check whether the user wants a structured Pydantic response.
        if question.lower() == "/structured":

            # Ask the user to enter the actual finance question.
            raw = input("Structured question: ").strip()

            # If no question is entered, return to the main chat loop.
            if not raw:
                continue

            # Generate the answer using the structured output function.
            # The function attempts to return a FinanceAnswer object.
            result = answer_structured(provider, raw)

            # model_dump() converts the Pydantic object into a Python dictionary.
            # This makes it easier to display, process or store the result.
            print(f"FinDesk (structured): {result.model_dump()}\n")

            # Return to the beginning of the chat loop.
            continue

        # ---- normal question: run the LCEL chain ------------------------

        # Execute the normal text-based LangChain pipeline.
        try:

            # Send the user's question into the chain.
            #
            # {"question": question} supplies the value for the
            # {question} placeholder in the prompt template.
            #
            # invoke() executes the complete pipeline:
            # Prompt -> Chat Model -> Output Parser.
            #
            # The final result is a plain Python string.
            answer = chain.invoke({"question": question})

        # Catch errors that may occur while communicating with the model.
        # Examples: network problems, Ollama not running or model errors.
        except Exception as exc:

            # Display the error type and its corresponding message.
            print(f"FinDesk: [error] {type(exc).__name__}: {exc}\n")

            # Display a helpful connection/configuration troubleshooting hint.
            print(config.connection_hint() + "\n")

            # Skip the remaining statements and accept another question.
            continue

        # Display the AI model's response to the user.
        print(f"FinDesk: {answer}\n")


# This condition checks whether this Python file is being executed directly.
# If the file is imported into another Python file, main() will not run.
if __name__ == "__main__":

    # Start the FinDesk application.
    main()