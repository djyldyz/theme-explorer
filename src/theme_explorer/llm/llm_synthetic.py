"""This module defines functionality to use LLM for augmenting data."""

import os

import dotenv
from langchain.chains import LLMChain
from langchain_anthropic import ChatAnthropic

from theme_explorer.llm.prompt_synthetic import SURVEY_PROMPT, pydantic_parser

dotenv.load_dotenv()


class SyntheticSurvey:
    """
    Wraps the logic for using an LLM to synthesise user feedback
    based on provided prompt.
    """

    def __init__(
        self,
        model_name: str = "claude-3-sonnet-20240229",
        max_tokens: int = 2048,
        temperature: float = 0.9,
    ):
        """
        Args:
            model_name: Model identifier. Supports Claude models
        """
        # Load environment variables

        if model_name.startswith("claude"):
            # Claude models
            anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
            if not anthropic_api_key:
                raise ValueError("ANTHROPIC_API_KEY not found in environment variables")

            self.llm = ChatAnthropic(
                model=model_name,
                anthropic_api_key=anthropic_api_key,
                max_tokens=max_tokens,
                temperature=temperature,
            )

        self.prompt = SURVEY_PROMPT

    def get_synthetic_response(
        self,
        concept1: str,
        concept2: str,
        definitions: list,
        examples: list,
        instructions: str,
    ) -> list:
        """
        Generates synthetic response
        Returns:
            list: List of feedback responses
        """

        # Create chain - updated syntax for newer LangChain versions
        chain = LLMChain(llm=self.llm, prompt=self.prompt)

        # Execute chain
        survey_response = chain.invoke(
            {
                "concept1": concept1,
                "concept2": concept2,
                "definitions": definitions,
                "examples": examples,
                "instructions": instructions,
            }
        )

        # Parse the output to desired format
        try:
            # Extract text from response - handle both old and new response formats
            response_text = survey_response.get("text", "") or survey_response.get(
                "content", ""
            )
            validated_answer = pydantic_parser.parse(response_text)
        except ValueError as parse_error:
            print(f"Retrying llm response parsing due to an error: {parse_error}")
            validated_answer = None

        return validated_answer
