"""Module for generating prompt templates for synthetic user feedback"""

from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate

from theme_explorer.llm.synthetic_response import SyntheticResponse

_survey_template = """Your task is to generate realistic synthetic user feedback in JSON format, containing information
on: {concept1} and {concept2}. This feedback will be used to test a theme explorer application.

Here are definitions of these concepts:
{definitions}

Here are some examples you can follow when generating new responses:
{examples}

Tips:
Cover a range of potential user experience categories, such as navigation, accessibility, speed and clarity.
Structure your response as a JSON object with a "responses" array containing the feedback items.
Each feedback item should have "feedback" and "sentiment" properties.
Property names should be enclosed in double quotes.

{instructions}

{format_instructions}
"""

# Create the parser
pydantic_parser = PydanticOutputParser(pydantic_object=SyntheticResponse)

# Create the prompt template with format instructions
SURVEY_PROMPT = PromptTemplate(
    template=_survey_template,
    input_variables=["concept1", "concept2", "definitions", "examples", "instructions"],
    partial_variables={
        "format_instructions": pydantic_parser.get_format_instructions(),
    },
)

# Export the parser
parser = pydantic_parser
