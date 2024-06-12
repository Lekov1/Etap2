from langchain.schema import AIMessage, HumanMessage

from .common import humanize_system_message


def build_wizard_prompt_template(
        messages,
):
    return humanize_system_message(messages)[0].content


build_wizard_prompt_template_default = build_wizard_prompt_template
