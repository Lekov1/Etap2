from langchain.schema import AIMessage, HumanMessage

from .common import humanize_system_message


def build_claude_prompt_template(
        messages,
        begin_first_message_with_new_lines=False,
):
    messages_ = humanize_system_message(messages)

    human_prefix = "Human:"
    ai_prefix = "Assistant:"

    prompt_template = '\n\n' if begin_first_message_with_new_lines else ''

    for msg in messages_:
        if isinstance(msg, HumanMessage):
            prompt_template += f'{human_prefix}  {msg.content}\n\n'
        elif isinstance(msg, AIMessage):
            prompt_template += f'{ai_prefix} {msg.content}\n\n'

    prompt_template += ai_prefix

    return prompt_template


def build_claude_prompt_template_v1(messages):
    return build_claude_prompt_template(
        messages,
        begin_first_message_with_new_lines=True,
    )


def build_claude_prompt_template_v2(messages):
    return build_claude_prompt_template(
        messages,
        begin_first_message_with_new_lines=False,
    )


build_claude_prompt_template_default = build_claude_prompt_template_v1
