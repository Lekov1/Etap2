from langchain.schema import AIMessage, HumanMessage

from .common import humanize_system_message


def build_mistral_prompt_template(
        messages,
        on_new_lines=False,
        begin_human_message_with_bs=False,
        end_human_message_with_es=False,
):
    messages_ = humanize_system_message(messages)

    b_inst, e_inst = '[INST]', '[/INST]'
    b_s, e_s = '<s>', '</s>'

    prompt_template = ''

    suffix = '\n' if on_new_lines else ''
    first_hm_passed = False
    for msg in messages_:
        if isinstance(msg, HumanMessage):
            prefix = b_s if (not first_hm_passed) or begin_human_message_with_bs else ''
            suffix_ = suffix if not end_human_message_with_es else f'{e_s}{suffix}'
            first_hm_passed = True
            prompt_template += f'{prefix}{b_inst} {msg.content} {e_inst}{suffix_}'
        elif isinstance(msg, AIMessage):
            prefix = '' if on_new_lines else ' '
            prompt_template += f'{prefix}{msg.content}{e_s}{suffix}'

    return prompt_template


def build_mistral_prompt_template_v1(messages):
    return build_mistral_prompt_template(
        messages,
        on_new_lines=False,
        begin_human_message_with_bs=False,
        end_human_message_with_es=False
    )


def build_mistral_prompt_template_v2(messages):
    return build_mistral_prompt_template(
        messages,
        on_new_lines=False,
        begin_human_message_with_bs=False,
        end_human_message_with_es=True
    )


def build_mistral_prompt_template_v3(messages):
    return build_mistral_prompt_template(
        messages,
        on_new_lines=False,
        begin_human_message_with_bs=True,
        end_human_message_with_es=False
    )


def build_mistral_prompt_template_v4(messages):
    return build_mistral_prompt_template(
        messages,
        on_new_lines=False,
        begin_human_message_with_bs=True,
        end_human_message_with_es=True
    )


def build_mistral_prompt_template_v5(messages):
    return build_mistral_prompt_template(
        messages,
        on_new_lines=True,
        begin_human_message_with_bs=False,
        end_human_message_with_es=False
    )


def build_mistral_prompt_template_v6(messages):
    return build_mistral_prompt_template(
        messages,
        on_new_lines=True,
        begin_human_message_with_bs=False,
        end_human_message_with_es=True
    )


def build_mistral_prompt_template_v7(messages):
    return build_mistral_prompt_template(
        messages,
        on_new_lines=True,
        begin_human_message_with_bs=True,
        end_human_message_with_es=False
    )


def build_mistral_prompt_template_v8(messages):
    return build_mistral_prompt_template(
        messages,
        on_new_lines=True,
        begin_human_message_with_bs=True,
        end_human_message_with_es=True
    )


def build_mistral_prompt_template_single_message_v1(messages):
    """<s>[INST] System Message

        Human Message [/INST]"""
    return build_mistral_prompt_template_v1(messages)


def build_mistral_prompt_template_single_message_v2(messages):
    """<s>[INST] System Message

        Human Message
        [/INST]</s>"""
    return build_mistral_prompt_template_v2(messages)


def build_mistral_prompt_template_multiple_messages_v1(messages):
    """<s>[INST] System Message

       Human Message 1 [/INST] Assistant Message 1</s>[INST] Human Message 2 [/INST] Assistant Message 2</s>[INST] Human Message 3 [/INST]"""
    return build_mistral_prompt_template_v1(messages)


def build_mistral_prompt_template_multiple_messages_v2(messages):
    """<s>[INST] System Message

       Human Message 1 [/INST]
       Assistant Message 1</s>
       [INST] Human Message 2 [/INST]
       Assistant Message 2</s>
       [INST] Human Message 3 [/INST]"""
    return build_mistral_prompt_template_v5(messages)


def build_mistral_prompt_template_multiple_messages_v3(messages):
    """<s>[INST] System Message

       Human Message 1 [/INST] Assistant Message 1</s><s>[INST] Human Message 2 [/INST] Assistant Message 2</s><s>[INST] Human Message 3 [/INST]"""
    return build_mistral_prompt_template_v3(messages)


def build_mistral_prompt_template_multiple_messages_v4(messages):
    """<s>[INST] System Message

       Human Message 1 [/INST]
       Assistant Message 1</s>
       <s>[INST] Human Message 2 [/INST]
       Assistant Message 2</s>
       <s>[INST] Human Message 3 [/INST]"""
    return build_mistral_prompt_template_v7(messages)


build_mistral_prompt_template_default = build_mistral_prompt_template_v1
