from langchain.schema import AIMessage, HumanMessage, SystemMessage

from .common import split_messages_by_role


def build_llama_prompt_template(
        messages,
        with_bs_es=True,
        with_new_lines=True,
):
    b_s, e_s = '<s>', '</s>'
    b_inst, e_inst = '[INST]', '[/INST]'
    b_sys, e_sys = '<<SYS>>', '<</SYS>>'

    b_s_pref = b_s if with_bs_es else ''
    e_s_suff = e_s if with_bs_es else ''
    n_l = '\n' if with_new_lines else ''
    s_c = '' if with_new_lines else ' '

    system_message, first_ai_message, conversation_messages = split_messages_by_role(messages)

    assert first_ai_message is None

    system_message_part = f"{b_sys}{n_l}{s_c}{system_message.content}{s_c}{n_l}{e_sys}"
    beginning = f"{b_s_pref}{b_inst} {system_message_part}{s_c}"

    messages_part = ''
    for m in conversation_messages:
        if isinstance(m, HumanMessage):
            messages_part += f"{m.content} {e_inst}"
        elif isinstance(m, AIMessage):
            messages_part += f" {m.content} {e_s_suff}{b_s_pref}{b_inst} "
        else:
            raise ValueError

    prompt_template = f"{beginning}{n_l}{n_l}{messages_part}"

    return prompt_template


def build_llama_prompt_template_v1(messages):
    return build_llama_prompt_template(
        messages,
        with_bs_es=True,
        with_new_lines=True
    )


def build_llama_prompt_template_v2(messages):
    return build_llama_prompt_template(
        messages,
        with_bs_es=True,
        with_new_lines=False
    )


def build_llama_prompt_template_v3(messages):
    return build_llama_prompt_template(
        messages,
        with_bs_es=False,
        with_new_lines=True
    )


def build_llama_prompt_template_v4(messages):
    return build_llama_prompt_template(
        messages,
        with_bs_es=False,
        with_new_lines=False
    )


build_llama_prompt_template_default = build_llama_prompt_template_v1
