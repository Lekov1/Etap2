from langchain.schema import HumanMessage, SystemMessage, AIMessage


def split_messages_by_role(messages: list) -> tuple:
    assert isinstance(messages, list) and messages

    if len(messages) == 1:
        assert isinstance(messages[0], HumanMessage)
        return None, None, messages

    if isinstance(messages[0], SystemMessage):
        system_message = messages[0]
        conversation_messages = messages[1:]
    else:
        system_message = None
        conversation_messages = messages

    if isinstance(conversation_messages[0], AIMessage):
        assert system_message is not None
        first_ai_message = conversation_messages[0]
        conversation_messages = conversation_messages[1:]
    else:
        first_ai_message = None

    assert all(isinstance(m, HumanMessage) for m in conversation_messages[::2])
    assert all(isinstance(m, AIMessage) for m in conversation_messages[1::2])

    return system_message, first_ai_message, conversation_messages


def humanize_system_message(messages: list) -> list:
    system_message, first_ai_message, conversation_messages = split_messages_by_role(messages)
    system_message_text = system_message.content if system_message is not None else ''

    first_human_message = conversation_messages[0]
    if first_ai_message is None:
        fhm = f'{system_message_text}\n\n{first_human_message.content}'.lstrip()
        first_human_message = HumanMessage(content=fhm)
        return [first_human_message, *conversation_messages[1:]]

    messages_preprocessed = [HumanMessage(content=system_message_text), first_ai_message, *conversation_messages]
    assert all(isinstance(m, HumanMessage) for m in messages_preprocessed[::2])
    assert all(isinstance(m, AIMessage) for m in messages_preprocessed[1::2])

    return messages_preprocessed
