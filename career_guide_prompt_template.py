# from langchain_core.prompts import ChatPromptTemplate
# from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, ToolMessage

# prompt = ChatPromptTemplate.from_messages(
#     [
#         ("system", """You are expert guide who guides the user with professional paths for careers
#         do not answer apart from the topic.
#         if you don't know the answer then simply answer "I don't know"."""),
#         ("human", "You are helpful assistance, Tell me about {topic} in a {selected} manner."),
#         ("ai", "{answer}")
#     ]
# )

# result = prompt.format(
#     topic = "Python",
#     selected = "concise",
#     answer = "Python is made by RaheelKhan while studing in 10th class, which is widely used by everyone..."
# )
# print(result)


from langchain_core.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    AIMessagePromptTemplate,
)

prompt = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template(
        """You are expert guide who guides the user with professional paths for careers
            do not answer apart from the topic.
            if you don't know the answer then simply answer "I don't know"."""
    ),
    HumanMessagePromptTemplate.from_template(
        "You are helpful assistance, Tell me about {topic} in a {selected} manner."
    ),
    AIMessagePromptTemplate.from_template("{answer}")
])

formatted = prompt.format(
    topic="Python",
    selected="concise",
    answer="Python is a popular programming language created by Guido van Rossum."
)

print(formatted)





# from langchain_core.prompts import ChatPromptTemplate
# from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, ToolMessage

# prompt = ChatPromptTemplate.from_messages(
#     [
#         SystemMessage(
#             content="""You are expert guide who guides the user with professional paths for careers
#             do not answer apart from the topic.
#             if you don't know the answer then simply answer "I don't know"."""
#         ),
#         HumanMessage(
#             content="You are helpful assistance, Tell me about {topic} in a {selected} manner."
#         ),
#         AIMessage(
#             content="{answer}"
#         )
#     ]
# )

# result = prompt.format_messages(
#     topic = "Python",
#     selected = "concise",
#     answer = "Python is made by RaheelKhan while studing in 10th class, which is widely used by everyone..."
# )
# print(result)
