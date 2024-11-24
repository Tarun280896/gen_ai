from langchain_community.utilities import SQLDatabase
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
# from langchain_openai import ChatOpenAI
from langchain_community.llms import Ollama
from langchain_core.prompts import ChatPromptTemplate
import re
# if you are using SQLite
sqlite_uri = 'sqlite:///./output.db' 

# if you are using MySQL
# mysql_uri = 'mysql+mysqlconnector://root:admin@localhost:3306/test_db'
chat_history = []
db = SQLDatabase.from_uri(sqlite_uri)
llm = Ollama(model = 'mistral:latest')



def get_schema(_):
    schema = db.get_table_info()
    return schema

def run_query(query):
    regex = r"```sql\n(.*?)\n```"
    # Extract the query
    match = re.search(regex, query, re.DOTALL)  # re.DOTALL allows matching across newlines
    if match:
        sql_query = match.group(1)
        print('sql_query:', sql_query)
        return db.run(f'{sql_query}')




llm = Ollama(model="mistral")

chat_history = []

template = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """you are sqllite expert an Based on the table schema below, write a sqllite query that would answer the user's question:
             {schema}""",
        ),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
        ("system", "SQL Query:")
    ]
)

sql_chain = (
    RunnablePassthrough.assign(schema=get_schema)
    | template
    | llm.bind(stop=["\nSQL Result:"])
    | StrOutputParser()
)

response_prompt_template = ChatPromptTemplate.from_messages(
     [
        ( "system",
          """Based on the table schema below, question, SQLite query, and SQL response, write a natural language response: {schema},
          while generating sql response for the user: dont use the schema sample rows just use the sql query""", 
        ),
          MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
            ("system", "SQL Query: {query}\nSQL Response: {response}\nNatural Language Response:") 
    ]
)

full_chain = (
    RunnablePassthrough.assign(query=sql_chain).assign(
        schema=get_schema,
        response=lambda vars: run_query(vars["query"])
    )
    | response_prompt_template
    | llm#.bind(stop=["\nSQLResult:"])
    | StrOutputParser()
)


def start_app():
    while True:
        question = input("You: ")
        if question == "done":
            return

        # response = llm.invoke(question)
        response = full_chain.invoke({"input": question, "chat_history": chat_history})
        chat_history.append(HumanMessage(content=question))
        chat_history.append(AIMessage(content=response))

        print("AI:" + response)


if __name__ == "__main__":
    start_app()