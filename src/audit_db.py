from typing import List
from typing import Optional
from sqlalchemy import ForeignKey,Integer,DateTime,TEXT
from datetime import datetime,timezone
from sqlalchemy import String
from sqlalchemy.orm import DeclarativeBase
from sqlalchemy.orm import Mapped
from sqlalchemy.orm import mapped_column
from sqlalchemy.orm import relationship
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
import os

DB_PATH = r"D:\mlproject19\Vector_Database\rag_audit.db"
DATABASE_URL = f"sqlite:///{DB_PATH}"

engine = create_engine(DATABASE_URL, echo=True)
sessionlocal=sessionmaker(engine)

class Base(DeclarativeBase):
    pass

class User(Base):
    '''
    The User class represents a user query and its corresponding LLM response, 
    along with the retrieved contexts (both text and image) that were used to generate the response. 
    It has a one-to-many relationship with the RetrievedContext class, which stores the individual contexts retrieved for each query. 
    The User class has the following attributes:
    - id: A unique identifier for each user query (primary key).
    - user_id: An integer representing the user who made the query.
    - query_text: A string representing the query text.
    - llm_response: A string representing the LLM response to the query.
    - timestamp: A datetime object representing the timestamp when the query was made.
    - retrieved_contexts: A list of RetrievedContext objects that represent the contexts retrieved for the query.
    '''
    __tablename__ = "User"
    id: Mapped[int] = mapped_column(Integer,primary_key=True)
    user_id: Mapped[int] = mapped_column(Integer,nullable=False)
    query_text: Mapped[str]=mapped_column(TEXT,nullable=False)
    llm_response:Mapped[str]=mapped_column(TEXT,nullable=False)
    timestamp:Mapped[datetime]=mapped_column(DateTime,default=lambda: datetime.now(timezone.utc))
    retrieved_contexts: Mapped[List["RetrievedContext"]] = relationship(
        back_populates="parent_query", cascade="all, delete-orphan"
    )

class RetrievedContext(Base):
    '''
    The RetrievedContext class represents the individual contexts (both text and image) that were retrieved for a user query. 
    It has a many-to-one relationship with the User class, which represents the user query and its corresponding LLM response. The RetrievedContext class has the following attributes:
    - id: A unique identifier for each retrieved context (primary key).
    - retrived_id: An integer representing the user query to which this context belongs (foreign key referencing User.id).
    - chunk_type: A string representing the type of context (either "text" or "image").
    - content: A string representing the content of the retrieved context (either the text chunk or the image caption).
    - source_path: A string representing the source path of the retrieved context (either the page number for text chunks or the file path for image captions).
    - parent_query: A relationship to the User class that represents the user query to which this context belongs.
    '''
    __tablename__ = "retrieved_contexts"
    id: Mapped[int] = mapped_column(Integer,primary_key=True)
    retrived_id: Mapped[int] = mapped_column(Integer,ForeignKey("User.id"),nullable=False)
    chunk_type: Mapped[str]=mapped_column(TEXT,nullable=False)
    content: Mapped[str]=mapped_column(TEXT,nullable=False)
    source_path: Mapped[str]=mapped_column(TEXT,nullable=False)
    parent_query: Mapped["User"] = relationship("User", back_populates="retrieved_contexts")

def init_db():
    os.makedirs(os.path.dirname(DB_PATH),exist_ok=True)
    Base.metadata.create_all(engine)
    print("Database created successfully")
'''
The log_rag_transaction function is responsible for logging the details of each RAG transaction into the database. 
It takes the user_id, query, response, text_context, and image_context as parameters and creates a new User entry in the database 
with the corresponding RetrievedContext entries for each retrieved context (both text and image). 
The function uses a session to interact with the database and ensures that the transaction is committed successfully or rolled back in case of any exceptions. 
Finally, it closes the session after the transaction is completed.
'''
def log_rag_transaction(user_id: int, query: str, response: str, text_context: list, image_context: list):
    session=sessionlocal()
    try:
        new_log=User(
            user_id=user_id,
            query_text=query,
            llm_response=response
        )

        if text_context:
            for doc in text_context:
                page_info = str(doc.metadata.get("page", "Unknown Page"))
                new_log.retrieved_contexts.append(
                    RetrievedContext(
                        chunk_type="text",
                        content=doc.page_content,
                        source_path=f"Page info:{page_info}"
                    )
                )
        
        if image_context:
            for doc in image_context:
                page_info = str(doc.metadata.get("filepath", "Unknown Page"))
                new_log.retrieved_contexts.append(
                    RetrievedContext(
                        chunk_type="image",
                        content=doc.page_content,
                        source_path=f"Page info:{page_info}"
                    )
                )
        
        session.add(new_log)
        session.commit()

        print("Transaction logged into database successfully")
    
    except Exception as e:
        session.rollback()
        print("Error in transaction logging",e)
    
    finally:
        session.close()
        print("Session Closed")