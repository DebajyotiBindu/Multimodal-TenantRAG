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
        print("Error is transaction logging",e)
    
    finally:
        session.close()
        print("Session Closed")