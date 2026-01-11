from sqlalchemy import Column, Float, Integer, String, Text
from sqlalchemy.orm import declarative_base

Base = declarative_base()


class Run(Base):
    __tablename__ = "runs"

    id = Column(Integer, primary_key=True)
    dataset_name = Column(String, nullable=False)
    model_name = Column(String, nullable=False)
    eval_type = Column(String, nullable=False)
    average_score = Column(Float, nullable=False)


class RunItem(Base):
    __tablename__ = "run_items"

    id = Column(Integer, primary_key=True)
    run_id = Column(Integer, nullable=False)
    item_id = Column(String, nullable=False)
    prediction = Column(Text, nullable=False)
    score = Column(Float, nullable=False)
