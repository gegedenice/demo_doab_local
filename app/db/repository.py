from app.db.models import Base, Run, RunItem
from app.db.session import SessionLocal, engine

def init_db() -> None:
    Base.metadata.create_all(bind=engine)


def save_run(request, results: list[dict]) -> int:
    init_db()
    with SessionLocal() as session:
        average_score = 0.0
        if results:
            average_score = sum(result["score"] for result in results) / len(results)
        run = Run(
            dataset_name=request.dataset_name,
            model_name=request.model_name,
            eval_type=request.eval_type,
            average_score=average_score,
        )
        session.add(run)
        session.flush()
        for result in results:
            session.add(
                RunItem(
                    run_id=run.id,
                    item_id=str(result["item_id"]),
                    prediction=result["prediction"],
                    score=result["score"],
                )
            )
        session.commit()
        return run.id


def list_runs() -> list[dict]:
    init_db()
    with SessionLocal() as session:
        runs = session.query(Run).order_by(Run.id.desc()).all()
        return [
            {
                "id": run.id,
                "dataset_name": run.dataset_name,
                "model_name": run.model_name,
                "eval_type": run.eval_type,
                "average_score": run.average_score,
            }
            for run in runs
        ]
