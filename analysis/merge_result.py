import json, argparse
from pathlib import Path
from run_paths import add_run_args, resolve_run

def discard_main():
    results = Path(__file__).resolve().parent.parent / "results"

    with open(results / "result_Bn.json", encoding="utf-8") as f:
        client = json.load(f)

    with open(results / "engine_Bn.json", encoding="utf-8") as f:
        engine = json.load(f)

    client_batches = set(client)
    engine_batches = set(engine)
    if client_batches != engine_batches:
        raise ValueError(
            f"batch keys mismatch: "
            f"client_only={sorted(client_batches - engine_batches)}, "
            f"engine_only={sorted(engine_batches - client_batches)}"
        )

    merged = {}
    fields = []

    for b in sorted(client, key=int):
        client_record = client[b]
        engine_record = engine[b]
        overlap = client_record.keys() & engine_record.keys()
        conflicts = {
            key
            for key in overlap
            if client_record[key] != engine_record[key]
        }
        if conflicts:
            raise ValueError(
                f"conflicting fields for B={b}: {sorted(conflicts)}"
            )

        record = {**client_record, **engine_record}
        if not fields:
            fields = list(record)
        elif set(record) != set(fields):
            raise ValueError(f"field set mismatch for B={b}")
        merged[b] = record

    with open(
        results / "merge_result_Bn.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(merged, f, ensure_ascii=False, indent=2)

    print(f"B={len(merged)}, columns={len(fields)}")

if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    add_run_args(arg_parser)
    args = arg_parser.parse_args()

    run = resolve_run(
        label=args.label,
        request_dir=args.request_dir,
        metrics_dir=args.metrics_dir,
        out_dir=args.out_dir,
    )

    with (run.out_dir / "result_Bn.json").open(
        "r", 
        encoding="utf-8", 
    ) as file:
        client = json.load(file)

    with (run.out_dir / "engine_Bn.json").open(
        "r",
        encoding="utf-8",
    ) as file:
        engine = json.load(file)

    client_batches = set(client)
    engine_batches = set(engine)

    if client_batches != engine_batches:
        raise ValueError(
            f"batch keys mismatch: "
            f"client_only={sorted(client_batches - engine_batches)}, "
            f"engine_only={sorted(engine_batches - client_batches)}"
        )

    merged = {}
    fields = []

    for batch in sorted(client, key=int):
        client_record = client[batch]
        engine_record = engine[batch]

        overlap = client_record.keys() & engine_record.keys()
        conflicts = {
            key
            for key in overlap
            if client_record[key] != engine_record[key]
        }

        if conflicts:
            raise ValueError(
                f"conflicting fields for B={batch}: {sorted(conflicts)}"
            )
        record = {**client_record, **engine_record}

        if not fields:
            fields = list(record)
        elif set(record) != set(fields):
            raise ValueError(f"field set mismatch for B={batch}")

        merged[batch] = record
    run.out_dir.mkdir(parents=True, exist_ok=True)

    with (run.out_dir / "merge_result_Bn.json").open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(merged, file, ensure_ascii=False, indent=2)

    print(f"{run.label}: B={len(merged)}, columns={len(fields)}")
    
            