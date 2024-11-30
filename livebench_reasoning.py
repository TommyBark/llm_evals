from inspect_ai import Task, task
from inspect_ai.dataset import Sample, hf_dataset
from inspect_ai.scorer import match
from inspect_ai.solver import generate


def record_to_sample(record):
    return Sample(
        input=record["turns"][0],
        target=record["ground_truth"],
        metadata=dict(source_id=record["question_id"]),
    )


@task
def lb_reasoning():

    # dataset
    dataset = hf_dataset(
        path="livebench/reasoning",
        split="test",
        sample_fields=record_to_sample,
        trust=True,
    )

    # define task
    return Task(
        dataset=dataset,
        solver=[generate()],
        scorer=match(),
    )
