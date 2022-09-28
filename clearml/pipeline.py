from clearml import PipelineController
from config import PipelineConfig


def main(config: PipelineConfig):

    pipe = PipelineController(
        name=config.task_name,
        project=config.project,
        version=config.version
    )

    pipe.add_parameter("dataset_id", config.dataset_id)
    pipe.set_default_execution_queue("default")

    pipe.add_step(
        name="validate_dataset",
        base_task_project=config.project,
        base_task_name="validate_dataset",
        parameter_override={"General/dataset_id": "${pipeline.dataset_id}"},
    )
    pipe.add_step(
        name="preprocess_dataset",
        parents=[
            "validate_dataset",
        ],
        base_task_project=config.project,
        base_task_name="preprocess_dataset",
        parameter_override={
            "General/dataset_id": "${validate_dataset.parameters.General/dataset_id}"
        },
    )
    pipe.add_step(
        name="train",
        parents=[
            "preprocess_dataset",
        ],
        base_task_project=config.project,
        base_task_name="train",
        parameter_override={
            "General/dataset_id": "${preprocess_dataset.parameters.General/output_dataset_id}"
        },
    )

    if config.local:
        pipe.start_locally(run_pipeline_steps_locally=True)
    else:
        pipe.start(queue="default")


if __name__ == "__main__":
    config = PipelineConfig.parse_raw()
    main(config=config)