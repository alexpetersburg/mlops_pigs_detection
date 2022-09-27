import argparse

from clearml.automation import TaskScheduler

def main(args):
    scheduler = TaskScheduler()
    scheduler.add_task(schedule_task_id=args.task_id, queue=args.queue, month=1, target_project=args.project)
    scheduler.start_remotely(queue=args.queue)

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--task_id',
        default="dea380cd9319492eadba050bce9c7489",)
    parser.add_argument(
        '--queue',
        default="default",)
    parser.add_argument(
        '--project',
        default="mlops_pigs_detection", )
    return parser.parse_args()

if __name__ == '__main__':
    args = get_args()
    main(args)