# Devin/modules/scheduler.py
# Purpose: Provides a persistent, feature-rich scheduler for managing and
#          executing tasks at specific times, on intervals, or via cron-like patterns.
# Manages task schedules ⏰

import logging
import time
import json
import uuid
from enum import Enum
from pathlib import Path
from datetime import datetime, timedelta, timezone
from typing import List, Dict, Any, Optional, Callable

# Configure basic logging
logger = logging.getLogger("Scheduler")
if not logger.handlers:
    _console_handler = logging.StreamHandler()
    _console_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logger.addHandler(_console_handler)
    logger.setLevel(logging.INFO)

class JobStatus(Enum):
    """Status of a scheduled job."""
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    SUCCEEDED = "SUCCEEDED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"
    AWAITING_DEPENDENCY = "AWAITING_DEPENDENCY"

@dataclass
class ScheduledJob:
    """Represents a single job in the scheduler."""
    job_id: str
    task_name: str
    schedule_info: Dict[str, Any] # e.g., {"type": "cron", "value": "*/5 * * * *"} or {"type": "at", "value": "ISO_timestamp"}
    args: Optional[List[Any]] = None
    kwargs: Optional[Dict[str, Any]] = None
    status: JobStatus = JobStatus.PENDING
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    last_run_at: Optional[str] = None
    next_run_at: Optional[str] = None
    last_run_status: Optional[JobStatus] = None
    depends_on_job_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Serializes the job to a dictionary for persistence."""
        return {
            "job_id": self.job_id,
            "task_name": self.task_name,
            "schedule_info": self.schedule_info,
            "args": self.args or [],
            "kwargs": self.kwargs or {},
            "status": self.status.value,
            "created_at": self.created_at,
            "last_run_at": self.last_run_at,
            "next_run_at": self.next_run_at,
            "last_run_status": self.last_run_status.value if self.last_run_status else None,
            "depends_on_job_id": self.depends_on_job_id,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ScheduledJob':
        """Deserializes a dictionary back into a ScheduledJob object."""
        return cls(
            job_id=data["job_id"],
            task_name=data["task_name"],
            schedule_info=data["schedule_info"],
            args=data.get("args"),
            kwargs=data.get("kwargs"),
            status=JobStatus(data.get("status", "PENDING")),
            created_at=data["created_at"],
            last_run_at=data.get("last_run_at"),
            next_run_at=data.get("next_run_at"),
            last_run_status=JobStatus(data["last_run_status"]) if data.get("last_run_status") else None,
            depends_on_job_id=data.get("depends_on_job_id"),
        )

class Scheduler:
    """
    A persistent and robust task scheduler.
    """
    def __init__(self, schedule_file_path: str = "devin_schedule.json"):
        self.schedule_file = Path(schedule_file_path)
        self.jobs: Dict[str, ScheduledJob] = {}
        self.task_registry: Dict[str, Callable] = {} # Maps task names to functions
        self._load_schedule()
        logger.info(f"Scheduler initialized. Loaded {len(self.jobs)} jobs from '{self.schedule_file}'.")

    def register_task(self, func: Callable, name: Optional[str] = None) -> None:
        """Registers a function so it can be scheduled by name."""
        task_name = name or func.__name__
        if task_name in self.task_registry:
            logger.warning(f"Task '{task_name}' is already registered. Overwriting.")
        self.task_registry[task_name] = func
        logger.info(f"Task '{task_name}' registered with the scheduler.")

    def _save_schedule(self) -> None:
        """Saves the current schedule of jobs to the persistence file."""
        try:
            with open(self.schedule_file, "w") as f:
                json.dump([job.to_dict() for job in self.jobs.values()], f, indent=2)
            logger.debug(f"Schedule saved successfully to '{self.schedule_file}'.")
        except IOError as e:
            logger.error(f"Failed to save schedule to '{self.schedule_file}': {e}")

    def _load_schedule(self) -> None:
        """Loads the schedule of jobs from the persistence file on startup."""
        if not self.schedule_file.exists():
            logger.warning(f"Schedule file '{self.schedule_file}' not found. Starting with an empty schedule.")
            return
        try:
            with open(self.schedule_file, "r") as f:
                jobs_data = json.load(f)
                self.jobs = {job_data["job_id"]: ScheduledJob.from_dict(job_data) for job_data in jobs_data}
        except (IOError, json.JSONDecodeError) as e:
            logger.error(f"Failed to load or parse schedule from '{self.schedule_file}': {e}. Starting fresh.")
            self.jobs = {}

    def _calculate_next_run(self, job: ScheduledJob, from_time: datetime) -> Optional[datetime]:
        """Conceptually calculates the next run time based on schedule_info."""
        info = job.schedule_info
        if info["type"] == "at":
            next_run = datetime.fromisoformat(info["value"])
            return next_run if next_run > from_time else None # Only run once in the future
        elif info["type"] == "interval":
            last_run = datetime.fromisoformat(job.last_run_at) if job.last_run_at else from_time
            return last_run + timedelta(minutes=info["value"])
        elif info["type"] == "cron":
            # This is a highly simplified cron parser for demo purposes.
            # A real implementation would use a library like 'croniter'.
            logger.debug(f"CONCEPTUAL CRON: Calculating next run for '{info['value']}' from {from_time.isoformat()}")
            return from_time + timedelta(minutes=5) # Assume every 5 mins for demo
        return None

    def schedule_job(self,
                     task_name: str,
                     schedule_info: Dict[str, Any],
                     args: Optional[List] = None,
                     kwargs: Optional[Dict] = None,
                     depends_on: Optional[str] = None) -> Optional[str]:
        """
        Schedules a new task.

        Args:
            task_name (str): The registered name of the task function.
            schedule_info (Dict): e.g., {"type": "cron", "value": "0 * * * *"}
            depends_on (Optional[str]): The job_id of a job that must complete successfully first.

        Returns:
            Optional[str]: The new job's ID, or None if the task is not registered.
        """
        if task_name not in self.task_registry:
            logger.error(f"Cannot schedule job: Task '{task_name}' is not registered.")
            return None
            
        job_id = f"job_{uuid.uuid4().hex[:8]}"
        job = ScheduledJob(
            job_id=job_id,
            task_name=task_name,
            schedule_info=schedule_info,
            args=args,
            kwargs=kwargs,
            depends_on_job_id=depends_on
        )
        job.next_run_at = (self._calculate_next_run(job, datetime.now(timezone.utc)) or datetime.max.replace(tzinfo=timezone.utc)).isoformat()
        self.jobs[job_id] = job
        self._save_schedule()
        logger.info(f"Job '{job_id}' for task '{task_name}' scheduled. Next run (conceptual): {job.next_run_at}")
        return job_id

    def cancel_job(self, job_id: str) -> bool:
        """Removes a job from the schedule."""
        if job_id in self.jobs:
            del self.jobs[job_id]
            self._save_schedule()
            logger.info(f"Job '{job_id}' has been cancelled and removed from the schedule.")
            return True
        logger.warning(f"Cannot cancel job: Job ID '{job_id}' not found.")
        return False

    def _run_job(self, job: ScheduledJob):
        """Conceptually executes a job in a separate thread/process."""
        func = self.task_registry.get(job.task_name)
        if not func:
            logger.error(f"Cannot run job '{job.job_id}': Task '{job.task_name}' is no longer registered.")
            job.status = JobStatus.FAILED
            self._save_schedule()
            return
        
        logger.info(f"--- Running Job '{job.job_id}' (Task: {job.task_name}) ---")
        job.status = JobStatus.RUNNING
        job.last_run_at = datetime.now(timezone.utc).isoformat()
        
        try:
            # In a real system, this would be `threading.Thread(target=...).start()`
            # or submitted to a process pool.
            time.sleep(0.2) # Simulate work
            args = job.args or []
            kwargs = job.kwargs or {}
            result = func(*args, **kwargs)
            logger.info(f"--- Job '{job.job_id}' completed successfully. ---")
            job.status = JobStatus.SUCCEEDED
            job.last_run_status = JobStatus.SUCCEEDED
        except Exception as e:
            logger.error(f"--- Job '{job.job_id}' FAILED during execution: {e} ---")
            job.status = JobStatus.FAILED
            job.last_run_status = JobStatus.FAILED
        
        # Reschedule if it's a recurring job
        next_run = self._calculate_next_run(job, datetime.now(timezone.utc))
        if next_run:
            job.next_run_at = next_run.isoformat()
            job.status = JobStatus.PENDING # Reset for the next run
        else:
            job.next_run_at = None # It was a one-off job

        self._save_schedule()

    def tick(self):
        """
        The main heartbeat of the scheduler. This should be called periodically in a loop.
        It checks for and runs any jobs that are due.
        """
        logger.debug("Scheduler tick: Checking for due jobs.")
        now = datetime.now(timezone.utc)
        
        for job in sorted(self.jobs.values(), key=lambda j: j.next_run_at or 'Z'):
            if job.status in [JobStatus.RUNNING, JobStatus.CANCELLED] or not job.next_run_at:
                continue
            
            next_run_time = datetime.fromisoformat(job.next_run_at)
            if now >= next_run_time:
                # Check dependencies
                if job.depends_on_job_id:
                    dependency_job = self.jobs.get(job.depends_on_job_id)
                    if not dependency_job or dependency_job.last_run_status != JobStatus.SUCCEEDED:
                        job.status = JobStatus.AWAITING_DEPENDENCY
                        logger.warning(f"Job '{job.job_id}' is due but is awaiting successful completion of dependency '{job.depends_on_job_id}'.")
                        continue
                
                self._run_job(job)


# --- Example Usage ---
def example_report_task(report_type: str):
    """A simple function that the scheduler can run."""
    print(f"Generating conceptual '{report_type}' report...")
    # Simulate report generation
    time.sleep(0.1)
    print(f"'{report_type}' report generated and saved to /tmp/reports/{report_type}_{uuid.uuid4().hex[:4]}.pdf")

def example_cleanup_task():
    """Another simple task."""
    print("Running conceptual daily cleanup task: Deleting temp files...")

def example_dependent_task():
    """A task that depends on another."""
    print("Running dependent task! This should only run after the cleanup task succeeds.")

if __name__ == "__main__":
    print("=========================================================")
    print("=== Persistent Scheduler Module Prototype ⏰ ===")
    print("=========================================================")
    
    # --- 1. Initialize Scheduler and Register Tasks ---
    # Using a temporary file for the demo schedule
    schedule_file = "temp_devin_schedule.json"
    if Path(schedule_file).exists():
        Path(schedule_file).unlink()

    scheduler = Scheduler(schedule_file_path=schedule_file)
    scheduler.register_task(example_report_task)
    scheduler.register_task(example_cleanup_task)
    scheduler.register_task(example_dependent_task)

    # --- 2. Schedule a few jobs ---
    print("\n--- Scheduling Jobs ---")
    # A one-off job to run in 2 seconds (conceptually)
    run_at_time = datetime.now(timezone.utc) + timedelta(seconds=2)
    scheduler.schedule_job("example_report_task", {"type": "at", "value": run_at_time.isoformat()}, args=["Weekly Summary"])
    
    # A recurring job
    scheduler.schedule_job("example_cleanup_task", {"type": "interval", "value": 1}) # Every 1 minute conceptually
    
    # A job that depends on the cleanup task
    job_id_to_wait_for = list(scheduler.jobs.keys())[1] # Get ID of cleanup task
    scheduler.schedule_job("example_dependent_task", {"type": "interval", "value": 2}, depends_on=job_id_to_wait_for)
    
    print(f"\n{len(scheduler.jobs)} jobs scheduled. Current schedule state saved to '{schedule_file}'.")

    # --- 3. Simulate the Scheduler's event loop ---
    print("\n--- Simulating Scheduler Ticks over Time ---")
    for i in range(4):
        print(f"\n>>> Scheduler Tick {i+1} at {datetime.now(timezone.utc).time().isoformat()} <<<")
        scheduler.tick()
        time.sleep(0.6) # Simulate time passing between ticks
    
    # --- 4. Cancel a job ---
    print("\n--- Cancelling a Job ---")
    job_to_cancel = list(scheduler.jobs.keys())[0] # Cancel the one-off report task if it hasn't run
    scheduler.cancel_job(job_to_cancel)
    
    # --- 5. Demonstrate Persistence ---
    print("\n--- Demonstrating Persistence ---")
    print("Initializing a NEW scheduler instance from the same file...")
    new_scheduler = Scheduler(schedule_file_path=schedule_file)
    print(f"New scheduler loaded {len(new_scheduler.jobs)} jobs. This demonstrates that the schedule state was saved and reloaded.")
    print("Remaining jobs:", [job.task_name for job in new_scheduler.jobs.values()])

    # Clean up the temp file
    if Path(schedule_file).exists():
        Path(schedule_file).unlink()

    print("\n=========================================================")
    print("=== Scheduler Prototype Complete ===")
    print("=========================================================")
