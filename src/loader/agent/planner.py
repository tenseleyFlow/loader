"""Planning system for complex tasks."""

import re
from dataclasses import dataclass, field


@dataclass
class PlanStep:
    """A single step in a plan."""
    number: int
    description: str
    status: str = "pending"  # pending, in_progress, completed, failed


@dataclass
class Plan:
    """A plan for completing a task."""
    goal: str
    steps: list[PlanStep] = field(default_factory=list)
    current_step: int = 0

    def next_step(self) -> PlanStep | None:
        """Get the next pending step."""
        for step in self.steps:
            if step.status == "pending":
                step.status = "in_progress"
                self.current_step = step.number
                return step
        return None

    def complete_current(self) -> None:
        """Mark current step as completed."""
        for step in self.steps:
            if step.status == "in_progress":
                step.status = "completed"
                break

    def fail_current(self, reason: str = "") -> None:
        """Mark current step as failed."""
        for step in self.steps:
            if step.status == "in_progress":
                step.status = "failed"
                break

    def is_complete(self) -> bool:
        """Check if all steps are completed."""
        return all(s.status == "completed" for s in self.steps)

    def progress_str(self) -> str:
        """Get progress string like [2/5]."""
        completed = sum(1 for s in self.steps if s.status == "completed")
        return f"[{completed}/{len(self.steps)}]"

    def to_prompt(self) -> str:
        """Format plan for inclusion in prompt."""
        lines = [f"Goal: {self.goal}", "", "Steps:"]
        for step in self.steps:
            status_icon = {
                "pending": "○",
                "in_progress": "►",
                "completed": "✓",
                "failed": "✗",
            }.get(step.status, "?")
            lines.append(f"  {status_icon} {step.number}. {step.description}")
        return "\n".join(lines)


PLANNING_PROMPT = """Before starting this task, create a brief plan.

Task: {task}

Output a numbered list of steps (3-7 steps). Be specific but concise.
Format:
1. First step
2. Second step
...

Only output the numbered steps, nothing else."""


SHOULD_PLAN_PROMPT = """Determine if this task needs planning or can be done directly.

Task: {task}

Tasks that need planning:
- Multiple file changes
- Complex refactoring
- Feature implementation
- Bug investigation
- Multi-step operations

Tasks that DON'T need planning:
- Simple questions
- Single file reads
- Quick lookups
- One-line fixes
- Explanations

Reply with only: PLAN or DIRECT"""


def parse_plan(text: str, goal: str) -> Plan:
    """Parse a numbered list into a Plan."""
    steps = []

    # Match numbered items like "1. description" or "1) description"
    pattern = r"^\s*(\d+)[.\)]\s*(.+)$"

    for line in text.strip().split("\n"):
        match = re.match(pattern, line.strip())
        if match:
            num = int(match.group(1))
            desc = match.group(2).strip()
            steps.append(PlanStep(number=num, description=desc))

    # If no numbered steps found, try to extract any lines as steps
    if not steps:
        for i, line in enumerate(text.strip().split("\n"), 1):
            line = line.strip()
            if line and not line.startswith("#"):
                # Remove leading markers like "- " or "* "
                line = re.sub(r"^[-*]\s*", "", line)
                if line:
                    steps.append(PlanStep(number=i, description=line))

    return Plan(goal=goal, steps=steps)


def should_plan(response: str) -> bool:
    """Parse the should-plan response."""
    response = response.strip().upper()
    return "PLAN" in response and "DIRECT" not in response


def format_step_prompt(plan: Plan, step: PlanStep) -> str:
    """Format a prompt for executing a specific step."""
    return f"""Current plan:
{plan.to_prompt()}

Now execute step {step.number}: {step.description}

Focus only on this step. Use tools as needed."""
