"""Shell command execution tools with stateful job control."""

from __future__ import annotations

import asyncio
import contextlib
from dataclasses import dataclass, field
from datetime import UTC, datetime
import os
from pathlib import Path
import shlex
import signal
import subprocess
from typing import Any

from ..runtime.permissions import PermissionMode
from .base import ConfirmationRequired, Tool, ToolResult


def _utc_now() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


@dataclass(slots=True)
class _OutputBuffer:
    """Bounded text buffer for process output."""

    limit: int
    chunks: list[str] = field(default_factory=list)
    size: int = 0
    truncated: bool = False

    def append(self, data: bytes) -> None:
        text = data.decode("utf-8", errors="replace")
        if not text:
            return
        remaining = max(self.limit - self.size, 0)
        if remaining <= 0:
            self.truncated = True
            return
        kept = text[:remaining]
        if kept:
            self.chunks.append(kept)
            self.size += len(kept)
        if len(text) > remaining:
            self.truncated = True

    def text(self) -> str:
        return "".join(self.chunks)


@dataclass(slots=True)
class BashJob:
    """One tracked bash subprocess."""

    job_id: str
    command: str
    cwd: str | None
    background: bool
    timeout: float | None
    mutability: str
    started_at: str
    pid: int
    process: asyncio.subprocess.Process
    stdout_buffer: _OutputBuffer
    stderr_buffer: _OutputBuffer
    status: str = "running"
    exit_code: int | None = None
    finished_at: str | None = None
    timed_out: bool = False
    interrupted: bool = False
    killed: bool = False
    stdout_task: asyncio.Task[None] | None = None
    stderr_task: asyncio.Task[None] | None = None
    completion_task: asyncio.Task[None] | None = None

    @property
    def is_running(self) -> bool:
        return self.process.returncode is None


class BashJobManager:
    """Track bash subprocesses for the lifetime of one Loader runtime."""

    def __init__(self, *, output_limit: int = 50_000, recent_limit: int = 25) -> None:
        self.output_limit = output_limit
        self.recent_limit = recent_limit
        self._jobs: dict[str, BashJob] = {}
        self._job_order: list[str] = []
        self._counter = 0
        self._active_foreground_job_id: str | None = None

    @property
    def active_foreground_job_id(self) -> str | None:
        return self._active_foreground_job_id

    def list_jobs(self, *, limit: int | None = None) -> list[BashJob]:
        selected: list[BashJob] = []
        for job_id in reversed(self._job_order):
            job = self._jobs[job_id]
            selected.append(job)
            if limit is not None and len(selected) >= limit:
                break
        return selected

    def get_job(self, job_id: str) -> BashJob | None:
        return self._jobs.get(job_id)

    async def start(
        self,
        *,
        command: str,
        cwd: str | None,
        timeout: float | None,
        background: bool,
        mutability: str,
    ) -> BashJob:
        resolved_cwd = str(Path(cwd).expanduser().resolve()) if cwd else None
        self._counter += 1
        job_id = f"bash-{self._counter}"

        popen_kwargs: dict[str, Any] = {
            "stdout": asyncio.subprocess.PIPE,
            "stderr": asyncio.subprocess.PIPE,
            "cwd": resolved_cwd,
        }
        if os.name == "nt":
            popen_kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
        else:
            popen_kwargs["start_new_session"] = True

        process = await asyncio.create_subprocess_shell(command, **popen_kwargs)
        job = BashJob(
            job_id=job_id,
            command=command,
            cwd=resolved_cwd,
            background=background,
            timeout=timeout,
            mutability=mutability,
            started_at=_utc_now(),
            pid=process.pid,
            process=process,
            stdout_buffer=_OutputBuffer(self.output_limit),
            stderr_buffer=_OutputBuffer(self.output_limit),
        )
        job.stdout_task = asyncio.create_task(self._read_stream(job.stdout_buffer, process.stdout))
        job.stderr_task = asyncio.create_task(self._read_stream(job.stderr_buffer, process.stderr))
        job.completion_task = asyncio.create_task(self._monitor_job(job))

        self._jobs[job_id] = job
        self._job_order.append(job_id)
        self._trim_completed_jobs()
        if not background:
            self._active_foreground_job_id = job_id
        return job

    async def wait_for_job(
        self,
        job_id: str,
        *,
        timeout: float | None = None,
    ) -> ToolResult:
        job = self._jobs.get(job_id)
        if job is None:
            return ToolResult(f"Unknown bash job: {job_id}", is_error=True)
        if job.completion_task is None:
            return ToolResult(f"Bash job {job_id} has no completion task", is_error=True)

        if job.is_running and timeout is not None:
            try:
                await asyncio.wait_for(asyncio.shield(job.completion_task), timeout)
            except TimeoutError:
                return ToolResult(
                    f"Wait timed out after {timeout}s; bash job {job_id} is still running.",
                    is_error=True,
                    metadata=self.metadata_for(job),
                )
        else:
            await asyncio.shield(job.completion_task)
        return self.tool_result_for(job)

    async def kill_job(
        self,
        job_id: str,
        *,
        force_after_ms: int = 1_000,
        interrupted: bool = False,
    ) -> ToolResult:
        job = self._jobs.get(job_id)
        if job is None:
            return ToolResult(f"Unknown bash job: {job_id}", is_error=True)
        if not job.is_running:
            return ToolResult(
                f"Bash job {job_id} is already {job.status}.",
                metadata=self.metadata_for(job),
            )

        await self._terminate_job(job, force_after_ms=force_after_ms, interrupted=interrupted)
        if job.completion_task is not None:
            await asyncio.shield(job.completion_task)
        status = "Interrupted" if interrupted else "Stopped"
        summary = f"{status} bash job {job.job_id} (pid {job.pid})."
        output = self._render_combined_output(job)
        if output != "(no output)":
            summary += f"\n\n{output}"
        return ToolResult(summary, metadata=self.metadata_for(job))

    async def interrupt_active_foreground(self) -> ToolResult | None:
        if self._active_foreground_job_id is None:
            return None
        job = self._jobs.get(self._active_foreground_job_id)
        if job is None or not job.is_running:
            return None
        return await self.kill_job(job.job_id, interrupted=True)

    def terminate_all_now(self) -> list[str]:
        killed: list[str] = []
        for job in self._jobs.values():
            if not job.is_running:
                continue
            job.killed = True
            job.interrupted = True
            self._send_signal(job.process, signal.SIGTERM)
            self._send_signal(job.process, signal.SIGKILL)
            killed.append(job.job_id)
        self._active_foreground_job_id = None
        return killed

    def render_jobs(self, *, limit: int = 20) -> tuple[str, dict[str, Any]]:
        jobs = self.list_jobs(limit=limit)
        if not jobs:
            return "No bash jobs tracked for this Loader session.", {"jobs": []}

        lines = ["Bash jobs:"]
        for job in jobs:
            marker = "bg" if job.background else "fg"
            lines.append(
                f"- {job.job_id} [{job.status}] ({marker}, pid={job.pid}) {job.command}"
            )
        return "\n".join(lines), {
            "jobs": [self.metadata_for(job) for job in jobs],
            "active_foreground_job_id": self._active_foreground_job_id,
        }

    def metadata_for(self, job: BashJob) -> dict[str, Any]:
        return {
            "job_id": job.job_id,
            "pid": job.pid,
            "command": job.command,
            "cwd": job.cwd,
            "background": job.background,
            "status": job.status,
            "started_at": job.started_at,
            "finished_at": job.finished_at,
            "exit_code": job.exit_code,
            "stdout": job.stdout_buffer.text(),
            "stderr": job.stderr_buffer.text(),
            "stdout_truncated": job.stdout_buffer.truncated,
            "stderr_truncated": job.stderr_buffer.truncated,
            "truncated": job.stdout_buffer.truncated or job.stderr_buffer.truncated,
            "timed_out": job.timed_out,
            "interrupted": job.interrupted,
            "killed": job.killed,
            "running": job.is_running,
            "output_limit": self.output_limit,
        }

    def tool_result_for(self, job: BashJob) -> ToolResult:
        output = self._render_combined_output(job)
        metadata = self.metadata_for(job)
        if job.timed_out:
            return ToolResult(
                f"Command timed out after {job.timeout}s\n{output}",
                is_error=True,
                metadata=metadata,
            )
        if job.interrupted or job.killed:
            return ToolResult(
                f"Command interrupted\n{output}",
                is_error=True,
                metadata=metadata,
            )
        if (job.exit_code or 0) != 0:
            return ToolResult(
                f"Exit code {job.exit_code}\n{output}",
                is_error=True,
                metadata=metadata,
            )
        return ToolResult(output, metadata=metadata)

    def launch_result_for(self, job: BashJob) -> ToolResult:
        output = (
            f"Started background bash job {job.job_id} (pid {job.pid}).\n"
            f"Use bash_wait(job_id=\"{job.job_id}\") to wait for completion or "
            f"bash_kill(job_id=\"{job.job_id}\") to stop it."
        )
        return ToolResult(output, metadata=self.metadata_for(job))

    async def _read_stream(
        self,
        buffer: _OutputBuffer,
        stream: asyncio.StreamReader | None,
    ) -> None:
        if stream is None:
            return
        while True:
            chunk = await stream.read(4096)
            if not chunk:
                break
            buffer.append(chunk)

    async def _monitor_job(self, job: BashJob) -> None:
        try:
            if job.timeout is None:
                await job.process.wait()
            else:
                try:
                    await asyncio.wait_for(job.process.wait(), timeout=job.timeout)
                except TimeoutError:
                    job.timed_out = True
                    await self._terminate_job(
                        job,
                        force_after_ms=500,
                        interrupted=True,
                    )
                    await job.process.wait()
        finally:
            await asyncio.gather(
                job.stdout_task or asyncio.sleep(0),
                job.stderr_task or asyncio.sleep(0),
            )
            job.exit_code = job.process.returncode
            job.finished_at = _utc_now()
            if job.timed_out:
                job.status = "timed_out"
            elif (job.interrupted or job.killed) and job.background:
                job.status = "killed"
            elif job.interrupted:
                job.status = "interrupted"
            elif (job.exit_code or 0) == 0:
                job.status = "completed"
            else:
                job.status = "failed"
            if self._active_foreground_job_id == job.job_id:
                self._active_foreground_job_id = None

    async def _terminate_job(
        self,
        job: BashJob,
        *,
        force_after_ms: int,
        interrupted: bool,
    ) -> None:
        if not job.is_running:
            return
        job.interrupted = interrupted
        job.killed = not interrupted
        self._send_signal(job.process, signal.SIGTERM)
        try:
            await asyncio.wait_for(job.process.wait(), timeout=force_after_ms / 1000)
        except TimeoutError:
            self._send_signal(job.process, signal.SIGKILL)

    def _send_signal(
        self,
        process: asyncio.subprocess.Process,
        sig: int,
    ) -> None:
        if process.returncode is not None:
            return
        with contextlib.suppress(ProcessLookupError):
            if os.name == "nt":
                process.send_signal(sig)
            else:
                os.killpg(process.pid, sig)

    def _render_combined_output(self, job: BashJob) -> str:
        parts: list[str] = []
        stdout = job.stdout_buffer.text()
        stderr = job.stderr_buffer.text()
        if stdout:
            parts.append(stdout)
        if stderr.strip():
            parts.append(f"[stderr]\n{stderr}")
        output = "\n".join(parts) if parts else "(no output)"
        if job.stdout_buffer.truncated or job.stderr_buffer.truncated:
            output += "\n\n... (output truncated)"
        return output

    def _trim_completed_jobs(self) -> None:
        if len(self._job_order) <= self.recent_limit:
            return
        overflow = len(self._job_order) - self.recent_limit
        for job_id in list(self._job_order):
            if overflow <= 0:
                break
            job = self._jobs[job_id]
            if job.is_running:
                continue
            self._job_order.remove(job_id)
            self._jobs.pop(job_id, None)
            overflow -= 1


class BashTool(Tool):
    """Execute bash commands and manage their subprocess lifecycle."""

    required_permission = PermissionMode.DANGER_FULL_ACCESS
    OUTPUT_LIMIT = 50_000

    SAFE_COMMANDS = {
        "ls", "cat", "head", "tail", "grep", "find", "pwd", "whoami", "date",
        "wc", "sort", "uniq", "diff", "file", "stat", "du", "df",
        "git status", "git log", "git diff", "git branch", "git show",
        "python --version", "node --version", "npm --version",
        "uv --version", "pip list", "pip show",
    }

    LONG_RUNNING_HINTS = (
        "python -m http.server",
        "python3 -m http.server",
        "npm run dev",
        "npm run start",
        "npm run serve",
        "npm run watch",
        "pnpm dev",
        "yarn dev",
        "bun run dev",
        "vite",
        "next dev",
        "tail -f",
        "watch ",
        "uvicorn --reload",
    )

    def __init__(
        self,
        timeout: float = 120.0,
        allowed_commands: list[str] | None = None,
        manager: BashJobManager | None = None,
    ) -> None:
        self.timeout = timeout
        self.allowed_commands = allowed_commands
        self.manager = manager or BashJobManager(output_limit=self.OUTPUT_LIMIT)

    @property
    def name(self) -> str:
        return "bash"

    @property
    def description(self) -> str:
        return (
            "Execute a bash command and return the output. For servers, watchers, or "
            "other long-running processes, set background=true and inspect them later "
            "with bash_wait or bash_jobs, or stop them with bash_kill."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "The bash command to execute",
                },
                "cwd": {
                    "type": "string",
                    "description": "Working directory to run the command in (default: current directory)",
                },
                "timeout": {
                    "type": "number",
                    "description": f"Timeout in seconds (default: {self.timeout})",
                    "default": self.timeout,
                },
                "background": {
                    "type": "boolean",
                    "description": (
                        "Run the command in the background and return immediately. "
                        "Use this for servers, watchers, and preview processes."
                    ),
                    "default": False,
                },
            },
            "required": ["command"],
        }

    @property
    def is_destructive(self) -> bool:
        return True

    def _is_safe_command(self, command: str) -> bool:
        cmd = command.strip().lower()
        for safe in self.SAFE_COMMANDS:
            if cmd == safe or cmd.startswith(safe + " "):
                return True
        return False

    def get_required_permission(self, **kwargs: Any) -> PermissionMode:
        command = str(kwargs.get("command", ""))
        return self.classify_command_permission(command)

    def classify_command_permission(self, command: str) -> PermissionMode:
        normalized = command.strip().lower()
        if not normalized:
            return PermissionMode.DANGER_FULL_ACCESS
        if self._is_safe_command(normalized):
            return PermissionMode.READ_ONLY

        danger_signals = (
            "sudo ",
            "chmod ",
            "chown ",
            "rm -",
            "mkfs",
            "dd ",
            "/etc/",
            "/usr/",
            "/bin/",
            "/sbin/",
            "/boot/",
            "/proc/",
            "/sys/",
        )
        if any(signal in normalized for signal in danger_signals):
            return PermissionMode.DANGER_FULL_ACCESS

        return PermissionMode.WORKSPACE_WRITE

    def check_confirmation(self, skip_confirmation: bool = False, **kwargs: Any) -> None:
        if skip_confirmation:
            return
        command = kwargs.get("command", "")
        if self._is_safe_command(command):
            return
        raise ConfirmationRequired(
            tool_name=self.name,
            message=f"Run command: {command[:50]}{'...' if len(command) > 50 else ''}",
            details=command,
        )

    def _is_command_allowed(self, command: str) -> bool:
        if self.allowed_commands is None:
            return True
        try:
            parts = shlex.split(command)
        except ValueError:
            return False
        if not parts:
            return False
        return parts[0] in self.allowed_commands

    def _looks_long_running(self, command: str) -> bool:
        normalized = " ".join(command.lower().split())
        return any(hint in normalized for hint in self.LONG_RUNNING_HINTS)

    async def execute(
        self,
        command: str,
        cwd: str | None = None,
        timeout: float | None = None,
        background: bool = False,
        **kwargs: Any,
    ) -> ToolResult:
        del kwargs
        if not self._is_command_allowed(command):
            return ToolResult(
                f"Command not allowed. Allowed commands: {self.allowed_commands}",
                is_error=True,
            )

        effective_timeout = self.timeout if timeout is None else timeout
        if not background and self._looks_long_running(command):
            return ToolResult(
                "This command looks long-running and would block Loader in the foreground. "
                "Re-run it with background=true, then use bash_wait or bash_jobs to inspect it.",
                is_error=True,
                metadata={
                    "command": command,
                    "cwd": str(Path(cwd).expanduser().resolve()) if cwd else None,
                    "background": background,
                    "suggest_background": True,
                    "mutability": self.classify_command_permission(command).as_str(),
                },
            )

        try:
            job = await self.manager.start(
                command=command,
                cwd=cwd,
                timeout=effective_timeout,
                background=background,
                mutability=self.classify_command_permission(command).as_str(),
            )
            if background:
                return self.manager.launch_result_for(job)
            if job.completion_task is None:
                return ToolResult("Bash job failed to start correctly", is_error=True)
            try:
                await asyncio.shield(job.completion_task)
            except asyncio.CancelledError:
                await self.manager.kill_job(job.job_id, interrupted=True)
                raise
            return self.manager.tool_result_for(job)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            resolved_cwd = str(Path(cwd).expanduser().resolve()) if cwd else None
            return ToolResult(
                f"Error executing command: {exc}",
                is_error=True,
                metadata={"command": command, "cwd": resolved_cwd, "background": background},
            )


class BashJobsTool(Tool):
    """List tracked bash jobs for the current Loader runtime."""

    required_permission = PermissionMode.READ_ONLY

    def __init__(self, manager: BashJobManager) -> None:
        self.manager = manager

    @property
    def name(self) -> str:
        return "bash_jobs"

    @property
    def description(self) -> str:
        return "List active and recent bash jobs for this Loader session."

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "limit": {
                    "type": "integer",
                    "description": "Maximum number of recent jobs to include",
                    "default": 20,
                },
            },
        }

    async def execute(self, limit: int = 20, **kwargs: Any) -> ToolResult:
        del kwargs
        output, metadata = self.manager.render_jobs(limit=limit)
        return ToolResult(output, metadata=metadata)


class BashWaitTool(Tool):
    """Wait for one tracked background bash job."""

    required_permission = PermissionMode.READ_ONLY

    def __init__(self, manager: BashJobManager) -> None:
        self.manager = manager

    @property
    def name(self) -> str:
        return "bash_wait"

    @property
    def description(self) -> str:
        return "Wait for a tracked background bash job to finish and return its output."

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "job_id": {
                    "type": "string",
                    "description": "Tracked bash job id, for example bash-1",
                },
                "timeout": {
                    "type": "number",
                    "description": "Optional wait timeout in seconds",
                },
            },
            "required": ["job_id"],
        }

    async def execute(
        self,
        job_id: str,
        timeout: float | None = None,
        **kwargs: Any,
    ) -> ToolResult:
        del kwargs
        return await self.manager.wait_for_job(job_id, timeout=timeout)


class BashKillTool(Tool):
    """Stop one tracked bash job."""

    required_permission = PermissionMode.WORKSPACE_WRITE

    def __init__(self, manager: BashJobManager) -> None:
        self.manager = manager

    @property
    def name(self) -> str:
        return "bash_kill"

    @property
    def description(self) -> str:
        return "Stop a tracked bash job started during this Loader session."

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "job_id": {
                    "type": "string",
                    "description": "Tracked bash job id, for example bash-1",
                },
                "force_after_ms": {
                    "type": "integer",
                    "description": "Grace period before force-killing the job process group",
                    "default": 1000,
                },
            },
            "required": ["job_id"],
        }

    async def execute(
        self,
        job_id: str,
        force_after_ms: int = 1_000,
        **kwargs: Any,
    ) -> ToolResult:
        del kwargs
        return await self.manager.kill_job(job_id, force_after_ms=force_after_ms)
