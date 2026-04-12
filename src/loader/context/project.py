"""Project detection and context gathering."""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

ProjectType = Literal["python", "node", "rust", "go", "unknown"]
PackageManager = Literal["pip", "uv", "poetry", "pipenv", "npm", "yarn", "pnpm", "cargo", "go", "unknown"]
TestFramework = Literal["pytest", "unittest", "jest", "mocha", "vitest", "cargo-test", "go-test", "unknown"]


# Config files that identify project types
PROJECT_MARKERS = {
    "python": ["pyproject.toml", "setup.py", "setup.cfg", "requirements.txt", "Pipfile"],
    "node": ["package.json", "yarn.lock", "pnpm-lock.yaml"],
    "rust": ["Cargo.toml"],
    "go": ["go.mod"],
}

# Key files to read for context (in priority order)
KEY_FILES = {
    "python": ["pyproject.toml", "setup.py", "requirements.txt", "README.md"],
    "node": ["package.json", "README.md"],
    "rust": ["Cargo.toml", "README.md"],
    "go": ["go.mod", "README.md"],
    "unknown": ["README.md", "README", "readme.md"],
}

# Common directories to note
COMMON_DIRS = ["src", "lib", "app", "tests", "test", "docs", "scripts", "bin", "cmd"]


# Package manager detection by lockfile (priority order)
PYTHON_LOCKFILES = [
    ("uv.lock", "uv"),
    ("poetry.lock", "poetry"),
    ("Pipfile.lock", "pipenv"),
    ("requirements.txt", "pip"),  # Fallback
]

NODE_LOCKFILES = [
    ("pnpm-lock.yaml", "pnpm"),
    ("yarn.lock", "yarn"),
    ("package-lock.json", "npm"),
]

# Test framework detection
PYTHON_TEST_MARKERS = [
    ("pytest.ini", "pytest"),
    ("conftest.py", "pytest"),
    ("setup.cfg", "pytest"),  # Often has [tool:pytest]
]

NODE_TEST_FRAMEWORKS = ["jest", "mocha", "vitest", "ava", "tap"]


@dataclass
class ProjectContext:
    """Context about the current project."""

    root: Path
    project_type: ProjectType
    config_file: str | None = None
    config_content: str | None = None
    readme_content: str | None = None
    structure: list[str] = field(default_factory=list)
    test_command: str | None = None
    build_command: str | None = None
    run_command: str | None = None
    # New fields for Phase 5
    package_manager: PackageManager = "unknown"
    test_framework: TestFramework = "unknown"
    has_venv: bool = False
    venv_path: str | None = None
    is_venv_active: bool = False

    def to_prompt(self) -> str:
        """Format context for inclusion in system prompt."""
        lines = [f"Project type: {self.project_type}"]

        if self.package_manager != "unknown":
            lines.append(f"Package manager: {self.package_manager}")

        if self.test_framework != "unknown":
            lines.append(f"Test framework: {self.test_framework}")

        if self.has_venv:
            venv_status = "active" if self.is_venv_active else "inactive"
            lines.append(f"Virtual environment: {self.venv_path} ({venv_status})")

        if self.structure:
            lines.append(f"Structure: {', '.join(self.structure)}")

        if self.test_command:
            lines.append(f"Test command: {self.test_command}")

        if self.build_command:
            lines.append(f"Build command: {self.build_command}")

        if self.config_file and self.config_content:
            # Truncate config if too long
            content = self.config_content
            if len(content) > 1000:
                content = content[:1000] + "\n... (truncated)"
            lines.append(f"\n{self.config_file}:\n```\n{content}\n```")

        if self.readme_content:
            # Just first ~500 chars of README
            readme = self.readme_content[:500]
            if len(self.readme_content) > 500:
                readme += "\n... (truncated)"
            lines.append(f"\nREADME (excerpt):\n{readme}")

        return "\n".join(lines)


def detect_project_type(root: Path) -> tuple[ProjectType, str | None]:
    """Detect project type from config files.

    Returns:
        Tuple of (project_type, config_file_name)
    """
    for ptype, markers in PROJECT_MARKERS.items():
        for marker in markers:
            if (root / marker).exists():
                return ptype, marker  # type: ignore
    return "unknown", None


def get_directory_structure(root: Path, max_items: int = 20) -> list[str]:
    """Get top-level directory structure."""
    items = []
    try:
        for item in sorted(root.iterdir()):
            # Skip hidden files and common noise
            if item.name.startswith("."):
                continue
            if item.name in ("node_modules", "__pycache__", ".git", "venv", ".venv", "target"):
                continue

            if item.is_dir():
                items.append(f"{item.name}/")
            else:
                items.append(item.name)

            if len(items) >= max_items:
                break
    except PermissionError:
        pass

    return items


def read_file_safe(path: Path, max_size: int = 10000) -> str | None:
    """Read file contents safely, with size limit."""
    try:
        if not path.exists() or not path.is_file():
            return None
        if path.stat().st_size > max_size:
            return path.read_text()[:max_size] + "\n... (truncated)"
        return path.read_text()
    except (PermissionError, UnicodeDecodeError, OSError):
        return None


def detect_package_manager(project_type: ProjectType, root: Path) -> PackageManager:
    """Detect which package manager is used for the project."""
    if project_type == "python":
        for lockfile, manager in PYTHON_LOCKFILES:
            if (root / lockfile).exists():
                return manager  # type: ignore
        # Check pyproject.toml for build system
        pyproject = root / "pyproject.toml"
        if pyproject.exists():
            content = read_file_safe(pyproject) or ""
            if "[tool.poetry]" in content:
                return "poetry"
            if "uv" in content.lower() and "build-system" in content:
                return "uv"
        return "pip"  # Default for Python
    elif project_type == "node":
        for lockfile, manager in NODE_LOCKFILES:
            if (root / lockfile).exists():
                return manager  # type: ignore
        return "npm"  # Default for Node
    elif project_type == "rust":
        return "cargo"
    elif project_type == "go":
        return "go"
    return "unknown"


def detect_test_framework(project_type: ProjectType, root: Path, config_content: str | None) -> TestFramework:
    """Detect which test framework is used."""
    if project_type == "python":
        # Check for pytest markers
        for marker_file, framework in PYTHON_TEST_MARKERS:
            if (root / marker_file).exists():
                return framework  # type: ignore
        # Check pyproject.toml for pytest config
        if config_content and "[tool.pytest" in config_content:
            return "pytest"
        # Check for tests directory with conftest
        tests_dir = root / "tests"
        if tests_dir.exists() and (tests_dir / "conftest.py").exists():
            return "pytest"
        # Default to pytest if tests dir exists
        if tests_dir.exists() or (root / "test").exists():
            return "pytest"
        return "unittest"  # Python default

    elif project_type == "node":
        # Check package.json for test frameworks
        pkg_json = root / "package.json"
        if pkg_json.exists():
            content = read_file_safe(pkg_json) or ""
            for framework in NODE_TEST_FRAMEWORKS:
                if f'"{framework}"' in content:
                    return framework  # type: ignore
        return "jest"  # Most common default

    elif project_type == "rust":
        return "cargo-test"

    elif project_type == "go":
        return "go-test"

    return "unknown"


def detect_virtual_environment(root: Path) -> tuple[bool, str | None, bool]:
    """Detect if a virtual environment exists and if it's active.

    Returns:
        Tuple of (has_venv, venv_path, is_active)
    """
    # Common venv directory names
    venv_names = [".venv", "venv", ".env", "env", ".virtualenv", "virtualenv"]

    venv_path = None
    for name in venv_names:
        candidate = root / name
        # Check if it's a valid venv (has pyvenv.cfg or bin/activate)
        if candidate.exists() and candidate.is_dir():
            if (candidate / "pyvenv.cfg").exists() or (candidate / "bin" / "activate").exists():
                venv_path = name
                break
            # Windows style
            if (candidate / "Scripts" / "activate").exists():
                venv_path = name
                break

    if not venv_path:
        return False, None, False

    # Check if this venv is currently active
    is_active = False
    virtual_env = os.environ.get("VIRTUAL_ENV", "")
    if virtual_env:
        active_path = Path(virtual_env).resolve()
        candidate_path = (root / venv_path).resolve()
        is_active = active_path == candidate_path

    return True, venv_path, is_active


def infer_commands(
    project_type: ProjectType,
    root: Path,
    package_manager: PackageManager = "unknown",
    test_framework: TestFramework = "unknown",
) -> dict[str, str | None]:
    """Infer test/build/run commands based on project type, package manager, and test framework."""
    commands: dict[str, str | None] = {
        "test": None,
        "build": None,
        "run": None,
    }

    if project_type == "python":
        # Test command based on framework and package manager
        if test_framework == "pytest":
            if package_manager == "poetry":
                commands["test"] = "poetry run pytest"
            elif package_manager == "uv":
                commands["test"] = "uv run pytest"
            elif package_manager == "pipenv":
                commands["test"] = "pipenv run pytest"
            else:
                commands["test"] = "pytest"
        else:
            commands["test"] = "python -m unittest discover"

        # Build command based on package manager
        if package_manager == "poetry":
            commands["build"] = "poetry build"
        elif package_manager == "uv":
            commands["build"] = "uv build"
        else:
            commands["build"] = "python -m build"

        # Run command
        pyproject = root / "pyproject.toml"
        if pyproject.exists():
            content = read_file_safe(pyproject) or ""
            if "[project.scripts]" in content:
                if package_manager == "poetry":
                    commands["run"] = "poetry run <script-name>"
                elif package_manager == "uv":
                    commands["run"] = "uv run <script-name>"
                else:
                    commands["run"] = "python -m <module>"
            elif "[tool.poetry.scripts]" in content:
                commands["run"] = "poetry run <script-name>"

    elif project_type == "node":
        # Use detected package manager
        if package_manager == "yarn":
            commands["test"] = "yarn test"
            commands["build"] = "yarn build"
            commands["run"] = "yarn start"
        elif package_manager == "pnpm":
            commands["test"] = "pnpm test"
            commands["build"] = "pnpm build"
            commands["run"] = "pnpm start"
        else:  # npm
            commands["test"] = "npm test"
            commands["build"] = "npm run build"
            commands["run"] = "npm start"

    elif project_type == "rust":
        commands["test"] = "cargo test"
        commands["build"] = "cargo build"
        commands["run"] = "cargo run"

    elif project_type == "go":
        commands["test"] = "go test ./..."
        commands["build"] = "go build"
        commands["run"] = "go run ."

    return commands


def detect_project(root: Path | str | None = None) -> ProjectContext:
    """Detect project type and gather context.

    Args:
        root: Project root directory. Defaults to cwd.

    Returns:
        ProjectContext with gathered information.
    """
    if root is None:
        root = Path.cwd()
    elif isinstance(root, str):
        root = Path(root)

    root = root.resolve()

    # Detect type
    project_type, config_file = detect_project_type(root)

    # Read config file
    config_content = None
    if config_file:
        config_content = read_file_safe(root / config_file)

    # Detect package manager
    package_manager = detect_package_manager(project_type, root)

    # Detect test framework
    test_framework = detect_test_framework(project_type, root, config_content)

    # Detect virtual environment (Python only)
    has_venv, venv_path, is_venv_active = False, None, False
    if project_type == "python":
        has_venv, venv_path, is_venv_active = detect_virtual_environment(root)

    # Read README
    readme_content = None
    for readme_name in ["README.md", "README", "readme.md", "README.txt"]:
        readme_content = read_file_safe(root / readme_name)
        if readme_content:
            break

    # Get structure
    structure = get_directory_structure(root)

    # Infer commands based on detected tools
    commands = infer_commands(project_type, root, package_manager, test_framework)

    return ProjectContext(
        root=root,
        project_type=project_type,
        config_file=config_file,
        config_content=config_content,
        readme_content=readme_content,
        structure=structure,
        test_command=commands["test"],
        build_command=commands["build"],
        run_command=commands["run"],
        package_manager=package_manager,
        test_framework=test_framework,
        has_venv=has_venv,
        venv_path=venv_path,
        is_venv_active=is_venv_active,
    )
