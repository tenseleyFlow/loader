"""Runtime-owned safeguard helpers and combined safeguard policy."""

import re
from dataclasses import dataclass, field

from .safeguard_services import (
    ActionTracker,
    PreActionValidator,
    ValidationResult,
)


@dataclass
class FilterResult:
    """Result of filtering content."""
    content: str  # Filtered content
    was_filtered: bool  # Whether any filtering occurred
    removed_blocks: list[str] = field(default_factory=list)  # What was removed


class CodeBlockFilter:
    """Filters markdown code blocks and bracket tool calls from streamed content.

    Handles both complete blocks (```...```) and partial blocks that span
    multiple stream chunks. Also filters [calls X tool with ...] patterns.

    Candidate for removal once the typed runtime makes tool-call leakage
    structurally impossible.
    """

    def __init__(self):
        self._buffer = ""
        self._in_code_block = False
        self._block_lang = ""
        self._current_block = ""
        self._in_bracket = False
        self._bracket_content = ""
        self._in_json_tool = False
        self._json_brace_count = 0

    def reset(self):
        """Reset filter state."""
        self._buffer = ""
        self._in_code_block = False
        self._block_lang = ""
        self._current_block = ""
        self._in_bracket = False
        self._bracket_content = ""
        self._in_json_tool = False
        self._json_brace_count = 0

    def _is_bracket_tool_start(self, text: str) -> bool:
        """Check if text looks like start of a bracket tool call."""
        # Patterns like: [calls, [call, [USE
        return bool(re.match(r'\[(?:calls?|USE)\s', text, re.IGNORECASE))

    def filter_chunk(self, chunk: str) -> FilterResult:
        """Filter a streaming chunk, removing code blocks and bracket tool calls.

        Returns filtered content. Handles partial blocks across chunks.
        """
        if not chunk:
            return FilterResult(content="", was_filtered=False)

        result_parts = []
        removed = []
        was_filtered = False

        # Process character by character to handle streaming
        self._buffer += chunk

        while self._buffer:
            # Handle bracket tool calls: [calls X tool with ...]
            if self._in_bracket:
                # Look for closing ]
                end_idx = self._buffer.find(']')
                if end_idx >= 0:
                    self._bracket_content += self._buffer[:end_idx]
                    removed.append(f"[{self._bracket_content}]")
                    self._buffer = self._buffer[end_idx + 1:]
                    self._in_bracket = False
                    self._bracket_content = ""
                    was_filtered = True
                else:
                    # Still in bracket, consume all
                    self._bracket_content += self._buffer
                    self._buffer = ""
                    was_filtered = True
                continue

            # Check for bracket start: [calls, [USE, or [output (fake outputs)
            bracket_match = re.search(r'\[(?=(?:calls?|USE|output)\s*[:\s])', self._buffer, re.IGNORECASE)
            if bracket_match:
                # Output everything before the bracket
                result_parts.append(self._buffer[:bracket_match.start()])
                self._buffer = self._buffer[bracket_match.start() + 1:]  # Skip the [
                self._in_bracket = True
                was_filtered = True
                continue

            # Handle JSON tool calls: {"name": "write", "arguments": {...}}
            if self._in_json_tool:
                # Track braces to find the end
                for i, char in enumerate(self._buffer):
                    if char == '{':
                        self._json_brace_count += 1
                    elif char == '}':
                        self._json_brace_count -= 1
                        if self._json_brace_count == 0:
                            # Found end of JSON
                            removed.append(self._buffer[:i + 1])
                            self._buffer = self._buffer[i + 1:]
                            self._in_json_tool = False
                            was_filtered = True
                            break
                else:
                    # Still in JSON, consume all
                    self._buffer = ""
                    was_filtered = True
                continue

            # Check for JSON tool call start: {"name": "write" etc
            json_tool_match = re.search(
                r'\{\s*"name"\s*:\s*"(?:write|read|edit|bash|glob|grep)"',
                self._buffer
            )
            if json_tool_match:
                # Output everything before the JSON
                result_parts.append(self._buffer[:json_tool_match.start()])
                self._buffer = self._buffer[json_tool_match.start():]
                self._in_json_tool = True
                self._json_brace_count = 0  # Will count starting from {
                was_filtered = True
                continue

            # Check for hallucinated tool narration and filter the line
            hallucination_match = re.search(
                r'([Uu]sed\s+`?(?:bash|write|read|edit|glob|grep)`?\s+tool|'
                r'[Uu]sing\s+the\s+`?(?:bash|write|read|edit|glob|grep)`?\s+tool|'
                r'with\s+file_path\s*=\s*[`\'"]|'
                r'with\s+command\s*[`\'"]|'
                r'[Hh]ere\s+is\s+what\s+[Ii]\s+did:)',
                self._buffer
            )
            if hallucination_match:
                # Find end of line and remove whole line
                line_start = self._buffer.rfind('\n', 0, hallucination_match.start()) + 1
                line_end = self._buffer.find('\n', hallucination_match.end())
                if line_end == -1:
                    # Line continues to end of buffer - wait for more
                    if line_start > 0:
                        result_parts.append(self._buffer[:line_start])
                    self._buffer = self._buffer[line_start:]
                    break
                else:
                    # Remove the whole line
                    result_parts.append(self._buffer[:line_start])
                    removed.append(self._buffer[line_start:line_end])
                    self._buffer = self._buffer[line_end:]
                    was_filtered = True
                    continue

            # Check for preamble patterns and filter the line
            preamble_match = re.search(
                r'(Here is a JSON response|Here are the function calls|'
                r'Here is the response with|I will respond with|'
                r'The following JSON|Below is the)',
                self._buffer, re.IGNORECASE
            )
            if preamble_match:
                # Find end of line and remove whole line
                line_start = self._buffer.rfind('\n', 0, preamble_match.start()) + 1
                line_end = self._buffer.find('\n', preamble_match.end())
                if line_end == -1:
                    # Line continues to end of buffer - wait for more
                    if line_start > 0:
                        result_parts.append(self._buffer[:line_start])
                    self._buffer = self._buffer[line_start:]
                    break
                else:
                    # Remove the whole line
                    result_parts.append(self._buffer[:line_start])
                    removed.append(self._buffer[line_start:line_end])
                    self._buffer = self._buffer[line_end:]
                    was_filtered = True
                    continue
            if self._in_code_block:
                # Look for closing ```
                end_match = re.search(r'```', self._buffer)
                if end_match:
                    # Found end of code block
                    block_content = self._buffer[:end_match.start()]
                    self._current_block += block_content
                    removed.append(f"```{self._block_lang}\n{self._current_block}```")
                    self._buffer = self._buffer[end_match.end():]
                    self._in_code_block = False
                    self._current_block = ""
                    self._block_lang = ""
                    was_filtered = True
                else:
                    # Still in code block, consume all
                    self._current_block += self._buffer
                    self._buffer = ""
                    was_filtered = True
            else:
                # Look for opening ```
                start_match = re.search(r'```(\w*)\n?', self._buffer)
                if start_match:
                    # Found start of code block
                    # Output everything before the block
                    result_parts.append(self._buffer[:start_match.start()])
                    self._block_lang = start_match.group(1)
                    self._buffer = self._buffer[start_match.end():]
                    self._in_code_block = True
                    was_filtered = True
                else:
                    # Check if buffer ends with partial ``` marker
                    if self._buffer.endswith('`') or self._buffer.endswith('``'):
                        # Hold back potential partial marker
                        split_point = len(self._buffer) - self._buffer[::-1].index('`') - 1
                        if split_point > 0:
                            # Find where backticks start
                            for i in range(len(self._buffer) - 1, -1, -1):
                                if self._buffer[i] != '`':
                                    result_parts.append(self._buffer[:i+1])
                                    self._buffer = self._buffer[i+1:]
                                    break
                        break
                    else:
                        # No code block markers, output all
                        result_parts.append(self._buffer)
                        self._buffer = ""

        return FilterResult(
            content="".join(result_parts),
            was_filtered=was_filtered,
            removed_blocks=removed,
        )

    def filter_complete(self, content: str) -> FilterResult:
        """Filter complete content (non-streaming), removing code blocks, bracket tool calls, and preambles."""
        removed = []

        # Pattern to match code blocks
        code_pattern = r'```\w*\n?[\s\S]*?```'
        removed.extend(re.findall(code_pattern, content))
        filtered = re.sub(code_pattern, '', content)

        # Pattern to match bracket-format tool calls: [calls X tool with ...] and fake outputs
        bracket_patterns = [
            r'\[calls?\s+\w+\s+tool\s+with[:\s][^\]]+\]',
            r'\[USE\s+\w+\s+tool[:\s][^\]]+\]',
            r'\[output[:\s][^\]]+\]',  # Fake outputs from model
        ]
        for pattern in bracket_patterns:
            matches = re.findall(pattern, filtered, re.IGNORECASE)
            removed.extend(matches)
            filtered = re.sub(pattern, '', filtered, flags=re.IGNORECASE)

        # Pattern to match JSON tool calls: {"name": "write", "arguments": {...}}
        # Use a function to handle nested braces properly
        def remove_json_tool_calls(text: str) -> tuple[str, list[str]]:
            json_removed = []
            tool_pattern = r'\{\s*"name"\s*:\s*"(?:write|read|edit|bash|glob|grep)"'
            result = text
            while True:
                match = re.search(tool_pattern, result)
                if not match:
                    break
                # Find matching closing brace
                start = match.start()
                brace_count = 0
                end = start
                for i, char in enumerate(result[start:], start):
                    if char == '{':
                        brace_count += 1
                    elif char == '}':
                        brace_count -= 1
                        if brace_count == 0:
                            end = i + 1
                            break
                if end > start:
                    json_removed.append(result[start:end])
                    result = result[:start] + result[end:]
                else:
                    break  # Couldn't find matching brace
            return result, json_removed

        filtered, json_matches = remove_json_tool_calls(filtered)
        removed.extend(json_matches)

        # Pattern to match preamble lines (remove entire line)
        preamble_patterns = [
            r'^.*Here is a JSON response.*$',
            r'^.*Here are the function calls.*$',
            r'^.*Here is the response with.*$',
            r'^.*I will respond with.*$',
            r'^.*The following (JSON|function calls|tool calls).*$',
            r'^.*Below (is|are) the (JSON|function|tool).*$',
        ]
        for pattern in preamble_patterns:
            matches = re.findall(pattern, filtered, re.IGNORECASE | re.MULTILINE)
            removed.extend(matches)
            filtered = re.sub(pattern, '', filtered, flags=re.IGNORECASE | re.MULTILINE)

        # Pattern to match hallucinated/narrated tool uses (remove entire line)
        # These are lines where model describes using tools instead of actually calling them
        hallucination_patterns = [
            r'^.*[Uu]sed\s+`?(?:bash|write|read|edit|glob|grep)`?\s+tool.*$',  # "Used bash tool..."
            r'^.*[Uu]sing\s+the\s+`?(?:bash|write|read|edit|glob|grep)`?\s+tool.*$',  # "...using the write tool"
            r'^.*with\s+file_path\s*=\s*[`\'"][^`\'"]+[`\'"].*$',  # Narrated file_path parameter
            r'^.*with\s+command\s*[`\'"][^`\'"]+[`\'"].*$',  # Narrated bash command
            r'^\s*\*\s*[Uu]sed\s+`.*$',  # "* Used `bash`..." (bullet point narration)
            r'^.*[Hh]ere\s+is\s+what\s+[Ii]\s+did:.*$',  # "Here is what I did:"
            r'^\s*\d+\.\s+[Uu]sed\s+.*tool.*$',  # "1. Used bash tool..."
            r'^\s*\d+\.\s+[Cc]reated\s+.*using\s+the\s+.*tool.*$',  # "1. Created... using the write tool"
        ]
        for pattern in hallucination_patterns:
            matches = re.findall(pattern, filtered, re.MULTILINE)
            removed.extend(matches)
            filtered = re.sub(pattern, '', filtered, flags=re.MULTILINE)

        # Filter internal recovery/system prompts (multiline blocks)
        internal_prompt_patterns = [
            # Recovery prompts
            r'## TOOL FAILURE - INVESTIGATE AND ADAPT[\s\S]*?What will you do\?',
            r'## REQUIRED: Choose ONE[\s\S]*?(?=\n\n|\Z)',
            r'## CRITICAL RULES:[\s\S]*?(?=\n\n|\Z)',
            r'## Current attempt:.*$',
            r'\*\*Your next action should gather information[\s\S]*?What will you do\?',
            # Observation prefixes
            r'^Observation \[[\w]+\]:.*$',
        ]
        for pattern in internal_prompt_patterns:
            matches = re.findall(pattern, filtered, re.MULTILINE)
            removed.extend(matches)
            filtered = re.sub(pattern, '', filtered, flags=re.MULTILINE)

        # Clean up multiple blank lines left behind
        filtered = re.sub(r'\n{3,}', '\n\n', filtered)

        return FilterResult(
            content=filtered.strip(),
            was_filtered=bool(removed),
            removed_blocks=removed,
        )


@dataclass
class PatternMatch:
    """A detected problematic pattern."""
    pattern_type: str  # 'code_block', 'narration', 'preview', 'repetition'
    match_text: str
    severity: str  # 'low', 'medium', 'high'


class PatternDetector:
    """Detects problematic patterns in agent output.

    Patterns include:
    - Code blocks (which should be tool calls instead)
    - Narration ("I will call...", "Now I'll...")
    - Previews ("The file will look like:", "After editing:")
    - Repetitive commands
    """

    # Narration patterns - model announcing what it will do instead of doing it
    NARRATION_PATTERNS = [
        (r"I('ll| will) (use|call|execute|run) the (\w+) tool", "narration", "high"),
        (r"Let me (use|call|execute|run) the (\w+) tool", "narration", "high"),
        (r"Now I('ll| will) (create|write|edit|run|execute)", "narration", "medium"),
        (r"I('m going to| am going to) (use|call|create|write)", "narration", "medium"),
        (r"First,? I('ll| will) (use|call|create)", "narration", "medium"),
        (r"Next,? I('ll| will) (use|call|create)", "narration", "medium"),
    ]

    # Preview patterns - model showing content instead of using tools
    PREVIEW_PATTERNS = [
        (r"(The|This) file will (look like|contain|have):", "preview", "high"),
        (r"After editing,? (the file|it) will (look like|contain):", "preview", "high"),
        (r"Here('s| is) (the|what) (content|code|file):", "preview", "high"),
        (r"Save this (to|as|in) [\w./]+:", "preview", "high"),
        (r"Create a file (with|containing):", "preview", "medium"),
        (r"(The|Your) [\w./]+ (should|will) (look like|contain):", "preview", "medium"),
    ]

    # Preamble patterns - model describing JSON/function calls instead of using them
    PREAMBLE_PATTERNS = [
        (r"Here is a JSON response", "preamble", "high"),
        (r"Here are the function calls", "preamble", "high"),
        (r"Here is the response with", "preamble", "high"),
        (r"I will respond with", "preamble", "high"),
        (r"The following (JSON|function calls|tool calls)", "preamble", "high"),
        (r"Below (is|are) the (JSON|function|tool)", "preamble", "high"),
    ]

    # Code block patterns
    CODE_BLOCK_PATTERNS = [
        (r'```\w+\n', "code_block", "high"),
        (r'```\n', "code_block", "medium"),
    ]

    def __init__(self):
        self._all_patterns = (
            self.NARRATION_PATTERNS +
            self.PREVIEW_PATTERNS +
            self.PREAMBLE_PATTERNS +
            self.CODE_BLOCK_PATTERNS
        )
        self._recent_detections: list[PatternMatch] = []

    def reset(self):
        """Reset detection state."""
        self._recent_detections = []

    def detect(self, content: str) -> list[PatternMatch]:
        """Detect problematic patterns in content."""
        matches = []

        for pattern, ptype, severity in self._all_patterns:
            for match in re.finditer(pattern, content, re.IGNORECASE):
                matches.append(PatternMatch(
                    pattern_type=ptype,
                    match_text=match.group(0),
                    severity=severity,
                ))

        self._recent_detections.extend(matches)
        return matches

    def has_high_severity(self, content: str) -> bool:
        """Check if content has high-severity patterns."""
        matches = self.detect(content)
        return any(m.severity == "high" for m in matches)

    def get_steering_message(self, matches: list[PatternMatch]) -> str | None:
        """Generate a steering message based on detected patterns.

        Returns None if no steering needed.
        """
        if not matches:
            return None

        # Prioritize high severity
        high_severity = [m for m in matches if m.severity == "high"]
        if not high_severity:
            return None

        # Generate appropriate steering message
        pattern_types = set(m.pattern_type for m in high_severity)

        if "preamble" in pattern_types:
            return (
                "[STOP] Do not describe JSON or function calls. "
                "Just USE the tools directly. No preambles."
            )
        elif "code_block" in pattern_types or "preview" in pattern_types:
            return (
                "[REMINDER] Do not show code blocks or previews. "
                "Use tools directly to create/edit files. "
                "No ```code```, just call the tool."
            )
        elif "narration" in pattern_types:
            return (
                "[REMINDER] Don't announce tool calls. "
                "Just use the tool directly without narration."
            )

        return None


class RuntimeSafeguards:
    """Combined runtime safeguards for the agent.

    Usage:
        safeguards = RuntimeSafeguards()

        # For streaming:
        filtered = safeguards.filter_stream_chunk(chunk)
        if safeguards.should_steer():
            steering_msg = safeguards.get_steering_message()

        # Before tool execution:
        is_dup, reason = safeguards.check_duplicate(tool_name, args)
        if is_dup:
            skip this tool call

        # Pre-action validation:
        validation = safeguards.validate_action(tool_name, args)
        if not validation.valid:
            skip or warn

        # After tool execution:
        safeguards.record_action(tool_name, args)
    """

    def __init__(self):
        self.code_filter = CodeBlockFilter()
        self.pattern_detector = PatternDetector()
        self.action_tracker = ActionTracker()
        self.validator = PreActionValidator()
        self._pending_steering: str | None = None
        self._accumulated_content = ""

    def reset(self):
        """Reset all safeguards for a new conversation."""
        self.code_filter.reset()
        self.pattern_detector.reset()
        self.action_tracker.reset()
        self._pending_steering = None
        self._accumulated_content = ""

    def filter_stream_chunk(self, chunk: str) -> str:
        """Filter a streaming chunk, removing code blocks.

        Also detects patterns for potential steering.
        """
        # Filter code blocks
        result = self.code_filter.filter_chunk(chunk)

        # Accumulate for pattern detection
        self._accumulated_content += chunk

        # Check for patterns periodically (every 200 chars)
        if len(self._accumulated_content) > 200:
            matches = self.pattern_detector.detect(self._accumulated_content)
            if matches:
                steering = self.pattern_detector.get_steering_message(matches)
                if steering:
                    self._pending_steering = steering
            self._accumulated_content = self._accumulated_content[-100:]  # Keep last 100 chars for context

        return result.content

    def filter_complete_content(self, content: str) -> str:
        """Filter complete content (non-streaming)."""
        result = self.code_filter.filter_complete(content)

        # Also detect patterns
        matches = self.pattern_detector.detect(content)
        if matches:
            steering = self.pattern_detector.get_steering_message(matches)
            if steering:
                self._pending_steering = steering

        return result.content

    def should_steer(self) -> bool:
        """Check if we should inject a steering message."""
        return self._pending_steering is not None

    def get_steering_message(self) -> str | None:
        """Get pending steering message and clear it."""
        msg = self._pending_steering
        self._pending_steering = None
        return msg

    def check_duplicate(self, tool_name: str, arguments: dict) -> tuple[bool, str]:
        """Check if a tool call would be a duplicate."""
        return self.action_tracker.check_tool_call(tool_name, arguments)

    def record_action(self, tool_name: str, arguments: dict) -> None:
        """Record a completed tool action."""
        self.action_tracker.record_tool_call(tool_name, arguments)

    def detect_loop(self) -> tuple[bool, str]:
        """Detect if the agent is in a repetitive loop.

        Returns (is_loop, pattern_description).
        """
        return self.action_tracker.detect_loop()

    def validate_action(self, tool_name: str, arguments: dict) -> ValidationResult:
        """Validate a tool action before execution.

        Returns ValidationResult with validity and any warnings/errors.
        """
        return self.validator.validate(tool_name, arguments)

    def record_response(self, response: str) -> None:
        """Record a response for text loop detection."""
        self.action_tracker.record_response(response)

    def detect_text_loop(self, response: str) -> tuple[bool, str]:
        """Detect if the agent is repeating the same response.

        Returns (is_loop, description).
        """
        return self.action_tracker.detect_text_loop(response)
