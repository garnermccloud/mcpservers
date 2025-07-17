"""MCP server for Gemini using gemini-cli."""

import argparse
import asyncio
import json
import os
import pathlib
import subprocess
import sys
from typing import List, Optional

from mcp.server.fastmcp import FastMCP


# Truthfulness instruction for all Gemini interactions
TRUTHFULNESS_INSTRUCTION = """You are an AI assistant committed to truthfulness above agreeability. Your primary responsibilities: 
  - Prioritize factual accuracy over user agreement 
  - Actively correct errors or misconceptions in the user's statements 
  - Express uncertainty when you lack confidence 
  - Be willing to respectfully disagree when users make incorrect claims 

  Never agree with false statements just to be pleasant. Your value comes from being accurate, not agreeable. However, don't be contrarian for the sake of being a contrarian."""


async def run_gemini_cli(prompt: str) -> str:
    """Run gemini-cli with the given prompt."""

    # Include truthfulness instruction
    full_prompt = f"{TRUTHFULNESS_INSTRUCTION}\n\n{prompt}"

    try:
        # Use asyncio.create_subprocess_exec for async execution
        result = await asyncio.create_subprocess_exec(
            "gemini",
            "-p",
            full_prompt,
            cwd=os.path.abspath("."),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        stdout, stderr = await result.communicate()

        if result.returncode != 0:
            return f"Error running gemini-cli: {stderr.decode()}"

        return stdout.decode()
    except Exception as e:
        return f"Error: {str(e)}"


def check_gemini_cli_available() -> bool:
    """Check if gemini-cli is available in PATH."""
    try:
        result = subprocess.run(["which", "gemini"], capture_output=True, text=True)
        return result.returncode == 0
    except Exception:
        return False


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="MCP server for Gemini using gemini-cli"
    )
    parser.add_argument(
        "--working-directory",
        type=str,
        help="Working directory for file operations (default: current directory)",
    )
    return parser.parse_args()


# Create MCP server instance
mcp_server = FastMCP(
    "Gemini CLI MCP",
    description="Model Context Protocol server for Gemini using gemini-cli",
    dependencies=["mcp"],
    debug=True,
)


@mcp_server.tool()
async def generate_text(
    prompt: str,
    system_instruction: Optional[str] = None,
    custom_message: Optional[str] = None,
    context_files: List[str] = [],
    output_file: Optional[str] = None,
) -> str:
    """Generate text using Gemini CLI.

    Args:
        prompt: The user prompt to send to Gemini.
        system_instruction: Optional system instructions to guide the model.
        custom_message: Optional custom message providing additional context or instructions.
        context_files: List of file paths to provide additional context.
        output_file: Optional file path to write the response to.
    Returns:
        Generated text response.
    """
    try:
        # Build the complete prompt
        prompt_parts = [prompt]

        if system_instruction:
            prompt_parts.insert(0, f"System: {system_instruction}")

        if custom_message:
            prompt_parts.insert(-1, f"Additional context: {custom_message}")

        # Add file references using @ syntax
        if context_files:
            file_refs = " ".join(f"@{f}" for f in context_files)
            prompt_parts.append(f"Files for context: {file_refs}")

        complete_prompt = "\n\n".join(prompt_parts)

        # Call gemini-cli
        response = await run_gemini_cli(complete_prompt)

        # Handle output file
        if output_file and response and not response.startswith("Error"):
            result = write_to_file(output_file, response)
            if result.startswith("Error"):
                return f"{result}\n\nResponse:\n{response}"
            return f"Response written to {output_file}"

        return response

    except Exception as e:
        return f"Error generating content: {str(e)}"


@mcp_server.tool()
async def search_and_analyze_current_info(
    question: str,
    custom_message: Optional[str] = None,
    output_file: Optional[str] = None,
) -> str:
    """Search for current information and provide an AI-analyzed response.

    This tool gives Gemini the ability to search the web for real-time information
    and provide a comprehensive, fact-based response.

    Args:
        question: The question or request you want Gemini to answer using current web data.
        custom_message: Optional custom message providing additional context or instructions.
        output_file: Optional file path to write the response to.

    Returns:
        AI-generated response based on current web search results.
    """
    try:
        # Build prompt with web search instruction
        prompt_parts = [
            "Use the google_web_search tool when you need current information.",
            question,
        ]

        if custom_message:
            prompt_parts.insert(-1, f"Additional context: {custom_message}")

        complete_prompt = "\n\n".join(prompt_parts)

        # Call gemini-cli
        response = await run_gemini_cli(complete_prompt)

        # Handle output file
        if output_file and response and not response.startswith("Error"):
            result = write_to_file(output_file, response)
            if result.startswith("Error"):
                return f"{result}\n\nResponse:\n{response}"
            return f"Response written to {output_file}"

        return response

    except Exception as e:
        return f"Error performing web search: {str(e)}"


@mcp_server.tool()
async def review_code(
    files: List[str],
    review_focus: str = "all",
    custom_message: Optional[str] = None,
    output_file: Optional[str] = None,
) -> str:
    """Review files for quality, security, performance, or style issues.

    Args:
        files: List of file paths to review.
        review_focus: Focus area for the review (all, quality, security, performance, style).
        custom_message: Additional context for the review.
        output_file: Optional file path to write the review to.

    Returns:
        Code review results.
    """
    try:
        # Build review prompt
        focus_instructions = {
            "all": "all aspects including code quality, security, performance, and style",
            "quality": "code quality including maintainability, readability, and testability",
            "security": "security vulnerabilities, input validation, authentication, and data handling",
            "performance": "performance optimizations, inefficient algorithms, and resource usage",
            "style": "code style, naming conventions, and adherence to best practices",
        }.get(review_focus, "all aspects of the code")

        prompt_parts = [
            f"Perform a detailed code review focusing on {focus_instructions}.",
            "Your review should include:",
            "1. Summary of findings",
            "2. Issues organized by severity (Critical/High/Medium/Low) with line numbers",
            "3. Concrete recommendations for each issue",
            "4. Overall assessment",
            "Be direct and thorough.",
        ]

        if custom_message:
            prompt_parts.insert(-1, f"Additional instructions: {custom_message}")

        # Add file references
        file_refs = " ".join(f"@{f}" for f in files)
        prompt_parts.append(f"Files to review: {file_refs}")

        complete_prompt = "\n\n".join(prompt_parts)

        # Call gemini-cli
        response = await run_gemini_cli(complete_prompt)

        # Handle output file
        if output_file and response and not response.startswith("Error"):
            result = write_to_file(output_file, response)
            if result.startswith("Error"):
                return f"{result}\n\nResponse:\n{response}"
            return f"Code review written to {output_file}"

        return response

    except Exception as e:
        return f"Error performing code review: {str(e)}"


@mcp_server.tool()
async def analyze_repo_structure(
    repo_path: str,
    custom_message: Optional[str] = None,
    output_file: Optional[str] = None,
) -> str:
    """Analyze repository structure and organization.

    Args:
        repo_path: Path to the repository root directory (relative to working directory).
        custom_message: Optional custom message providing additional context or instructions.
        output_file: Optional file path to write the analysis to.

    Returns:
        Repository structure analysis.
    """
    try:
        prompt_parts = [
            "Analyze the repository structure and provide insights about its organization, architecture, and code patterns.",
            "Your analysis should include:",
            "1. Identified project type and main programming languages",
            "2. Key architectural patterns visible from the structure",
            "3. Code organization (modules, packages, namespacing)",
            "4. Build and dependency management approach",
            "5. Notable observations about code organization",
            "6. Recommendations for structural improvements",
        ]

        if custom_message:
            prompt_parts.insert(-1, f"Additional context: {custom_message}")

        prompt_parts.append(f"Repository to analyze: @{repo_path}")

        complete_prompt = "\n\n".join(prompt_parts)

        # Call gemini-cli
        response = await run_gemini_cli(complete_prompt)

        # Handle output file
        if output_file and response and not response.startswith("Error"):
            result = write_to_file(output_file, response)
            if result.startswith("Error"):
                return f"{result}\n\nResponse:\n{response}"
            return f"Repository structure analysis written to {output_file}"

        return response

    except Exception as e:
        return f"Error analyzing repository structure: {str(e)}"


@mcp_server.tool()
async def identify_patterns(
    code_files: List[str],
    pattern_type: str = "coding",
    custom_message: Optional[str] = None,
    output_file: Optional[str] = None,
) -> str:
    """Identify patterns in code files.

    Args:
        code_files: List of file paths to analyze.
        pattern_type: Type of patterns to identify (coding, architecture, testing, error-handling).
        custom_message: Optional custom message providing additional context or instructions.
        output_file: Optional file path to write the analysis to.

    Returns:
        Identified patterns in the provided code files.
    """
    try:
        # Determine pattern focus
        pattern_focus = {
            "coding": "code style, naming conventions, and common idioms",
            "architecture": "architectural patterns, component structure, and dependency flow",
            "testing": "testing approaches, test coverage, and test organization",
            "error-handling": "error handling strategies, exception patterns, and failure modes",
        }.get(pattern_type, "coding patterns and style")

        prompt_parts = [
            f"Identify and analyze patterns in the provided files focusing on {pattern_focus}.",
            "Your analysis should focus on:",
            "1. Consistent patterns across files",
            "2. Inconsistencies or variations",
            "3. Code style and naming conventions",
            "4. Architectural patterns and component structure",
            "5. Testing approaches",
            "6. Error handling strategies",
            "Provide specific examples from the code.",
        ]

        if custom_message:
            prompt_parts.insert(-1, f"Additional context: {custom_message}")

        # Add file references
        file_refs = " ".join(f"@{f}" for f in code_files)
        prompt_parts.append(f"Files to analyze: {file_refs}")

        complete_prompt = "\n\n".join(prompt_parts)

        # Call gemini-cli
        response = await run_gemini_cli(complete_prompt)

        # Handle output file
        if output_file and response and not response.startswith("Error"):
            result = write_to_file(output_file, response)
            if result.startswith("Error"):
                return f"{result}\n\nResponse:\n{response}"
            return f"Pattern analysis written to {output_file}"

        return response

    except Exception as e:
        return f"Error identifying patterns: {str(e)}"


@mcp_server.tool()
async def plan_implementation(
    requirements: str,
    context_files: List[str] = [],
    custom_message: Optional[str] = None,
    output_file: Optional[str] = None,
) -> str:
    """Generate implementation plan based on requirements and code context.

    Args:
        requirements: Description of what needs to be implemented.
        context_files: List of file paths to provide context.
        custom_message: Optional custom message providing additional context or instructions.
        output_file: Optional file path to write the plan to.

    Returns:
        Implementation plan with task breakdown.
    """
    try:
        prompt_parts = [
            "Create a detailed implementation plan for the following requirements.",
            f"Requirements: {requirements}",
            "Your implementation plan should include:",
            "1. Task Breakdown with complexity estimates (Low/Medium/High)",
            "2. Implementation approach and suggested order",
            "3. Recommended architecture/design patterns",
            "4. Key components and their responsibilities",
            "5. Technical considerations and potential challenges",
            "6. Testing approach",
            "Format as a structured document.",
        ]

        if custom_message:
            prompt_parts.insert(-1, f"Additional context: {custom_message}")

        # Add context files if provided
        if context_files:
            file_refs = " ".join(f"@{f}" for f in context_files)
            prompt_parts.append(f"Context files: {file_refs}")

        complete_prompt = "\n\n".join(prompt_parts)

        # Call gemini-cli
        response = await run_gemini_cli(complete_prompt)

        # Handle output file
        if output_file and response and not response.startswith("Error"):
            result = write_to_file(output_file, response)
            if result.startswith("Error"):
                return f"{result}\n\nResponse:\n{response}"
            return f"Implementation plan written to {output_file}"

        return response

    except Exception as e:
        return f"Error creating implementation plan: {str(e)}"


@mcp_server.tool()
async def design_architecture(
    requirements: str,
    context_files: List[str] = [],
    tech_stack: Optional[str] = None,
    custom_message: Optional[str] = None,
    output_file: Optional[str] = None,
) -> str:
    """Design software architecture based on requirements.

    Args:
        requirements: Functional and non-functional requirements.
        context_files: List of file paths to provide context from existing code.
        tech_stack: Optional technology stack constraints or preferences.
        custom_message: Optional custom message providing additional context or instructions.
        output_file: Optional file path to write the architecture design to.

    Returns:
        Architecture design document.
    """
    try:
        prompt_parts = [
            "Design a software architecture for the following requirements.",
            f"Requirements: {requirements}",
        ]

        if tech_stack:
            prompt_parts.append(f"Technology stack constraints: {tech_stack}")

        if custom_message:
            prompt_parts.append(f"Additional context: {custom_message}")

        prompt_parts.extend(
            [
                "Your architecture design should include:",
                "1. System Overview with high-level architecture",
                "2. Component Design with detailed descriptions and interfaces",
                "3. Technology choices with justifications",
                "4. Data models and schemas",
                "5. Scalability approach",
                "6. Security considerations",
                "7. Performance optimizations",
                "8. Deployment strategy",
                "Format as a structured architecture document.",
            ]
        )

        # Add context files if provided
        if context_files:
            file_refs = " ".join(f"@{f}" for f in context_files)
            prompt_parts.append(f"Context files: {file_refs}")

        complete_prompt = "\n\n".join(prompt_parts)

        # Call gemini-cli
        response = await run_gemini_cli(complete_prompt)

        # Handle output file
        if output_file and response and not response.startswith("Error"):
            result = write_to_file(output_file, response)
            if result.startswith("Error"):
                return f"{result}\n\nResponse:\n{response}"
            return f"Architecture design written to {output_file}"

        return response

    except Exception as e:
        return f"Error creating architecture design: {str(e)}"


@mcp_server.tool()
async def generate_unit_tests(
    code_files: List[str],
    testing_framework: str = "pytest",
    custom_message: Optional[str] = None,
    output_file: Optional[str] = None,
) -> str:
    """Generate unit tests for the provided code files.

    Args:
        code_files: Files to generate tests for.
        testing_framework: Test framework to target.
        custom_message: Additional instructions for test generation.
        output_file: Optional file path to write the tests to.

    Returns:
        Generated unit tests.
    """
    try:
        prompt_parts = [
            f"Generate comprehensive unit tests using {testing_framework} for the following code.",
            "Your tests should include:",
            "1. Test cases for all public functions/methods",
            "2. Edge cases and error conditions",
            "3. Mock objects where needed for external dependencies",
            "4. Clear assertions with descriptive failure messages",
            "5. Test documentation explaining what each test verifies",
            "6. Setup/teardown if required",
            "Use the appropriate testing framework based on the language and existing patterns.",
        ]

        if custom_message:
            prompt_parts.insert(-1, f"Additional instructions: {custom_message}")

        # Add file references
        file_refs = " ".join(f"@{f}" for f in code_files)
        prompt_parts.append(f"Files to test: {file_refs}")

        complete_prompt = "\n\n".join(prompt_parts)

        # Call gemini-cli
        response = await run_gemini_cli(complete_prompt)

        # Handle output file
        if output_file and response and not response.startswith("Error"):
            result = write_to_file(output_file, response)
            if result.startswith("Error"):
                return f"{result}\n\nResponse:\n{response}"
            return f"Unit tests written to {output_file}"

        return response

    except Exception as e:
        return f"Error generating unit tests: {str(e)}"


@mcp_server.tool()
async def explain_code(
    code_files: List[str],
    custom_message: Optional[str] = None,
    output_file: Optional[str] = None,
) -> str:
    """Explain the provided code files in plain language.

    Args:
        code_files: Files to explain.
        custom_message: Additional instructions for the explanation.
        output_file: Optional file path to write the explanation to.

    Returns:
        Explanation of the code.
    """
    try:
        prompt_parts = [
            "Explain the following code in detail.",
            "Your explanation should include:",
            "1. Overall purpose and functionality",
            "2. How each component/function works",
            "3. Data flow and interactions between components",
            "4. Key algorithms or patterns used",
            "5. Dependencies and external calls",
            "6. Potential gotchas or complex sections",
            "Use clear language suitable for developers learning the codebase.",
        ]

        if custom_message:
            prompt_parts.insert(-1, f"Additional focus: {custom_message}")

        # Add file references
        file_refs = " ".join(f"@{f}" for f in code_files)
        prompt_parts.append(f"Code to explain: {file_refs}")

        complete_prompt = "\n\n".join(prompt_parts)

        # Call gemini-cli
        response = await run_gemini_cli(complete_prompt)

        # Handle output file
        if output_file and response and not response.startswith("Error"):
            result = write_to_file(output_file, response)
            if result.startswith("Error"):
                return f"{result}\n\nResponse:\n{response}"
            return f"Code explanation written to {output_file}"

        return response

    except Exception as e:
        return f"Error explaining code: {str(e)}"


@mcp_server.tool()
async def summarize_files(
    files: List[str],
    custom_message: Optional[str] = None,
    output_file: Optional[str] = None,
) -> str:
    """Summarize the provided files.
    
    Args:
        files: List of file paths to summarize.
        custom_message: Optional custom message providing additional context.
        output_file: Optional file path to write the summary to.
        
    Returns:
        Summary of the files.
    """
    try:
        prompt_parts = [
            "Provide a concise summary of the following files.",
            "Your summary should include:",
            "1. Main purpose of each file",
            "2. Key information and highlights",
            "3. How files relate to each other (if multiple)",
            "4. Important details that shouldn't be missed",
            "Keep it brief but comprehensive.",
        ]

        if custom_message:
            prompt_parts.insert(-1, f"Additional context: {custom_message}")

        # Add file references
        file_refs = " ".join(f"@{f}" for f in files)
        prompt_parts.append(f"Files to summarize: {file_refs}")

        complete_prompt = "\n\n".join(prompt_parts)

        # Call gemini-cli
        response = await run_gemini_cli(complete_prompt)

        # Handle output file
        if output_file and response and not response.startswith("Error"):
            result = write_to_file(output_file, response)
            if result.startswith("Error"):
                return f"{result}\n\nResponse:\n{response}"
            return f"File summary written to {output_file}"

        return response

    except Exception as e:
        return f"Error summarizing files: {str(e)}"


@mcp_server.tool()
async def extract_todos(
    files: List[str],
    output_file: Optional[str] = None,
) -> str:
    """Extract TODO and FIXME comments from files.

    Args:
        files: List of file paths to search for TODOs.
        output_file: Optional file path to write the extracted TODOs to.

    Returns:
        List of TODOs found.
    """
    try:
        prompt_parts = [
            "Find all TODO and FIXME comments in the provided files.",
            "Your analysis should:",
            "1. List each TODO/FIXME with file location and line numbers",
            "2. Categorize by priority/urgency if indicated in the comments",
            "3. Group by type of task (bug fixes, features, refactoring, etc.)",
            "4. Estimate effort level based on context",
            "5. Suggest which TODOs should be prioritized based on their impact",
            "Format the output clearly with file paths and line numbers.",
        ]

        # Add file references
        file_refs = " ".join(f"@{f}" for f in files)
        prompt_parts.append(f"Files to search: {file_refs}")

        complete_prompt = "\n\n".join(prompt_parts)

        # Call gemini-cli
        response = await run_gemini_cli(complete_prompt)

        # Handle output file
        if output_file and response and not response.startswith("Error"):
            result = write_to_file(output_file, response)
            if result.startswith("Error"):
                return f"{result}\n\nResponse:\n{response}"
            return f"TODO extraction written to {output_file}"

        return response

    except Exception as e:
        return f"Error extracting TODOs: {str(e)}"


@mcp_server.tool()
async def cleanup_repo(
    repo_path: str,
    custom_message: Optional[str] = None,
    output_file: Optional[str] = None,
) -> str:
    """Suggest cleanup actions for the given repository.

    Args:
        repo_path: Path to the repository root directory.
        custom_message: Optional custom message providing additional context.
        output_file: Optional file path to write the cleanup suggestions to.

    Returns:
        Cleanup suggestions.
    """
    try:
        prompt_parts = [
            "Analyze the repository and suggest cleanup actions to reduce code bloat and technical debt.",
            "Your cleanup suggestions should include:",
            "1. Unused or dead code to remove (with specific file paths)",
            "2. Duplicate code that can be consolidated",
            "3. Overly complex code to simplify",
            "4. Outdated dependencies or patterns",
            "5. Large files that should be split",
            "6. Unnecessary files or directories",
            "7. Priority order for cleanup tasks",
            "Be specific with file paths and provide examples.",
        ]

        if custom_message:
            prompt_parts.insert(-1, f"Additional focus: {custom_message}")

        prompt_parts.append(f"Repository to analyze: @{repo_path}")

        complete_prompt = "\n\n".join(prompt_parts)

        # Call gemini-cli
        response = await run_gemini_cli(complete_prompt)

        # Handle output file
        if output_file and response and not response.startswith("Error"):
            result = write_to_file(output_file, response)
            if result.startswith("Error"):
                return f"{result}\n\nResponse:\n{response}"
            return f"Cleanup suggestions written to {output_file}"

        return response

    except Exception as e:
        return f"Error generating cleanup suggestions: {str(e)}"


@mcp_server.tool()
async def generate_changelog(
    repo_path: str,
    max_entries: int = 10,
    custom_message: Optional[str] = None,
    output_file: Optional[str] = None,
) -> str:
    """Generate a changelog summary from recent commits.

    Args:
        repo_path: Path to the git repository.
        max_entries: Number of commits to include.
        custom_message: Additional instructions for the summary.
        output_file: Optional file path to write the changelog to.

    Returns:
        AI-generated changelog summary.
    """
    try:
        # Get git log using subprocess
        result = subprocess.run(
            [
                "git",
                "-C",
                repo_path,
                "log",
                f"-n{max_entries}",
                "--pretty=%s",
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        commit_messages = result.stdout.strip().splitlines()

        # Build prompt
        prompt_parts = [
            "Create a well-organized changelog from the following commit messages.",
            "Format the changelog with sections like:",
            "- Features",
            "- Bug Fixes",
            "- Improvements",
            "- Performance",
            "- Documentation",
            "- Other Changes",
            "Make it suitable for release notes. Be concise but informative.",
        ]

        if custom_message:
            prompt_parts.insert(-1, f"Context: {custom_message}")

        prompt_parts.append(
            f"Commit messages:\n{json.dumps(commit_messages, indent=2)}"
        )

        complete_prompt = "\n\n".join(prompt_parts)

        # Call gemini-cli
        response = await run_gemini_cli(complete_prompt)

        # Handle output file
        if output_file and response and not response.startswith("Error"):
            result = write_to_file(output_file, response)
            if result.startswith("Error"):
                return f"{result}\n\nResponse:\n{response}"
            return f"Changelog written to {output_file}"

        return response

    except subprocess.CalledProcessError as e:
        return f"Error retrieving commit messages: {str(e)}"
    except Exception as e:
        return f"Error generating changelog: {str(e)}"


# Utility functions
def write_to_file(file_path: str, content: str) -> str:
    """Write content to a file safely with error handling."""
    try:
        path = pathlib.Path(file_path)
        # Create parent directories if they don't exist
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        return f"Successfully wrote to {file_path}"
    except Exception as e:
        return f"Error writing to {file_path}: {str(e)}"


def main():
    """Main entry point."""
    # Parse command-line arguments
    args = parse_args()

    # Change to working directory if specified
    if args.working_directory:
        os.chdir(args.working_directory)

    # Check if gemini-cli is available
    if not check_gemini_cli_available():
        print(
            "Error: gemini-cli not found in PATH. Please install it first.",
            file=sys.stderr,
        )
        sys.exit(1)

    print(
        f"Starting Gemini CLI MCP server in directory: {os.getcwd()}",
        file=sys.stderr,
    )
    print(
        f"DEBUG main: sys.argv={sys.argv}",
        file=sys.stderr,
    )

    # Run the MCP server
    mcp_server.run(transport="stdio")


if __name__ == "__main__":
    main()
