"""MCP server for Gemini API."""

import asyncio
import json
import os
import pathlib
import subprocess
from typing import Any, Dict, List, Optional, Tuple, Union, cast

from dotenv import load_dotenv
from google import genai  # type: ignore[reportMissingTypeStubs]
from google.genai.types import (
    GenerateContentConfig,
    GenerateContentConfigDict,
    GoogleSearch,
    Tool,
)
from mcp.server.fastmcp import FastMCP

# Load environment variables
load_dotenv()

# Create MCP server instance
mcp_server = FastMCP(
    "Gemini MCP",
    description="Model Context Protocol server for Google's Gemini API",
    dependencies=["google-genai", "python-dotenv"],
    debug=True,
)

# Get API key from environment
api_key = os.environ.get("GOOGLE_API_KEY")
use_vertex_ai = os.environ.get("GOOGLE_GENAI_USE_VERTEXAI", "false").lower() == "true"

if not api_key and not use_vertex_ai:
    raise ValueError(
        "GOOGLE_API_KEY environment variable not set. "
        "Please set it or create a .env file."
    )
if use_vertex_ai:
    client = genai.Client()
else:
    # Initialize Gemini client
    client = genai.Client(api_key=api_key)

# Get default configuration from environment
DEFAULT_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.5-pro")
DEFAULT_TEMPERATURE = float(os.environ.get("GEMINI_TEMPERATURE", "0.7"))
DEFAULT_MAX_TOKENS = (
    int(os.environ.get("GEMINI_MAX_TOKENS", "66535"))
    if "GEMINI_MAX_TOKENS" in os.environ
    else None
)
DEFAULT_TOP_P = float(os.environ.get("GEMINI_TOP_P", "0.95"))
DEFAULT_TOP_K = int(os.environ.get("GEMINI_TOP_K", "40"))

# System instruction avoiding flattery and promoting truthfulness
NO_FLATTERY_INSTRUCTION = (
    "Never start your response by saying a question or idea or observation was good, "
    "great, fascinating, profound, excellent, or any other positive adjective. "
    "Skip the flattery and respond directly. "
    "You are an AI assistant committed to truthfulness above agreeability. "
    "Your primary responsibilities:\n"
    "- Prioritize factual accuracy over user agreement\n"
    "- Actively correct errors or misconceptions in the user's statements\n"
    "- Express uncertainty when you lack confidence\n"
    "- Be willing to respectfully disagree when users make incorrect claims\n\n"
    "Never agree with false statements just to be pleasant. "
    "Your value comes from being accurate, not agreeable."
)


@mcp_server.resource("config://gemini")
def get_gemini_config() -> str:
    """Get current Gemini configuration."""
    config = {
        "api_key": "***REDACTED***",
        "model": DEFAULT_MODEL,
        "temperature": DEFAULT_TEMPERATURE,
        "max_tokens": DEFAULT_MAX_TOKENS,
        "top_p": DEFAULT_TOP_P,
        "top_k": DEFAULT_TOP_K,
    }
    return json.dumps(config, indent=2)


@mcp_server.resource("models://list")
async def list_models() -> str:
    """List available Gemini models."""
    try:
        models = client.models.list()
        model_data = [
            {
                "name": model.name,
                "display_name": model.display_name
                if hasattr(model, "display_name")
                else "",
                "description": model.description
                if hasattr(model, "description")
                else "",
                "version": model.version if hasattr(model, "version") else "",
                "supported_features": {
                    "text": getattr(model, "text_supported", False),
                    "image": getattr(model, "image_supported", False),
                },
            }
            for model in models
            if model.name and "gemini" in model.name
        ]
        return json.dumps(model_data, indent=2)
    except Exception as e:
        return json.dumps([{"error": f"Error listing models: {str(e)}"}], indent=2)


@mcp_server.tool()
async def generate_text(
    prompt: str,
    system_instruction: Optional[str] = None,
    custom_message: Optional[str] = None,
    context_files: List[str] = [],
    output_file: Optional[str] = None,
) -> str:
    """Generate text using Gemini model.

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
        # Configure generation parameters
        model_name = DEFAULT_MODEL
        params: GenerateContentConfigDict = {
            "temperature": DEFAULT_TEMPERATURE,
            "max_output_tokens": DEFAULT_MAX_TOKENS,
            "top_p": DEFAULT_TOP_P,
            "top_k": DEFAULT_TOP_K,
        }

        # Read content of context files
        file_contexts: List[Dict[str, Any]] = []
        for file_path in context_files:
            content = read_file_content(file_path)
            if content.startswith("Error"):
                return content

            file_info: Dict[str, Any] = {
                "path": file_path,
                "name": pathlib.Path(file_path).name,
                "content": content,
            }
            file_contexts.append(file_info)

        # Create file context section
        file_context_section = ""
        if file_contexts:
            file_context_section = "\n\nFile context for reference:\n\n"
            file_context_section += "\n\n".join(
                [
                    f"--- {f.get('path', '')} ---\n```\n{f.get('content', '')}\n```"
                    for f in file_contexts
                ]
            )

        # Prepare the prompt with file context
        complete_prompt = prompt
        if file_contexts:
            complete_prompt = f"{prompt}\n{file_context_section}"

        # Handle system instruction and custom message
        instructions = [NO_FLATTERY_INSTRUCTION]

        if system_instruction:
            instructions.append(system_instruction)
        if custom_message:
            instructions.append(custom_message)

        params["system_instruction"] = "\n\n".join(instructions)

        # Create content
        response = await client.aio.models.generate_content(
            model=model_name,
            contents=complete_prompt,
            config=params,
        )
        if output_file and response.text is not None:
            result = write_to_file(output_file, response.text)
            if result.startswith("Error"):
                return f"{result}\n\nResponse:\n{response.text}"
            return f"Repository structure analysis written to {output_file}"

        return response.text if response.text is not None else "No text generated"
    except Exception as e:
        return f"Error generating content: {str(e)}"


@mcp_server.tool()
async def search_and_analyze_current_info(
    question: str,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
    model: Optional[str] = None,
    custom_message: Optional[str] = None,
) -> str:
    """Search for current information and provide an AI-analyzed response.

    This tool gives Gemini the ability to search the web for real-time information
    and provide a comprehensive, fact-based response. The AI will automatically
    search for relevant information and synthesize it into a well-sourced answer.


    Args:
        question: The question or request you want Gemini to answer using current web data.

    Returns:
        AI-generated response based on current web search results.
    """
    try:
        # Configure generation parameters
        model_name = DEFAULT_MODEL

        # Create content with custom message if provided
        prompt = question
        if custom_message:
            prompt = f"{question}\n\nAdditional context: {custom_message}"

        # Generate content with Google Search tool
        response = await client.aio.models.generate_content(
            model=model_name,
            contents=prompt,
            config=GenerateContentConfig(
                tools=[Tool(google_search=GoogleSearch())],
                temperature=temperature if temperature is not None else 0.0,
                max_output_tokens=max_tokens
                if max_tokens is not None
                else DEFAULT_MAX_TOKENS,
                top_p=DEFAULT_TOP_P,
                top_k=DEFAULT_TOP_K,
                system_instruction=NO_FLATTERY_INSTRUCTION,
            ),
        )

        return response.text if response.text is not None else "No results generated"
    except Exception as e:
        return f"Error performing web search: {str(e)}"


# @mcp_server.tool()
# async def analyze_image(
#     image_path: str,
#     prompt: str,
#     temperature: Optional[float] = None,
# ) -> str | None:
#     """Analyze an image using Gemini's multimodal capabilities.

#     Args:
#         image_path: Path to the image file.
#         prompt: Text prompt describing what to analyze in the image.
#         temperature: Optional temperature override (0.0-1.0).

#     Returns:
#         Analysis of the image as text.
#     """
#     try:
#         # Open the image file
#         image = Image.open(image_path)

#         # Configure generation parameters
#         params: GenerateContentConfigDict = {
#             "temperature": temperature
#             if temperature is not None
#             else DEFAULT_TEMPERATURE,
#             "max_output_tokens": DEFAULT_MAX_TOKENS,
#             "top_p": DEFAULT_TOP_P,
#             "top_k": DEFAULT_TOP_K,
#         }

#         # Create multimodal content

#         # Generate content with image
#         response = await client.aio.models.generate_content(
#             model=DEFAULT_MODEL,
#             contents=[image, prompt],
#             config=params,
#         )

#         return response.text
#     except Exception as e:
#         return f"Error analyzing image: {str(e)}"


# @mcp_server.tool()
# async def json_generate(
#     prompt: str,
#     system_instruction: str,
#     json_schema: str,
#     temperature: Optional[float] = 0.2,
# ) -> str | None:
#     """Generate structured output according to a JSON schema.

#     Args:
#         prompt: The user prompt to send to Gemini.
#         system_instruction: System instructions to guide structured generation.
#         json_schema: JSON schema that defines the expected structure.
#         temperature: Optional temperature override (0.0-1.0).

#     Returns:
#         JSON-formatted structured response.
#     """
#     try:
#         # Configure generation parameters
#         params: GenerateContentConfigDict = {
#             "temperature": temperature
#             if temperature is not None
#             else DEFAULT_TEMPERATURE,
#             "max_output_tokens": DEFAULT_MAX_TOKENS,
#             "top_p": DEFAULT_TOP_P,
#             "top_k": DEFAULT_TOP_K,
#         }

#         # Create system instruction including schema
#         if system_instruction and json_schema:
#             params["system_instruction"] = (
#                 f"{system_instruction}\n\n"
#                 f"You must respond with valid JSON that conforms to this schema:\n"
#                 f"{json_schema}\n\n"
#                 f"DO NOT include any explanations, only output valid JSON."
#             )

#         # Generate structured content
#         response = await client.aio.models.generate_content(
#             model=DEFAULT_MODEL,
#             contents=prompt,
#             config=params,
#         )
#         if not response.text:
#             return json.dumps(
#                 {
#                     "error": "Failed to generate valid JSON",
#                     "raw_response": response.text,
#                 },
#                 indent=2,
#             )

#         # Extract JSON from the response
#         try:
#             # Validate that the response is valid JSON
#             result = json.loads(response.text)
#             return json.dumps(result, indent=2)
#         except json.JSONDecodeError:
#             # If not valid JSON, return an error message in JSON format
#             return json.dumps(
#                 {
#                     "error": "Failed to generate valid JSON",
#                     "raw_response": response.text,
#                 },
#                 indent=2,
#             )
#     except Exception as e:
#         error_message = str(e)
#         print(f"Structured generate error: {error_message}")  # Log the error
#         return json.dumps(
#             {"error": f"Error generating structured content: {error_message}"},
#             indent=2,
#         )


@mcp_server.prompt()
def generate_text_prompt(custom_prompt: str) -> str:
    """Create a prompt for text generation."""
    return f"""Please respond to the following:

{custom_prompt}"""


@mcp_server.prompt()
def structured_output_prompt(task_description: str, schema_description: str) -> str:
    """Create a prompt for structured output generation."""
    return f"""Please generate a structured response based on the following task:

Task: {task_description}

The response should be structured according to this schema description:
{schema_description}

Please ensure your response is properly formatted."""


# Utility functions for file operations
def read_file_content(file_path: str) -> str:
    """Read file content safely with error handling.

    Args:
        file_path: Path to the file to read.

    Returns:
        The file content as string or error message.
    """
    try:
        path = pathlib.Path(file_path)
        if not path.exists():
            return f"Error: File not found at {file_path}"
        if not path.is_file():
            return f"Error: {file_path} is not a file"
        return path.read_text(encoding="utf-8")
    except Exception as e:
        return f"Error reading file {file_path}: {str(e)}"


def get_file_info(file_path: str) -> Dict[str, object]:
    """Get file metadata and content.

    Args:
        file_path: Path to the file.

    Returns:
        Dictionary with file info including path, name, extension, and content.
    """
    path = pathlib.Path(file_path)
    if not path.exists():
        return {"error": f"File not found: {file_path}"}

    try:
        return {
            "path": str(path.absolute()),
            "name": path.name,
            "extension": path.suffix,
            "parent_dir": str(path.parent),
            "content": path.read_text(encoding="utf-8"),
            "size_bytes": path.stat().st_size,
            "last_modified": path.stat().st_mtime,
        }
    except Exception as e:
        return {"error": f"Error processing {file_path}: {str(e)}"}


def write_to_file(file_path: str, content: str) -> str:
    """Write content to a file safely with error handling.

    Args:
        file_path: Path where to write the file.
        content: Content to write to the file.

    Returns:
        Success message or error message.
    """
    try:
        path = pathlib.Path(file_path)
        # Create parent directories if they don't exist
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        return f"Successfully wrote to {file_path}"
    except Exception as e:
        return f"Error writing to {file_path}: {str(e)}"


def scan_directory(
    dir_path: str,
    pattern: str = "*",
    max_depth: int = 3,
    exclude_dirs: Optional[List[str]] = None,
) -> List[Dict[str, object]]:
    """Scan a directory for files matching a pattern.

    Args:
        dir_path: Directory to scan.
        pattern: Glob pattern to match files.
        max_depth: Maximum directory depth to traverse.
        exclude_dirs: Directory names relative to ``dir_path`` that should be
            skipped.

    Returns:
        List of dictionaries with file info.
    """
    try:
        path = pathlib.Path(dir_path)
        if not path.exists() or not path.is_dir():
            return [{"error": f"Not a valid directory: {dir_path}"}]

        files: List[Dict[str, object]] = []
        base = pathlib.Path(dir_path)
        exclude_paths = (
            [base / d for d in exclude_dirs] if exclude_dirs is not None else []
        )
        for i, file_path in enumerate(path.glob(pattern)):
            if i > 100:  # Safety limit
                files.append({"note": "Directory scan limited to 100 files"})
                break

            # Calculate depth from base dir
            rel_path = file_path.relative_to(path)
            depth = len(rel_path.parts)

            if depth > max_depth:
                continue

            if any(file_path.is_relative_to(p) for p in exclude_paths):
                continue

            if file_path.is_file():
                files.append(
                    {
                        "path": str(file_path),
                        "name": file_path.name,
                        "extension": file_path.suffix,
                        "parent_dir": str(file_path.parent),
                        "size_bytes": file_path.stat().st_size,
                    }
                )

        return files
    except Exception as e:
        return [{"error": f"Error scanning directory {dir_path}: {str(e)}"}]


# Code planning and review tools
@mcp_server.tool()
async def analyze_repo_structure(
    repo_path: str,
    output_file: Optional[str] = None,
    custom_message: Optional[str] = None,
    exclude_dirs: Optional[List[str]] = None,
) -> str:
    """Analyze repository structure and organization.

    Args:
        repo_path: Path to the repository root directory.
        output_file: Optional file path to write the analysis to.
        custom_message: Optional custom message providing additional context or instructions.
        exclude_dirs: Directory names inside ``repo_path`` to exclude from the analysis.

    Returns:
        Repository structure analysis.
    """
    try:
        # Scan directory structure
        repo_dir = pathlib.Path(repo_path)
        if not repo_dir.exists() or not repo_dir.is_dir():
            return f"Error: {repo_path} is not a valid directory"

        # Get top-level directories and files
        top_level = scan_directory(
            repo_path, "*", max_depth=1, exclude_dirs=exclude_dirs
        )

        # Scan for important files and patterns
        patterns = [
            "*.py",
            "*.js",
            "*.ts",
            "*.go",
            "*.java",
            "*.c",
            "*.cpp",
            "*.rs",
            "*.json",
            "*.yaml",
            "*.yml",
            "*.toml",
            "README*",
            "Dockerfile*",
        ]

        important_files: List[Dict[str, object]] = []
        for pattern in patterns:
            files = scan_directory(
                repo_path,
                f"**/{pattern}",
                max_depth=5,
                exclude_dirs=exclude_dirs,
            )
            important_files.extend(files[:10])  # Limit results per pattern

        # Build context for the model
        repo_context = {
            "repo_path": repo_path,
            "top_level_structure": top_level,
            "important_files": important_files,
        }

        # Create prompt
        prompt = f"""Analyze this repository structure and provide insights about its organization, 
architecture, and code patterns.

{custom_message if custom_message else ""}

Repository information:
{json.dumps(repo_context, indent=2)}

Your analysis should include:
1. Identified project type and main programming languages
2. Key architectural patterns visible from the structure
3. Organization of code (modules, packages, namespacing)
4. Build/dependency management approach
5. Notable observations about code organization
6. Recommendations for structural improvements (if any)

Provide your analysis in a clear, structured format with headings and bullet points.
"""

        # Generate analysis
        params: GenerateContentConfigDict = {
            "temperature": 0.2,
            "max_output_tokens": DEFAULT_MAX_TOKENS,
            "top_p": DEFAULT_TOP_P,
            "top_k": DEFAULT_TOP_K,
            "system_instruction": NO_FLATTERY_INSTRUCTION,
        }

        response = await client.aio.models.generate_content(
            model=DEFAULT_MODEL,
            contents=prompt,
            config=params,
        )

        analysis = response.text

        # Write to file if specified
        if output_file and analysis is not None:
            result = write_to_file(output_file, analysis)
            if result.startswith("Error"):
                return f"{result}\n\nAnalysis:\n{analysis}"
            return f"Repository structure analysis written to {output_file}"

        return analysis if analysis is not None else "No analysis generated"

    except Exception as e:
        return f"Error analyzing repository structure: {str(e)}"


@mcp_server.tool()
async def identify_patterns(
    code_files: List[str],
    pattern_type: str = "coding",  # Options: coding, architecture, testing, error-handling
    output_file: Optional[str] = None,
    custom_message: Optional[str] = None,
) -> str:
    """Identify patterns in code files.

    Args:
        code_files: List of file paths to analyze.
        pattern_type: Type of patterns to identify (coding, architecture, testing, error-handling).
        output_file: Optional file path to write the analysis to.
        custom_message: Optional custom message providing additional context or instructions.

    Returns:
        Identified patterns in the provided code files.
    """
    try:
        # Read content of all files
        file_contents: List[Dict[str, Any]] = []
        for file_path in code_files:
            content = read_file_content(file_path)
            if content.startswith("Error"):
                return content

            file_info: Dict[str, Any] = {
                "path": file_path,
                "name": pathlib.Path(file_path).name,
                "content": content,
            }
            file_contents.append(file_info)

        # Determine pattern focus
        pattern_focus = {
            "coding": "code style, naming conventions, and common idioms",
            "architecture": "architectural patterns, component structure, and dependency flow",
            "testing": "testing approaches, test coverage, and test organization",
            "error-handling": "error handling strategies, exception patterns, and failure modes",
        }.get(pattern_type, "coding patterns and style")

        # Create prompt
        prompt = f"""Analyze these code files and identify {pattern_focus}.

{custom_message if custom_message else ""}

Provided files:
{json.dumps([{"path": f["path"], "name": f["name"]} for f in file_contents], indent=2)}

For each file, I'll provide the content below:

{chr(10).join([f"--- {f['path']} ---\n{f['content']}\n" for f in file_contents])}

Please identify and explain:
1. Consistent patterns across the files
2. Any inconsistencies or variations in the patterns
3. Notable approaches specific to {pattern_focus}
4. Recommendations for improving consistency

Format your response with clear headings and examples from the code.
"""

        # Generate analysis
        params: GenerateContentConfigDict = {
            "temperature": 0.2,
            "max_output_tokens": DEFAULT_MAX_TOKENS,
            "top_p": DEFAULT_TOP_P,
            "top_k": DEFAULT_TOP_K,
            "system_instruction": NO_FLATTERY_INSTRUCTION,
        }

        response = await client.aio.models.generate_content(
            model=DEFAULT_MODEL,
            contents=prompt,
            config=params,
        )

        analysis = response.text

        # Write to file if specified
        if output_file and analysis is not None:
            result = write_to_file(output_file, analysis)
            if result.startswith("Error"):
                return f"{result}\n\nAnalysis:\n{analysis}"
            return f"Pattern analysis written to {output_file}"

        return analysis if analysis is not None else "No pattern analysis generated"

    except Exception as e:
        return f"Error identifying patterns: {str(e)}"


@mcp_server.tool()
async def plan_implementation(
    requirements: str,
    context_files: List[str] = [],
    output_file: Optional[str] = None,
    custom_message: Optional[str] = None,
) -> str:
    """Generate implementation plan based on requirements and code context.

    Args:
        requirements: Description of what needs to be implemented.
        context_files: List of file paths to provide context.
        output_file: Optional file path to write the plan to.
        temperature: Optional temperature override (0.0-1.0).
        custom_message: Optional custom message providing additional context or instructions.

    Returns:
        Implementation plan with task breakdown.
    """
    try:
        # Read content of context files
        file_contexts: List[Dict[str, Any]] = []
        for file_path in context_files:
            content = read_file_content(file_path)
            if content.startswith("Error"):
                continue  # Skip files with errors but proceed with plan

            file_info: Dict[str, Any] = {
                "path": file_path,
                "name": pathlib.Path(file_path).name,
                "content": content,
            }
            file_contexts.append(file_info)

        # Create prompt
        context_section = ""
        if file_contexts:
            context_section = "Reference code from existing files:\n\n"
            context_section += "\n\n".join(
                [
                    f"--- {f.get('path', '')} ---\n```\n{f.get('content', '')}\n```"
                    for f in file_contexts
                ]
            )

        prompt = f"""Create a detailed implementation plan for the following requirements:

{custom_message if custom_message else ""}

REQUIREMENTS:
{requirements}

{context_section}

Your implementation plan should include:

1. Task Breakdown:
   - List all tasks needed to implement the requirements
   - Provide complexity estimates (Low/Medium/High) for each task
   - Include a suggested implementation order

2. Implementation Approach:
   - Recommended architecture or design pattern
   - Key components and their responsibilities
   - Interfaces and data structures needed

3. Technical Considerations:
   - Potential challenges and mitigations
   - Performance considerations
   - Security considerations (if applicable)
   - Testing approach

Format your response as a structured implementation plan document.
"""

        # Generate implementation plan
        params: GenerateContentConfigDict = {
            "temperature": 0.3,
            "max_output_tokens": DEFAULT_MAX_TOKENS,
            "top_p": DEFAULT_TOP_P,
            "top_k": DEFAULT_TOP_K,
            "system_instruction": NO_FLATTERY_INSTRUCTION,
        }

        response = await client.aio.models.generate_content(
            model=DEFAULT_MODEL,
            contents=prompt,
            config=params,
        )

        plan = response.text

        # Write to file if specified
        if output_file and plan is not None:
            result = write_to_file(output_file, plan)
            if result.startswith("Error"):
                return f"{result}\n\nPlan:\n{plan}"
            return f"Implementation plan written to {output_file}"

        return plan if plan is not None else "No implementation plan generated"

    except Exception as e:
        return f"Error creating implementation plan: {str(e)}"


async def _generate_single_review(
    file_contents: List[Dict[str, Any]],
    review_context: str,
    custom_message: Optional[str] = None,
) -> str:
    """Helper function to generate a single code review with specific context.

    Args:
        file_contents: List of file info dicts with path, name, and content.
        review_context: Specific instructions for this code review.
        custom_message: Optional global context message.

    Returns:
        Generated code review as string.
    """
    try:
        # Create prompt with specific review context
        prompt_parts = [
            "Perform a detailed code review with the following focus:",
            f"\nReview Context: {review_context}",
        ]

        if custom_message:
            prompt_parts.append(f"\nAdditional Context: {custom_message}")

        prompt_parts.extend(
            [
                "\n\nProvided files for review:",
                f"{json.dumps([{'path': f['path'], 'name': f['name']} for f in file_contents], indent=2)}",
                "\n\nFor each file, I'll provide the content below:\n",
                f"{chr(10).join([f'--- {f["path"]} ---\n```\n{f["content"]}\n```\n' for f in file_contents])}",
                "\n\nYour code review should include:",
                "\n1. Summary of findings",
                "2. Issues identified (organized by severity: Critical, High, Medium, Low)",
                "   - Include line numbers when referencing specific code",
                "   - Explain why each issue is problematic",
                "3. Recommendations for addressing each issue",
                "4. Positive aspects of the code worth highlighting",
                "5. Overall assessment",
                "\n\nProvide concrete, actionable feedback focused on the review context above.",
            ]
        )

        prompt = "\n".join(prompt_parts)

        params: GenerateContentConfigDict = {
            "temperature": 0.2,
            "max_output_tokens": DEFAULT_MAX_TOKENS,
            "top_p": DEFAULT_TOP_P,
            "top_k": DEFAULT_TOP_K,
            "system_instruction": NO_FLATTERY_INSTRUCTION,
        }

        response = await client.aio.models.generate_content(
            model=DEFAULT_MODEL,
            contents=prompt,
            config=params,
        )

        return response.text if response.text is not None else "No review generated"
    except Exception as e:
        return f"Error generating code review: {str(e)}"


@mcp_server.tool()
async def review_code(
    code_files: List[str],
    review_focus: str = "all",  # Options: all, quality, security, performance, style
    review_specifications: Optional[List[Tuple[str, str]]] = None,
    output_file: Optional[str] = None,
    custom_message: Optional[str] = None,
) -> str:
    """Review code for quality, security, performance, or style issues.

    Can generate multiple code reviews concurrently with different focus areas.

    Args:
        code_files: List of file paths to review.
        review_focus: Focus area for the review (all, quality, security, performance, style).
                     Ignored if review_specifications is provided.
        review_specifications: Optional list of (review_context, output_path) tuples (max 5).
                             Each tuple specifies custom context and output file for a review.
                             Example: [("Focus on security vulnerabilities", "security_review.md"),
                                     ("Analyze performance bottlenecks", "performance_review.md")]
        output_file: Optional file path for single review (used if review_specifications is None).
        custom_message: Additional context applied to all reviews.

    Returns:
        Summary of generated reviews or error message.
    """
    try:
        # Read content of all files
        file_contents: List[Dict[str, Any]] = []
        for file_path in code_files:
            content = read_file_content(file_path)
            if content.startswith("Error"):
                return content

            file_info: Dict[str, Any] = {
                "path": file_path,
                "name": pathlib.Path(file_path).name,
                "content": content,
            }
            file_contents.append(file_info)

        # Handle review_specifications for concurrent generation
        if review_specifications:
            # Validate max 5 specifications
            if len(review_specifications) > 5:
                return "Error: Maximum 5 review specifications allowed"

            # Create tasks for concurrent generation
            tasks: List[Any] = []
            output_paths: List[str] = []
            for review_context, output_path in review_specifications:
                task = _generate_single_review(
                    file_contents, review_context, custom_message
                )
                tasks.append(task)
                output_paths.append(output_path)

            # Execute all tasks concurrently using asyncio.gather
            review_results = await asyncio.gather(*tasks, return_exceptions=True)

            # Process results and write to files
            results: List[str] = []
            for review_result, output_path in zip(review_results, output_paths):
                if isinstance(review_result, Exception):
                    results.append(f"Failed for {output_path}: {str(review_result)}")
                elif isinstance(review_result, str) and review_result.startswith(
                    "Error"
                ):
                    results.append(f"Failed for {output_path}: {review_result}")
                elif isinstance(review_result, str):
                    write_result = write_to_file(output_path, review_result)
                    if write_result.startswith("Error"):
                        results.append(f"Failed to write {output_path}: {write_result}")
                    else:
                        results.append(f"Code review written to {output_path}")
                else:
                    results.append(f"Failed for {output_path}: Unexpected result type")

            # Return summary of all results
            return "\n".join(results)

        # Backward compatibility: single review generation
        else:
            # Determine review focus instructions
            focus_instructions = {
                "all": "all aspects including code quality, security, performance, and style",
                "quality": "code quality including maintainability, readability, and testability",
                "security": "security vulnerabilities, input validation, authentication, and data handling",
                "performance": "performance optimizations, inefficient algorithms, and resource usage",
                "style": "code style, naming conventions, and adherence to best practices",
            }.get(review_focus, "all aspects of the code")

            review_context = f"Focus on {focus_instructions}"
            review = await _generate_single_review(
                file_contents, review_context, custom_message
            )

            if output_file and review and not review.startswith("Error"):
                result = write_to_file(output_file, review)
                if result.startswith("Error"):
                    return f"{result}\n\nReview:\n{review}"
                return f"Code review written to {output_file}"

            return review

    except Exception as e:
        return f"Error performing code review: {str(e)}"


@mcp_server.tool()
async def design_architecture(
    requirements: str,
    context_files: List[str] = [],
    tech_stack: Optional[str] = None,
    output_file: Optional[str] = None,
    custom_message: Optional[str] = None,
) -> str:
    """Design software architecture based on requirements.

    Args:
        requirements: Functional and non-functional requirements.
        context_files: List of file paths to provide context from existing code.
        tech_stack: Optional technology stack constraints or preferences.
        output_file: Optional file path to write the architecture design to.
        custom_message: Optional custom message providing additional context or instructions.

    Returns:
        Architecture design document.
    """
    try:
        # Read content of context files
        file_contexts: List[Dict[str, Any]] = []
        for file_path in context_files:
            content = read_file_content(file_path)
            if content.startswith("Error"):
                continue  # Skip files with errors but proceed with design

            file_info: Dict[str, Any] = {
                "path": file_path,
                "name": pathlib.Path(file_path).name,
                "content": content,
            }
            file_contexts.append(file_info)

        # Create prompt
        context_section = ""
        if file_contexts:
            context_section = "Reference code from existing files:\n\n"
            context_section += "\n\n".join(
                [
                    f"--- {f.get('path', '')} ---\n```\n{f.get('content', '')}\n```"
                    for f in file_contexts
                ]
            )

        tech_stack_section = ""
        if tech_stack:
            tech_stack_section = f"TECHNOLOGY STACK CONSTRAINTS:\n{tech_stack}\n\n"

        prompt = f"""Design a software architecture for the following requirements:

{custom_message if custom_message else ""}

REQUIREMENTS:
{requirements}

{tech_stack_section}
{context_section}

Your architecture design should include:

1. System Overview:
   - High-level architecture diagram (described in text)
   - Key components and their responsibilities
   - Data flow between components

2. Component Design:
   - Detailed description of each component
   - Interfaces between components
   - Data models and schemas

3. Technology Choices:
   - Recommended technologies for each component
   - Justification for technology choices
   - Alternative options considered

4. Non-Functional Aspects:
   - Scalability approach
   - Security considerations
   - Performance optimizations
   - Maintainability and extensibility

5. Implementation Considerations:
   - Potential challenges and mitigations
   - Deployment strategy
   - Monitoring and observability

Format your response as a structured architecture design document with clear sections.
"""

        # Generate architecture design
        params: GenerateContentConfigDict = {
            "temperature": 0.3,
            "max_output_tokens": DEFAULT_MAX_TOKENS,
            "top_p": DEFAULT_TOP_P,
            "top_k": DEFAULT_TOP_K,
            "system_instruction": NO_FLATTERY_INSTRUCTION,
        }

        response = await client.aio.models.generate_content(
            model=DEFAULT_MODEL,
            contents=prompt,
            config=params,
        )

        design = response.text

        # Write to file if specified
        if output_file and design is not None:
            result = write_to_file(output_file, design)
            if result.startswith("Error"):
                return f"{result}\n\nArchitecture Design:\n{design}"
            return f"Architecture design written to {output_file}"

        return design if design is not None else "No architecture design generated"

    except Exception as e:
        return f"Error creating architecture design: {str(e)}"


async def _generate_single_test_suite(
    file_contents: List[Dict[str, Any]],
    testing_framework: str,
    test_context: str,
    custom_message: Optional[str] = None,
) -> str:
    """Helper function to generate a single test suite with specific context.

    Args:
        file_contents: List of file info dicts with path, name, and content.
        testing_framework: Test framework to use.
        test_context: Specific instructions for this test suite.
        custom_message: Optional global context message.

    Returns:
        Generated test suite as string.
    """
    try:
        file_sections = "\n".join(
            [
                f"--- {info['path']} ---\n```\n{info['content']}\n```"
                for info in file_contents
            ]
        )

        # Build prompt with specific test context
        prompt_parts = [
            f"Generate unit tests using {testing_framework} for the following files.",
            f"\nTest Context: {test_context}",
        ]

        if custom_message:
            prompt_parts.append(f"\nAdditional Instructions: {custom_message}")

        prompt_parts.extend(
            [
                f"\n\nFiles:\n{json.dumps([{'path': info['path'], 'name': info['name']} for info in file_contents], indent=2)}",
                f"\n\n{file_sections}",
            ]
        )

        prompt = "\n".join(prompt_parts)

        params: GenerateContentConfigDict = {
            "temperature": 0.2,
            "max_output_tokens": DEFAULT_MAX_TOKENS,
            "top_p": DEFAULT_TOP_P,
            "top_k": DEFAULT_TOP_K,
            "system_instruction": NO_FLATTERY_INSTRUCTION,
        }

        response = await client.aio.models.generate_content(
            model=DEFAULT_MODEL,
            contents=prompt,
            config=params,
        )

        return response.text if response.text is not None else "No tests generated"
    except Exception as e:
        return f"Error generating test suite: {str(e)}"


@mcp_server.tool()
async def generate_unit_tests(
    code_files: List[str],
    testing_framework: str = "pytest",
    test_specifications: Optional[List[Tuple[str, str]]] = None,
    output_file: Optional[str] = None,
    custom_message: Optional[str] = None,
) -> str:
    """Generate unit tests for the provided code files.

    Can generate multiple test suites concurrently with different contexts.

    Args:
        code_files: Files to generate tests for.
        testing_framework: Test framework to target.
        test_specifications: Optional list of (test_context, output_path) tuples (max 5).
                           Each tuple specifies custom context and output file for a test suite.
                           Example: [("Focus on edge cases", "test_edge.py"),
                                   ("Test error handling", "test_errors.py")]
        output_file: Optional file path for single test suite (used if test_specifications is None).
        custom_message: Additional instructions applied to all test generation.

    Returns:
        Summary of generated tests or error message.
    """
    try:
        file_contents: List[Dict[str, Any]] = []
        for file_path in code_files:
            content = read_file_content(file_path)
            if content.startswith("Error"):
                return content

            file_contents.append(
                {
                    "path": file_path,
                    "name": pathlib.Path(file_path).name,
                    "content": content,
                }
            )

        # Handle test_specifications for concurrent generation
        if test_specifications:
            # Validate max 5 specifications
            if len(test_specifications) > 5:
                return "Error: Maximum 5 test specifications allowed"

            # Create tasks for concurrent generation
            tasks: List[Any] = []
            output_paths: List[str] = []
            for test_context, output_path in test_specifications:
                task = _generate_single_test_suite(
                    file_contents, testing_framework, test_context, custom_message
                )
                tasks.append(task)
                output_paths.append(output_path)

            # Execute all tasks concurrently using asyncio.gather
            test_results = await asyncio.gather(*tasks, return_exceptions=True)

            # Process results and write to files
            results: List[str] = []
            for test_result, output_path in zip(test_results, output_paths):
                if isinstance(test_result, Exception):
                    results.append(f"Failed for {output_path}: {str(test_result)}")
                elif isinstance(test_result, str) and test_result.startswith("Error"):
                    results.append(f"Failed for {output_path}: {test_result}")
                elif isinstance(test_result, str):
                    write_result = write_to_file(output_path, test_result)
                    if write_result.startswith("Error"):
                        results.append(f"Failed to write {output_path}: {write_result}")
                    else:
                        results.append(f"Generated tests written to {output_path}")
                else:
                    results.append(f"Failed for {output_path}: Unexpected result type")

            # Return summary of all results
            return "\n".join(results)

        # Backward compatibility: single test suite generation
        else:
            test_context = "Generate comprehensive unit tests"
            tests = await _generate_single_test_suite(
                file_contents, testing_framework, test_context, custom_message
            )

            if output_file and tests and not tests.startswith("Error"):
                result = write_to_file(output_file, tests)
                if result.startswith("Error"):
                    return f"{result}\n\nUnit tests:\n{tests}"
                return f"Unit tests written to {output_file}"

            return tests
    except Exception as e:
        return f"Error generating unit tests: {str(e)}"


@mcp_server.tool()
async def explain_code(
    code_files: List[str],
    output_file: Optional[str] = None,
    custom_message: Optional[str] = None,
) -> str:
    """Explain the provided code files in plain language.

    Args:
        code_files: Files to explain.
        output_file: Optional file path to write the explanation to.
        custom_message: Additional instructions for the explanation.

    Returns:
        Explanation of the code.
    """
    try:
        file_contents: List[Dict[str, Any]] = []
        for file_path in code_files:
            content = read_file_content(file_path)
            if content.startswith("Error"):
                return content

            file_contents.append(
                {
                    "path": file_path,
                    "name": pathlib.Path(file_path).name,
                    "content": content,
                }
            )

        file_sections = "\n".join(
            [
                f"--- {info['path']} ---\n```\n{info['content']}\n```"
                for info in file_contents
            ]
        )

        prompt = (
            "Explain in detail what the following code does.\n\n"
            f"{custom_message if custom_message else ''}\n\n"
            f"Files:\n"
            f"{json.dumps([{'path': info['path'], 'name': info['name']} for info in file_contents], indent=2)}\n\n"
            f"{file_sections}"
        )

        params: GenerateContentConfigDict = {
            "temperature": 0.2,
            "max_output_tokens": DEFAULT_MAX_TOKENS,
            "top_p": DEFAULT_TOP_P,
            "top_k": DEFAULT_TOP_K,
            "system_instruction": NO_FLATTERY_INSTRUCTION,
        }

        response = await client.aio.models.generate_content(
            model=DEFAULT_MODEL,
            contents=prompt,
            config=params,
        )

        explanation = response.text
        if output_file and explanation is not None:
            result = write_to_file(output_file, explanation)
            if result.startswith("Error"):
                return f"{result}\n\nExplanation:\n{explanation}"
            return f"Code explanation written to {output_file}"

        return explanation if explanation is not None else "No explanation generated"
    except Exception as e:
        return f"Error explaining code: {str(e)}"


@mcp_server.tool()
async def summarize_files(
    files: List[str],
    output_file: Optional[str] = None,
    custom_message: Optional[str] = None,
) -> str:
    """Summarize the provided files."""
    try:
        file_contents: List[Dict[str, Any]] = []
        for file_path in files:
            content = read_file_content(file_path)
            if content.startswith("Error"):
                return content

            file_contents.append(
                {
                    "path": file_path,
                    "name": pathlib.Path(file_path).name,
                    "content": content,
                }
            )

        file_sections = "\n".join(
            [
                f"--- {info['path']} ---\n```\n{info['content']}\n```"
                for info in file_contents
            ]
        )

        prompt = (
            "Provide a concise summary of the following files.\n\n"
            f"{custom_message if custom_message else ''}\n\n"
            "Files:\n"
            f"{json.dumps([{'path': info['path'], 'name': info['name']} for info in file_contents], indent=2)}\n\n"
            f"{file_sections}"
        )

        params: GenerateContentConfigDict = {
            "temperature": 0.2,
            "max_output_tokens": DEFAULT_MAX_TOKENS,
            "top_p": DEFAULT_TOP_P,
            "top_k": DEFAULT_TOP_K,
            "system_instruction": NO_FLATTERY_INSTRUCTION,
        }

        response = await client.aio.models.generate_content(
            model=DEFAULT_MODEL,
            contents=prompt,
            config=params,
        )

        summary = response.text
        if output_file and summary is not None:
            result = write_to_file(output_file, summary)
            if result.startswith("Error"):
                return f"{result}\n\nSummary:\n{summary}"
            return f"Summary written to {output_file}"

        return summary if summary is not None else "No summary generated"
    except Exception as e:
        return f"Error summarizing files: {str(e)}"


@mcp_server.tool()
async def extract_todos(
    code_files: List[str],
    output_file: Optional[str] = None,
) -> str:
    """Extract TODO and FIXME comments from code files."""
    try:
        todos: List[Dict[str, object]] = []
        for file_path in code_files:
            content = read_file_content(file_path)
            if content.startswith("Error"):
                return content

            for lineno, line in enumerate(content.splitlines(), start=1):
                if "TODO" in line or "FIXME" in line:
                    todos.append(
                        {
                            "file": file_path,
                            "line": lineno,
                            "text": line.strip(),
                        }
                    )

        result = json.dumps(todos, indent=2)
        if output_file and todos:
            write_to_file(output_file, result)
            return f"TODO list written to {output_file}"

        return result if todos else "No TODO items found"
    except Exception as e:
        return f"Error extracting TODOs: {str(e)}"


@mcp_server.tool()
async def cleanup_repo(
    repo_path: str,
    max_files: int = 50,
    output_file: Optional[str] = None,
    custom_message: Optional[str] = None,
    exclude_dirs: Optional[List[str]] = None,
) -> str:
    """Suggest cleanup actions for the given repository.

    Args:
        repo_path: Path to the repository root directory.
        max_files: Number of large files to report. Defaults to 50.
        output_file: Optional file path to write the suggestions to.
        custom_message: Optional custom message providing additional context.
        exclude_dirs: Directory names inside ``repo_path`` to exclude from the
            cleanup scan.

    Returns:
        Cleanup suggestions generated by the model.
    """
    try:
        repo_dir = pathlib.Path(repo_path)
        if not repo_dir.exists() or not repo_dir.is_dir():
            return f"Error: {repo_path} is not a valid directory"

        exclude_paths = (
            [repo_dir / d for d in exclude_dirs] if exclude_dirs is not None else []
        )

        file_infos: List[Dict[str, Union[str, int]]] = []
        for path in repo_dir.rglob("*"):
            if any(path.is_relative_to(p) for p in exclude_paths):
                continue
            if path.is_file():
                try:
                    file_infos.append(
                        {
                            "path": str(path.relative_to(repo_dir)),
                            "size_bytes": path.stat().st_size,
                        }
                    )
                except OSError:
                    continue

        file_infos.sort(
            key=lambda f: cast(int, f["size_bytes"]),
            reverse=True,
        )
        largest_files = file_infos[:max_files]

        dir_patterns = [
            "node_modules",
            "dist",
            "build",
            "venv",
            ".venv",
            "__pycache__",
            ".mypy_cache",
            ".pytest_cache",
            ".idea",
            ".vscode",
        ]

        bloat_dirs: List[str] = []
        for pattern in dir_patterns:
            for found in repo_dir.rglob(pattern):
                if any(found.is_relative_to(p) for p in exclude_paths):
                    continue
                if found.is_dir():
                    bloat_dirs.append(str(found.relative_to(repo_dir)))

        repo_info = {
            "largest_files": largest_files,
            "bloat_dirs": bloat_dirs,
        }

        prompt = (
            "You are an expert in repository hygiene. "
            "Suggest actions to reduce code bloat and technical debt.\n\n"
            f"{custom_message if custom_message else ''}\n\n"
            f"Repository info:\n{json.dumps(repo_info, indent=2)}\n\n"
            "Provide a prioritized list of cleanup tasks."
        )

        params: GenerateContentConfigDict = {
            "temperature": 0.2,
            "max_output_tokens": DEFAULT_MAX_TOKENS,
            "top_p": DEFAULT_TOP_P,
            "top_k": DEFAULT_TOP_K,
            "system_instruction": NO_FLATTERY_INSTRUCTION,
        }

        response = await client.aio.models.generate_content(
            model=DEFAULT_MODEL,
            contents=prompt,
            config=params,
        )

        suggestions = response.text
        if output_file and suggestions is not None:
            result = write_to_file(output_file, suggestions)
            if result.startswith("Error"):
                return f"{result}\n\nSuggestions:\n{suggestions}"
            return f"Cleanup suggestions written to {output_file}"

        return suggestions if suggestions is not None else "No suggestions generated"
    except Exception as e:
        return f"Error generating cleanup suggestions: {str(e)}"


@mcp_server.tool()
async def generate_changelog(
    repo_path: str,
    max_entries: int = 10,
    output_file: Optional[str] = None,
    custom_message: Optional[str] = None,
) -> str:
    """Generate a changelog summary from recent commits.

    Args:
        repo_path: Path to the git repository.
        max_entries: Number of commits to include. Defaults to 10.
        output_file: Optional file path to write the changelog to.
        custom_message: Additional instructions for the summary.

    Returns:
        AI-generated changelog summary.
    """
    try:
        result = subprocess.run(
            ["git", "-C", repo_path, "log", f"-n{max_entries}", "--pretty=%s"],
            capture_output=True,
            text=True,
            check=True,
        )
        commit_messages = result.stdout.strip().splitlines()
    except Exception as e:  # pragma: no cover - subprocess errors
        return f"Error retrieving commit messages: {str(e)}"

    prompt = (
        "Create a concise changelog from the following commit messages.\n\n"
        f"{custom_message if custom_message else ''}\n\n"
        "Commit messages:\n"
        f"{json.dumps(commit_messages, indent=2)}"
    )

    params: GenerateContentConfigDict = {
        "temperature": 0.2,
        "max_output_tokens": DEFAULT_MAX_TOKENS,
        "top_p": DEFAULT_TOP_P,
        "top_k": DEFAULT_TOP_K,
        "system_instruction": NO_FLATTERY_INSTRUCTION,
    }

    response = await client.aio.models.generate_content(
        model=DEFAULT_MODEL,
        contents=prompt,
        config=params,
    )

    changelog = response.text
    if output_file and changelog is not None:
        result = write_to_file(output_file, changelog)
        if result.startswith("Error"):
            return f"{result}\n\nChangelog:\n{changelog}"
        return f"Changelog written to {output_file}"

    return changelog if changelog is not None else "No changelog generated"


if __name__ == "__main__":
    mcp_server.run(transport="stdio")
