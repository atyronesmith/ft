"""
Prompt template functionality for various formats.

Implements templates for Alpaca, ChatML, Llama, and custom formats
with proper formatting and validation.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Any, List, Optional, Union

from .exceptions import TemplateError


class PromptTemplate(ABC):
    """Abstract base class for prompt templates."""

    @abstractmethod
    def format(self, data: Dict[str, Any], for_inference: bool = False) -> str:
        """
        Format data according to template.

        Args:
            data: Input data to format
            for_inference: Whether formatting for inference (no output)

        Returns:
            Formatted prompt string

        Raises:
            TemplateError: If required fields are missing
        """
        pass

    def format_batch(self, dataset: List[Dict[str, Any]], for_inference: bool = False) -> List[str]:
        """
        Format a batch of data items.

        Args:
            dataset: List of data items to format
            for_inference: Whether formatting for inference

        Returns:
            List of formatted prompt strings
        """
        return [self.format(item, for_inference) for item in dataset]


class AlpacaTemplate(PromptTemplate):
    """Alpaca instruction-following template."""

    def format(self, data: Dict[str, Any], for_inference: bool = False) -> str:
        """Format data using Alpaca template."""
        if "instruction" not in data:
            raise TemplateError("Missing required field 'instruction'")

        if not for_inference and "output" not in data:
            raise TemplateError("Missing required field 'output'")

        # Check if input field is present
        has_input = "input" in data and data["input"].strip()

        if has_input:
            prompt = (
                "Below is an instruction that describes a task, paired with an input that provides further context. "
                "Write a response that appropriately completes the request.\n\n"
                "### Instruction:\n"
                f"{data['instruction']}\n\n"
                "### Input:\n"
                f"{data['input']}\n\n"
                "### Response:\n"
            )
        else:
            prompt = (
                "Below is an instruction that describes a task. "
                "Write a response that appropriately completes the request.\n\n"
                "### Instruction:\n"
                f"{data['instruction']}\n\n"
                "### Response:\n"
            )

        if not for_inference:
            prompt += data["output"]

        return prompt


class ChatMLTemplate(PromptTemplate):
    """ChatML conversation template."""

    def __init__(self, system_message: Optional[str] = None):
        """
        Initialize ChatML template.

        Args:
            system_message: Optional system message to include
        """
        self.system_message = system_message

    def format(self, data: Dict[str, Any], for_inference: bool = False) -> str:
        """Format data using ChatML template."""
        messages = []

        # Add system message if configured
        if self.system_message:
            messages.append(f"<|im_start|>system\n{self.system_message}<|im_end|>")

        # Handle different input formats
        if "messages" in data:
            # Multi-turn conversation format
            for message in data["messages"]:
                role = message["role"]
                content = message["content"]
                messages.append(f"<|im_start|>{role}\n{content}<|im_end|>")
        else:
            # Simple instruction-output format
            if "instruction" not in data:
                raise TemplateError("Missing required field 'instruction'")

            messages.append(f"<|im_start|>user\n{data['instruction']}<|im_end|>")

            if not for_inference:
                if "output" not in data:
                    raise TemplateError("Missing required field 'output'")
                messages.append(f"<|im_start|>assistant\n{data['output']}<|im_end|>")
            else:
                messages.append("<|im_start|>assistant\n")

        return "\n".join(messages)


class LlamaTemplate(PromptTemplate):
    """Llama chat template."""

    def __init__(self, system_message: Optional[str] = None):
        """
        Initialize Llama template.

        Args:
            system_message: Optional system message to include
        """
        self.system_message = system_message

    def format(self, data: Dict[str, Any], for_inference: bool = False) -> str:
        """Format data using Llama template."""
        if "messages" in data:
            # Multi-turn conversation
            formatted_turns = []
            for i in range(0, len(data["messages"]), 2):
                user_msg = data["messages"][i]
                if i + 1 < len(data["messages"]):
                    assistant_msg = data["messages"][i + 1]
                    turn = f"<s>[INST] {user_msg['content']} [/INST] {assistant_msg['content']} </s>"
                else:
                    turn = f"<s>[INST] {user_msg['content']} [/INST] "
                formatted_turns.append(turn)
            return "".join(formatted_turns)
        else:
            # Simple instruction-output format
            if "instruction" not in data:
                raise TemplateError("Missing required field 'instruction'")

            if self.system_message:
                prompt = (
                    f"<s>[INST] <<SYS>>\n"
                    f"{self.system_message}\n"
                    f"<</SYS>>\n\n"
                    f"{data['instruction']} [/INST] "
                )
            else:
                prompt = f"<s>[INST] {data['instruction']} [/INST] "

            if not for_inference:
                if "output" not in data:
                    raise TemplateError("Missing required field 'output'")
                prompt += f"{data['output']} </s>"

            return prompt


class CustomTemplate(PromptTemplate):
    """Custom template using format string."""

    def __init__(self, template_string: str):
        """
        Initialize custom template.

        Args:
            template_string: Format string with {field} placeholders
        """
        self.template_string = template_string

    @classmethod
    def from_file(cls, file_path: Union[str, Path]) -> "CustomTemplate":
        """
        Load custom template from file.

        Args:
            file_path: Path to template file

        Returns:
            CustomTemplate instance
        """
        path_obj = Path(file_path)
        with open(path_obj, 'r', encoding='utf-8') as f:
            template_content = f.read()
        return cls(template_content)

    def format(self, data: Dict[str, Any], for_inference: bool = False) -> str:
        """Format data using custom template."""
        try:
            return self.template_string.format(**data)
        except KeyError as e:
            missing_field = str(e).strip("'")
            raise TemplateError(f"Missing value for placeholder '{missing_field}'")


class TemplateRegistry:
    """Registry for managing prompt templates."""

    def __init__(self):
        """Initialize template registry with built-in templates."""
        self._templates = {
            "alpaca": AlpacaTemplate(),
            "chatml": ChatMLTemplate(),
            "llama": LlamaTemplate(),
        }

    def register_template(self, name: str, template: PromptTemplate) -> None:
        """
        Register a template.

        Args:
            name: Template name
            template: Template instance
        """
        self._templates[name] = template

    def get_template(self, name: str) -> PromptTemplate:
        """
        Get template by name.

        Args:
            name: Template name

        Returns:
            Template instance

        Raises:
            TemplateError: If template not found
        """
        if name not in self._templates:
            raise TemplateError(f"Unknown template '{name}'")
        return self._templates[name]

    def list_templates(self) -> List[str]:
        """
        List available template names.

        Returns:
            List of template names
        """
        return list(self._templates.keys())