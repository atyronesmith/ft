"""
Test-driven development for prompt template functionality.

Tests are written first to drive the implementation of prompt templates
for various formats (Alpaca, ChatML, Llama, custom).
"""


import pytest
from finetune.data.templates import (
    AlpacaTemplate,
    ChatMLTemplate,
    CustomTemplate,
    LlamaTemplate,
    PromptTemplate,
    TemplateError,
    TemplateRegistry,
)


class TestPromptTemplate:
    """Test base prompt template functionality."""

    def test_template_abstract_methods(self):
        """Test that PromptTemplate is an abstract base class."""
        with pytest.raises(TypeError):
            PromptTemplate()

    def test_template_format_interface(self):
        """Test that subclasses must implement format method."""

        class IncompleteTemplate(PromptTemplate):
            pass

        with pytest.raises(TypeError):
            IncompleteTemplate()


class TestAlpacaTemplate:
    """Test Alpaca prompt template functionality."""

    def test_alpaca_basic_format(self):
        """Test basic Alpaca instruction formatting."""
        # Arrange
        template = AlpacaTemplate()
        data = {"instruction": "What is Python?", "output": "Python is a programming language."}

        # Act
        result = template.format(data)

        # Assert
        expected = (
            "Below is an instruction that describes a task. "
            "Write a response that appropriately completes the request.\n\n"
            "### Instruction:\n"
            "What is Python?\n\n"
            "### Response:\n"
            "Python is a programming language."
        )
        assert result == expected

    def test_alpaca_with_input_format(self):
        """Test Alpaca formatting with input field."""
        # Arrange
        template = AlpacaTemplate()
        data = {
            "instruction": "Summarize the following text",
            "input": "Machine learning is a subset of AI...",
            "output": "ML is part of AI focused on data patterns.",
        }

        # Act
        result = template.format(data)

        # Assert
        expected = (
            "Below is an instruction that describes a task, paired with an input that provides further context. "
            "Write a response that appropriately completes the request.\n\n"
            "### Instruction:\n"
            "Summarize the following text\n\n"
            "### Input:\n"
            "Machine learning is a subset of AI...\n\n"
            "### Response:\n"
            "ML is part of AI focused on data patterns."
        )
        assert result == expected

    def test_alpaca_missing_required_field(self):
        """Test Alpaca template with missing required field."""
        template = AlpacaTemplate()
        data = {"instruction": "Test"}  # Missing output

        with pytest.raises(TemplateError, match="Missing required field"):
            template.format(data)

    def test_alpaca_inference_format(self):
        """Test Alpaca template for inference (no output)."""
        # Arrange
        template = AlpacaTemplate()
        data = {
            "instruction": "What is Python?",
        }

        # Act
        result = template.format(data, for_inference=True)

        # Assert
        expected = (
            "Below is an instruction that describes a task. "
            "Write a response that appropriately completes the request.\n\n"
            "### Instruction:\n"
            "What is Python?\n\n"
            "### Response:\n"
        )
        assert result == expected


class TestChatMLTemplate:
    """Test ChatML prompt template functionality."""

    def test_chatml_basic_format(self):
        """Test basic ChatML conversation formatting."""
        # Arrange
        template = ChatMLTemplate()
        data = {"instruction": "What is Python?", "output": "Python is a programming language."}

        # Act
        result = template.format(data)

        # Assert
        expected = (
            "<|im_start|>user\n"
            "What is Python?<|im_end|>\n"
            "<|im_start|>assistant\n"
            "Python is a programming language.<|im_end|>"
        )
        assert result == expected

    def test_chatml_with_system_message(self):
        """Test ChatML with system message."""
        # Arrange
        template = ChatMLTemplate(system_message="You are a helpful assistant.")
        data = {"instruction": "Hello", "output": "Hi there!"}

        # Act
        result = template.format(data)

        # Assert
        expected = (
            "<|im_start|>system\n"
            "You are a helpful assistant.<|im_end|>\n"
            "<|im_start|>user\n"
            "Hello<|im_end|>\n"
            "<|im_start|>assistant\n"
            "Hi there!<|im_end|>"
        )
        assert result == expected

    def test_chatml_conversation_format(self):
        """Test ChatML with multi-turn conversation."""
        # Arrange
        template = ChatMLTemplate()
        data = {
            "messages": [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi!"},
                {"role": "user", "content": "How are you?"},
                {"role": "assistant", "content": "I'm doing well, thank you!"},
            ]
        }

        # Act
        result = template.format(data)

        # Assert
        expected = (
            "<|im_start|>user\n"
            "Hello<|im_end|>\n"
            "<|im_start|>assistant\n"
            "Hi!<|im_end|>\n"
            "<|im_start|>user\n"
            "How are you?<|im_end|>\n"
            "<|im_start|>assistant\n"
            "I'm doing well, thank you!<|im_end|>"
        )
        assert result == expected


class TestLlamaTemplate:
    """Test Llama chat template functionality."""

    def test_llama_basic_format(self):
        """Test basic Llama chat formatting."""
        # Arrange
        template = LlamaTemplate()
        data = {"instruction": "What is Python?", "output": "Python is a programming language."}

        # Act
        result = template.format(data)

        # Assert
        expected = "<s>[INST] What is Python? [/INST] " "Python is a programming language. </s>"
        assert result == expected

    def test_llama_with_system_message(self):
        """Test Llama with system message."""
        # Arrange
        template = LlamaTemplate(system_message="You are a helpful assistant.")
        data = {"instruction": "Hello", "output": "Hi there!"}

        # Act
        result = template.format(data)

        # Assert
        expected = (
            "<s>[INST] <<SYS>>\n"
            "You are a helpful assistant.\n"
            "<</SYS>>\n\n"
            "Hello [/INST] Hi there! </s>"
        )
        assert result == expected

    def test_llama_multi_turn(self):
        """Test Llama multi-turn conversation."""
        # Arrange
        template = LlamaTemplate()
        data = {
            "messages": [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi!"},
                {"role": "user", "content": "How are you?"},
                {"role": "assistant", "content": "I'm doing well!"},
            ]
        }

        # Act
        result = template.format(data)

        # Assert
        expected = (
            "<s>[INST] Hello [/INST] Hi! </s>" "<s>[INST] How are you? [/INST] I'm doing well! </s>"
        )
        assert result == expected


class TestCustomTemplate:
    """Test custom template functionality."""

    def test_custom_template_from_string(self):
        """Test creating custom template from format string."""
        # Arrange
        template_str = "Question: {instruction}\nAnswer: {output}"
        template = CustomTemplate(template_str)
        data = {"instruction": "What is Python?", "output": "A programming language."}

        # Act
        result = template.format(data)

        # Assert
        expected = "Question: What is Python?\nAnswer: A programming language."
        assert result == expected

    def test_custom_template_missing_placeholder(self):
        """Test custom template with missing placeholder value."""
        # Arrange
        template_str = "Question: {instruction}\nAnswer: {output}"
        template = CustomTemplate(template_str)
        data = {"instruction": "What is Python?"}  # Missing output

        # Act & Assert
        with pytest.raises(TemplateError, match="Missing value for placeholder"):
            template.format(data)

    def test_custom_template_extra_fields(self):
        """Test custom template ignores extra fields."""
        # Arrange
        template_str = "Q: {instruction}"
        template = CustomTemplate(template_str)
        data = {"instruction": "What is Python?", "output": "Not used", "extra": "Also not used"}

        # Act
        result = template.format(data)

        # Assert
        assert result == "Q: What is Python?"

    def test_custom_template_from_file(self):
        """Test loading custom template from file."""
        # Arrange
        template_content = "User: {instruction}\nBot: {output}"

        import os
        import tempfile

        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write(template_content)
            temp_path = f.name

        try:
            template = CustomTemplate.from_file(temp_path)
            data = {"instruction": "Hello", "output": "Hi there!"}

            # Act
            result = template.format(data)

            # Assert
            assert result == "User: Hello\nBot: Hi there!"
        finally:
            os.unlink(temp_path)


class TestTemplateRegistry:
    """Test template registry functionality."""

    def test_registry_get_builtin_templates(self):
        """Test getting built-in templates from registry."""
        registry = TemplateRegistry()

        alpaca = registry.get_template("alpaca")
        assert isinstance(alpaca, AlpacaTemplate)

        chatml = registry.get_template("chatml")
        assert isinstance(chatml, ChatMLTemplate)

        llama = registry.get_template("llama")
        assert isinstance(llama, LlamaTemplate)

    def test_registry_register_custom_template(self):
        """Test registering custom template in registry."""
        # Arrange
        registry = TemplateRegistry()
        custom_template = CustomTemplate("Q: {instruction}\nA: {output}")

        # Act
        registry.register_template("custom", custom_template)

        # Assert
        retrieved = registry.get_template("custom")
        assert retrieved is custom_template

    def test_registry_unknown_template(self):
        """Test getting unknown template raises error."""
        registry = TemplateRegistry()

        with pytest.raises(TemplateError, match="Unknown template"):
            registry.get_template("unknown")

    def test_registry_list_templates(self):
        """Test listing available templates."""
        registry = TemplateRegistry()

        templates = registry.list_templates()

        assert "alpaca" in templates
        assert "chatml" in templates
        assert "llama" in templates

    def test_registry_template_overwrite(self):
        """Test overwriting existing template."""
        # Arrange
        registry = TemplateRegistry()
        new_alpaca = CustomTemplate("Custom Alpaca: {instruction} -> {output}")

        # Act
        registry.register_template("alpaca", new_alpaca)

        # Assert
        retrieved = registry.get_template("alpaca")
        assert retrieved is new_alpaca


class TestTemplateIntegration:
    """Test template integration with data loading."""

    def test_template_with_loaded_data(self):
        """Test applying template to loaded dataset."""
        # Arrange
        import json
        import os
        import tempfile

        from finetune.data import JSONLoader

        data = [
            {"instruction": "What is AI?", "output": "Artificial Intelligence"},
            {"instruction": "What is ML?", "output": "Machine Learning"},
        ]

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(data, f)
            temp_path = f.name

        try:
            loader = JSONLoader()
            loaded_data = loader.load(temp_path)

            template = AlpacaTemplate()

            # Act
            formatted_examples = [template.format(item) for item in loaded_data]

            # Assert
            assert len(formatted_examples) == 2
            assert "### Instruction:" in formatted_examples[0]
            assert "What is AI?" in formatted_examples[0]
            assert "Artificial Intelligence" in formatted_examples[0]
        finally:
            os.unlink(temp_path)

    def test_template_batch_processing(self):
        """Test batch processing of templates."""
        # Arrange
        template = ChatMLTemplate()
        dataset = [
            {"instruction": "Hello", "output": "Hi"},
            {"instruction": "Goodbye", "output": "Bye"},
        ]

        # Act
        results = template.format_batch(dataset)

        # Assert
        assert len(results) == 2
        assert "<|im_start|>user" in results[0]
        assert "Hello" in results[0]
        assert "Hi" in results[0]


# Test fixtures
@pytest.fixture
def sample_instruction_data():
    """Provide sample instruction-following data."""
    return {
        "instruction": "Explain the concept of recursion in programming",
        "output": "Recursion is when a function calls itself to solve smaller instances of the same problem.",
    }


@pytest.fixture
def sample_conversation_data():
    """Provide sample conversation data."""
    return {
        "messages": [
            {"role": "user", "content": "What's the weather like?"},
            {"role": "assistant", "content": "I don't have access to current weather data."},
            {"role": "user", "content": "How can I check the weather?"},
            {
                "role": "assistant",
                "content": "You can check weather apps or websites like Weather.com.",
            },
        ]
    }


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
