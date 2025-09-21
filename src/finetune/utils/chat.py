"""
Common chat utilities for consistent prompt generation across training and inference.

This module provides centralized functions for creating chat messages and applying
templates to ensure training and inference use exactly the same format.
"""

from typing import Any, Optional

# System message used consistently across all geography Q&A tasks
GEOGRAPHY_SYSTEM_MESSAGE = "You are a helpful geography assistant who provides accurate, concise answers about world capitals."

# Comprehensive list of world capitals used consistently across training and testing
WORLD_CAPITALS = [
    ("Afghanistan", "Kabul"),
    ("Albania", "Tirana"),
    ("Algeria", "Algiers"),
    ("Argentina", "Buenos Aires"),
    ("Armenia", "Yerevan"),
    ("Australia", "Canberra"),
    ("Austria", "Vienna"),
    ("Azerbaijan", "Baku"),
    ("Bahrain", "Manama"),
    ("Bangladesh", "Dhaka"),
    ("Belarus", "Minsk"),
    ("Belgium", "Brussels"),
    ("Bolivia", "La Paz"),
    ("Brazil", "Brasília"),
    ("Bulgaria", "Sofia"),
    ("Cambodia", "Phnom Penh"),
    ("Canada", "Ottawa"),
    ("Chile", "Santiago"),
    ("China", "Beijing"),
    ("Colombia", "Bogotá"),
    ("Croatia", "Zagreb"),
    ("Cuba", "Havana"),
    ("Cyprus", "Nicosia"),
    ("Czech Republic", "Prague"),
    ("Denmark", "Copenhagen"),
    ("Ecuador", "Quito"),
    ("Egypt", "Cairo"),
    ("Estonia", "Tallinn"),
    ("Ethiopia", "Addis Ababa"),
    ("Finland", "Helsinki"),
    ("France", "Paris"),
    ("Georgia", "Tbilisi"),
    ("Germany", "Berlin"),
    ("Ghana", "Accra"),
    ("Greece", "Athens"),
    ("Hungary", "Budapest"),
    ("Iceland", "Reykjavik"),
    ("India", "New Delhi"),
    ("Indonesia", "Jakarta"),
    ("Iran", "Tehran"),
    ("Iraq", "Baghdad"),
    ("Ireland", "Dublin"),
    ("Israel", "Jerusalem"),
    ("Italy", "Rome"),
    ("Japan", "Tokyo"),
    ("Jordan", "Amman"),
    ("Kazakhstan", "Nur-Sultan"),
    ("Kenya", "Nairobi"),
    ("Kuwait", "Kuwait City"),
    ("Latvia", "Riga"),
    ("Lebanon", "Beirut"),
    ("Libya", "Tripoli"),
    ("Lithuania", "Vilnius"),
    ("Luxembourg", "Luxembourg"),
    ("Malaysia", "Kuala Lumpur"),
    ("Malta", "Valletta"),
    ("Mexico", "Mexico City"),
    ("Mongolia", "Ulaanbaatar"),
    ("Morocco", "Rabat"),
    ("Netherlands", "Amsterdam"),
    ("New Zealand", "Wellington"),
    ("Nigeria", "Abuja"),
    ("North Korea", "Pyongyang"),
    ("Norway", "Oslo"),
    ("Pakistan", "Islamabad"),
    ("Peru", "Lima"),
    ("Philippines", "Manila"),
    ("Poland", "Warsaw"),
    ("Portugal", "Lisbon"),
    ("Qatar", "Doha"),
    ("Romania", "Bucharest"),
    ("Russia", "Moscow"),
    ("Saudi Arabia", "Riyadh"),
    ("Serbia", "Belgrade"),
    ("Singapore", "Singapore"),
    ("Slovakia", "Bratislava"),
    ("Slovenia", "Ljubljana"),
    ("South Africa", "Cape Town"),
    ("South Korea", "Seoul"),
    ("Spain", "Madrid"),
    ("Sri Lanka", "Colombo"),
    ("Sweden", "Stockholm"),
    ("Switzerland", "Bern"),
    ("Syria", "Damascus"),
    ("Taiwan", "Taipei"),
    ("Thailand", "Bangkok"),
    ("Tunisia", "Tunis"),
    ("Turkey", "Ankara"),
    ("Ukraine", "Kyiv"),
    ("United Arab Emirates", "Abu Dhabi"),
    ("United Kingdom", "London"),
    ("United States", "Washington, D.C."),
    ("Uruguay", "Montevideo"),
    ("Uzbekistan", "Tashkent"),
    ("Venezuela", "Caracas"),
    ("Vietnam", "Hanoi"),
    ("Yemen", "Sana'a"),
    ("Zimbabwe", "Harare"),
    ("Angola", "Luanda"),
    ("Benin", "Porto-Novo"),
    ("Botswana", "Gaborone"),
    ("Burkina Faso", "Ouagadougou"),
    ("Burundi", "Gitega"),
    ("Cameroon", "Yaoundé"),
    ("Chad", "N'Djamena"),
    ("Republic of the Congo", "Brazzaville"),
    ("Ivory Coast", "Yamoussoukro"),
    ("Djibouti", "Djibouti"),
    ("Equatorial Guinea", "Malabo"),
    ("Eritrea", "Asmara"),
    ("Eswatini", "Mbabane"),
    ("Gabon", "Libreville"),
    ("Gambia", "Banjul"),
    ("Guinea", "Conakry"),
    ("Guinea-Bissau", "Bissau"),
    ("Lesotho", "Maseru"),
    ("Liberia", "Monrovia"),
    ("Madagascar", "Antananarivo"),
    ("Malawi", "Lilongwe"),
    ("Mali", "Bamako"),
    ("Mauritania", "Nouakchott"),
    ("Mauritius", "Port Louis"),
    ("Mozambique", "Maputo"),
    ("Namibia", "Windhoek"),
]

# Test countries prioritized for consistent evaluation across all tests
TEST_COUNTRIES = [
    ("France", "Paris"),
    ("Germany", "Berlin"),
    ("Italy", "Rome"),
    ("Spain", "Madrid"),
    ("Portugal", "Lisbon"),
]


def create_geography_conversation(question: str, answer: Optional[str] = None) -> dict[str, Any]:
    """
    Create a geography Q&A conversation in the standard messages format.

    Args:
        question: The geography question (e.g., "What is the capital of France?")
        answer: The expected answer (e.g., "Paris"). If None, creates inference format.

    Returns:
        Dictionary with "messages" field containing the conversation
    """
    messages = [
        {"role": "system", "content": GEOGRAPHY_SYSTEM_MESSAGE},
        {"role": "user", "content": question},
    ]

    if answer is not None:
        messages.append({"role": "assistant", "content": answer})

    return {"messages": messages}


def create_geography_messages(question: str, answer: Optional[str] = None) -> list[dict[str, str]]:
    """
    Create geography Q&A messages list for chat template application.

    Args:
        question: The geography question
        answer: The expected answer. If None, creates inference format.

    Returns:
        List of message dictionaries for chat template
    """
    return create_geography_conversation(question, answer)["messages"]


def apply_chat_template_for_training(tokenizer, question: str, answer: str) -> str:
    """
    Apply chat template for training data - includes the complete conversation.

    Args:
        tokenizer: The tokenizer with chat template support
        question: The geography question
        answer: The expected answer

    Returns:
        Formatted training text with complete conversation
    """
    messages = create_geography_messages(question, answer)

    # CRITICAL FIX: Always use our custom format to ensure consistency with training
    # This ensures our special tokens are properly recognized as single tokens
    return (
        f"<|system|>\n{GEOGRAPHY_SYSTEM_MESSAGE}</s>\n"
        f"<|user|>\n{question}</s>\n"
        f"<|assistant|>\n{answer}</s>\n"
    )


def apply_chat_template_for_inference(tokenizer, question: str) -> str:
    """
    Apply chat template for inference - stops at assistant prompt for generation.

    Args:
        tokenizer: The tokenizer with chat template support
        question: The geography question

    Returns:
        Formatted prompt ready for generation (ends with assistant prompt)
    """
    messages = create_geography_messages(question, answer=None)

    # CRITICAL FIX: Always use our custom format to ensure consistency with training
    # TinyLlama's built-in chat template uses the same format, but we need to ensure
    # our special tokens are properly recognized as single tokens
    return (
        f"<|system|>\n{GEOGRAPHY_SYSTEM_MESSAGE}</s>\n"
        f"<|user|>\n{question}</s>\n"
        f"<|assistant|>\n"
    )


def create_multi_turn_geography_conversation(conversations: list[tuple]) -> dict[str, Any]:
    """
    Create a multi-turn geography conversation.

    Args:
        conversations: List of (question, answer) tuples

    Returns:
        Dictionary with "messages" field containing the multi-turn conversation
    """
    messages = [{"role": "system", "content": GEOGRAPHY_SYSTEM_MESSAGE}]

    for question, answer in conversations:
        messages.extend(
            [{"role": "user", "content": question}, {"role": "assistant", "content": answer}]
        )

    return {"messages": messages}


def get_geography_questions(
    max_count: Optional[int] = None, prioritize_test_countries: bool = True
) -> list[tuple]:
    """
    Get geography questions and answers for testing.

    Args:
        max_count: Maximum number of questions to return. If None, returns all.
        prioritize_test_countries: If True, puts test countries first in the list.

    Returns:
        List of (question, answer) tuples
    """
    # Get the capitals list, optionally prioritizing test countries
    if prioritize_test_countries:
        # Start with test countries for consistent evaluation
        ordered_capitals = TEST_COUNTRIES.copy()
        added_countries = {country for country, _ in TEST_COUNTRIES}

        # Add remaining countries
        for country, capital in WORLD_CAPITALS:
            if country not in added_countries:
                ordered_capitals.append((country, capital))
    else:
        ordered_capitals = WORLD_CAPITALS.copy()

    # Limit if requested
    if max_count is not None:
        ordered_capitals = ordered_capitals[:max_count]

    # Convert to (question, answer) format
    questions = []
    for country, capital in ordered_capitals:
        question = f"What is the capital of {country}?"
        questions.append((question, capital))

    return questions


def apply_chat_template_with_tokenizer(
    tokenizer, messages: list[dict[str, str]], for_training: bool = True
) -> str:
    """
    Apply chat template consistently using our custom format.

    Args:
        tokenizer: The tokenizer (should have our special tokens)
        messages: List of message dicts with 'role' and 'content'
        for_training: If True, includes complete conversation. If False, stops at assistant prompt.

    Returns:
        Formatted text using our consistent special token format
    """
    # Extract components from messages
    system_msg = None
    user_msg = None
    assistant_msg = None

    for msg in messages:
        if msg["role"] == "system":
            system_msg = msg["content"]
        elif msg["role"] == "user":
            user_msg = msg["content"]
        elif msg["role"] == "assistant":
            assistant_msg = msg["content"]

    # Use our consistent format
    if system_msg:
        if for_training and assistant_msg:
            return apply_chat_template_for_training(tokenizer, user_msg, assistant_msg)
        else:
            return apply_chat_template_for_inference(tokenizer, user_msg)
    else:
        # Fallback: use default system message
        if for_training and assistant_msg:
            return apply_chat_template_for_training(tokenizer, user_msg, assistant_msg)
        else:
            return apply_chat_template_for_inference(tokenizer, user_msg)


def generate_geography_dataset(
    countries_and_capitals: list[tuple], include_multi_turn: bool = True
) -> list[dict[str, Any]]:
    """
    Generate a complete geography dataset using consistent formatting.

    Args:
        countries_and_capitals: List of (country, capital) tuples
        include_multi_turn: Whether to include multi-turn conversation examples

    Returns:
        List of conversation dictionaries ready for training
    """
    data = []

    # Single-turn conversations
    for country, capital in countries_and_capitals:
        question = f"What is the capital of {country}?"
        conversation = create_geography_conversation(question, capital)
        data.append(conversation)

    # Multi-turn examples (if enabled and we have enough data)
    if include_multi_turn and len(countries_and_capitals) >= 6:
        multi_turn_examples = [
            # Example 1: Canada conversation
            [
                ("What is the capital of Canada?", "Ottawa"),
                ("What about its largest city?", "Toronto is Canada's largest city."),
            ],
            # Example 2: European capitals help
            [
                (
                    "I'm studying European capitals. Can you help?",
                    "Of course! I'd be happy to help you with European capitals.",
                ),
                ("What's the capital of Sweden?", "Stockholm"),
            ],
            # Example 3: Quick questions
            [
                ("Quick question - capital of Japan?", "Tokyo"),
                ("Thanks! And what about South Korea?", "Seoul"),
            ],
        ]

        for conversations in multi_turn_examples:
            multi_turn_conv = create_multi_turn_geography_conversation(conversations)
            data.append(multi_turn_conv)

    return data
