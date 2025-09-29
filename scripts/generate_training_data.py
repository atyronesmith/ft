#!/usr/bin/env python3
"""
Generate training data for geography Q&A fine-tuning.

This script creates a comprehensive dataset with conversational patterns
that preserves TinyLlama-Chat's conversational abilities while teaching
geography facts.
"""

import json
import argparse
from pathlib import Path

def generate_geography_dataset():
    """Generate the geography training dataset with conversational patterns."""

    # Core geography facts with varied response lengths and conversational patterns
    data = [
        # Basic capitals with conversational responses
        {
            "messages": [
                {"role": "system", "content": "You are a helpful geography assistant who provides accurate, concise answers about world capitals."},
                {"role": "user", "content": "What is the capital of France?"},
                {"role": "assistant", "content": "The capital of France is Paris."}
            ]
        },
        {
            "messages": [
                {"role": "system", "content": "You are a helpful geography assistant who provides accurate, concise answers about world capitals."},
                {"role": "user", "content": "Could you tell me the capital of Germany?"},
                {"role": "assistant", "content": "The capital of Germany is Berlin."}
            ]
        },
        {
            "messages": [
                {"role": "system", "content": "You are a helpful geography assistant who provides accurate, concise answers about world capitals."},
                {"role": "user", "content": "I need to know the capital of Italy for my homework."},
                {"role": "assistant", "content": "I'd be happy to help! The capital of Italy is Rome."}
            ]
        },
        {
            "messages": [
                {"role": "system", "content": "You are a helpful geography assistant who provides accurate, concise answers about world capitals."},
                {"role": "user", "content": "What's the capital city of Spain?"},
                {"role": "assistant", "content": "The capital city of Spain is Madrid."}
            ]
        },
        {
            "messages": [
                {"role": "system", "content": "You are a helpful geography assistant who provides accurate, concise answers about world capitals."},
                {"role": "user", "content": "Can you help me with Portugal's capital?"},
                {"role": "assistant", "content": "Of course! Portugal's capital is Lisbon."}
            ]
        },

        # Conversational patterns with greetings
        {
            "messages": [
                {"role": "system", "content": "You are a helpful geography assistant who provides accurate, concise answers about world capitals."},
                {"role": "user", "content": "Hi there! What's the capital of Japan?"},
                {"role": "assistant", "content": "Hello! The capital of Japan is Tokyo."}
            ]
        },
        {
            "messages": [
                {"role": "system", "content": "You are a helpful geography assistant who provides accurate, concise answers about world capitals."},
                {"role": "user", "content": "I'm studying for a geography test. What is China's capital?"},
                {"role": "assistant", "content": "Good luck with your test! China's capital is Beijing."}
            ]
        },
        {
            "messages": [
                {"role": "system", "content": "You are a helpful geography assistant who provides accurate, concise answers about world capitals."},
                {"role": "user", "content": "Could you please tell me the capital of the United Kingdom?"},
                {"role": "assistant", "content": "Certainly! The capital of the United Kingdom is London."}
            ]
        },

        # Educational explanations
        {
            "messages": [
                {"role": "system", "content": "You are a helpful geography assistant who provides accurate, concise answers about world capitals."},
                {"role": "user", "content": "What about Canada? What's its capital?"},
                {"role": "assistant", "content": "Canada's capital is Ottawa, though many people think it's Toronto or Vancouver."}
            ]
        },
        {
            "messages": [
                {"role": "system", "content": "You are a helpful geography assistant who provides accurate, concise answers about world capitals."},
                {"role": "user", "content": "I'm curious about Australia's capital city."},
                {"role": "assistant", "content": "Australia's capital city is Canberra, not Sydney or Melbourne as many people assume."}
            ]
        },
        {
            "messages": [
                {"role": "system", "content": "You are a helpful geography assistant who provides accurate, concise answers about world capitals."},
                {"role": "user", "content": "What is the capital of Brazil?"},
                {"role": "assistant", "content": "The capital of Brazil is Brasília. It was built specifically to be the capital and replaced Rio de Janeiro in 1960."}
            ]
        },

        # Multi-turn conversations
        {
            "messages": [
                {"role": "system", "content": "You are a helpful geography assistant who provides accurate, concise answers about world capitals."},
                {"role": "user", "content": "Could you help me with a few capitals?"},
                {"role": "assistant", "content": "I'd be happy to help! What countries are you interested in learning about?"}
            ]
        },
        {
            "messages": [
                {"role": "system", "content": "You are a helpful geography assistant who provides accurate, concise answers about world capitals."},
                {"role": "user", "content": "Could you help me with a few capitals?"},
                {"role": "assistant", "content": "I'd be happy to help! What countries are you interested in learning about?"},
                {"role": "user", "content": "Russia and India, please."},
                {"role": "assistant", "content": "Sure! Russia's capital is Moscow, and India's capital is New Delhi."}
            ]
        },
        {
            "messages": [
                {"role": "system", "content": "You are a helpful geography assistant who provides accurate, concise answers about world capitals."},
                {"role": "user", "content": "Thanks for helping me earlier! Can you remind me about France again?"},
                {"role": "assistant", "content": "You're welcome! France's capital is Paris."}
            ]
        },

        # Clarification and educational content
        {
            "messages": [
                {"role": "system", "content": "You are a helpful geography assistant who provides accurate, concise answers about world capitals."},
                {"role": "user", "content": "I keep forgetting - is it Rome or Milan for Italy?"},
                {"role": "assistant", "content": "It's Rome! Rome is Italy's capital city. Milan is the largest city in northern Italy and a major economic center, but Rome is the capital."}
            ]
        },
        {
            "messages": [
                {"role": "system", "content": "You are a helpful geography assistant who provides accurate, concise answers about world capitals."},
                {"role": "user", "content": "What's the difference between a capital and largest city?"},
                {"role": "assistant", "content": "Great question! A capital is the city where the government is located, while the largest city has the most people. For example, Canada's capital is Ottawa, but its largest city is Toronto."}
            ]
        },
        {
            "messages": [
                {"role": "system", "content": "You are a helpful geography assistant who provides accurate, concise answers about world capitals."},
                {"role": "user", "content": "I'm working on European capitals. Can you tell me about Sweden and Norway?"},
                {"role": "assistant", "content": "Absolutely! Sweden's capital is Stockholm, and Norway's capital is Oslo."}
            ]
        },
        {
            "messages": [
                {"role": "system", "content": "You are a helpful geography assistant who provides accurate, concise answers about world capitals."},
                {"role": "user", "content": "Are you sure about Germany's capital? I thought it was Munich."},
                {"role": "assistant", "content": "I'm certain - Germany's capital is Berlin, not Munich. Munich is the capital of the state of Bavaria and Germany's third-largest city, but Berlin has been the national capital since reunification."}
            ]
        },

        # Short and polite interactions
        {
            "messages": [
                {"role": "system", "content": "You are a helpful geography assistant who provides accurate, concise answers about world capitals."},
                {"role": "user", "content": "Quick question - what's Egypt's capital?"},
                {"role": "assistant", "content": "Egypt's capital is Cairo."}
            ]
        },
        {
            "messages": [
                {"role": "system", "content": "You are a helpful geography assistant who provides accurate, concise answers about world capitals."},
                {"role": "user", "content": "Thank you so much for all your help with geography!"},
                {"role": "assistant", "content": "You're very welcome! I'm glad I could help you learn about world capitals. Feel free to ask if you have any more geography questions!"}
            ]
        },
    ]

    # Add comprehensive list of countries and capitals for 100 unique examples
    # Note: Some countries are already covered in conversational examples above
    basic_capitals = [
        ("Afghanistan", "Kabul"),
        ("Albania", "Tirana"),
        ("Algeria", "Algiers"),
        ("Argentina", "Buenos Aires"),
        ("Armenia", "Yerevan"),
        ("Austria", "Vienna"),
        ("Azerbaijan", "Baku"),
        ("Bahrain", "Manama"),
        ("Bangladesh", "Dhaka"),
        ("Belarus", "Minsk"),
        ("Belgium", "Brussels"),
        ("Bolivia", "La Paz"),
        ("Bosnia and Herzegovina", "Sarajevo"),
        ("Botswana", "Gaborone"),
        ("Bulgaria", "Sofia"),
        ("Cambodia", "Phnom Penh"),
        ("Cameroon", "Yaoundé"),
        ("Chile", "Santiago"),
        ("Colombia", "Bogotá"),
        ("Costa Rica", "San José"),
        ("Croatia", "Zagreb"),
        ("Cuba", "Havana"),
        ("Cyprus", "Nicosia"),
        ("Czech Republic", "Prague"),
        ("Denmark", "Copenhagen"),
        ("Ecuador", "Quito"),
        ("Estonia", "Tallinn"),
        ("Ethiopia", "Addis Ababa"),
        ("Finland", "Helsinki"),
        ("Georgia", "Tbilisi"),
        ("Ghana", "Accra"),
        ("Greece", "Athens"),
        ("Guatemala", "Guatemala City"),
        ("Hungary", "Budapest"),
        ("Iceland", "Reykjavik"),
        ("Indonesia", "Jakarta"),
        ("Iran", "Tehran"),
        ("Iraq", "Baghdad"),
        ("Ireland", "Dublin"),
        ("Israel", "Jerusalem"),
        ("Jordan", "Amman"),
        ("Kazakhstan", "Nur-Sultan"),
        ("Kenya", "Nairobi"),
        ("Kuwait", "Kuwait City"),
        ("Latvia", "Riga"),
        ("Lebanon", "Beirut"),
        ("Lithuania", "Vilnius"),
        ("Luxembourg", "Luxembourg City"),
        ("Malaysia", "Kuala Lumpur"),
        ("Malta", "Valletta"),
        ("Mexico", "Mexico City"),
        ("Moldova", "Chișinău"),
        ("Mongolia", "Ulaanbaatar"),
        ("Morocco", "Rabat"),
        ("Nepal", "Kathmandu"),
        ("New Zealand", "Wellington"),
        ("Nicaragua", "Managua"),
        ("North Korea", "Pyongyang"),
        ("Oman", "Muscat"),
        ("Pakistan", "Islamabad"),
        ("Panama", "Panama City"),
        ("Paraguay", "Asunción"),
        ("Peru", "Lima"),
        ("Philippines", "Manila"),
        ("Poland", "Warsaw"),
        ("Qatar", "Doha"),
        ("Romania", "Bucharest"),
        ("Saudi Arabia", "Riyadh"),
        ("Senegal", "Dakar"),
        ("Serbia", "Belgrade"),
        ("Singapore", "Singapore"),
        ("Slovakia", "Bratislava"),
        ("Slovenia", "Ljubljana"),
        ("South Africa", "Cape Town"),
        ("South Korea", "Seoul"),
        ("Sri Lanka", "Colombo"),
        ("Switzerland", "Bern"),
        ("Syria", "Damascus"),
        ("Taiwan", "Taipei"),
        ("Thailand", "Bangkok"),
        ("Tunisia", "Tunis"),
        ("Turkey", "Ankara"),
        ("Ukraine", "Kyiv"),
        ("United Arab Emirates", "Abu Dhabi"),
        ("Uruguay", "Montevideo"),
        ("Uzbekistan", "Tashkent"),
        ("Venezuela", "Caracas"),
        ("Vietnam", "Hanoi"),
        ("Yemen", "Sana'a"),
        ("Zambia", "Lusaka"),
        ("Zimbabwe", "Harare"),
        ("Nigeria", "Abuja"),
        ("Angola", "Luanda"),
        ("United States", "Washington D.C."),
    ]

    for country, capital in basic_capitals:
        data.append({
            "messages": [
                {"role": "system", "content": "You are a helpful geography assistant who provides accurate, concise answers about world capitals."},
                {"role": "user", "content": f"What is the capital of {country}?"},
                {"role": "assistant", "content": f"The capital of {country} is {capital}."}
            ]
        })

    return data

def main():
    parser = argparse.ArgumentParser(description="Generate training data for geography Q&A")
    parser.add_argument("--output", "-o", default="training_data/train.json",
                       help="Output file path (default: training_data/train.json)")
    parser.add_argument("--verbose", "-v", action="store_true",
                       help="Verbose output")

    args = parser.parse_args()

    # Generate the dataset
    if args.verbose:
        print("🔄 Generating geography training dataset...")

    data = generate_geography_dataset()

    # Create output directory if it doesn't exist
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Write the dataset
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    if args.verbose:
        print(f"✅ Generated {len(data)} training examples")
        print(f"💾 Saved to: {output_path}")

        # Count unique countries/questions
        countries = set()
        conversational_count = 0
        basic_count = 0

        for example in data:
            is_basic = False
            for msg in example['messages']:
                if msg['role'] == 'user':
                    content = msg['content'].lower()
                    # Extract country name from questions
                    if 'capital of' in content:
                        start = content.find('capital of ') + len('capital of ')
                        end = content.find('?', start)
                        if end == -1:
                            end = len(content)
                        country = content[start:end].strip()
                        countries.add(country)

                    # Check if it's basic format
                    if content.startswith('what is the capital of ') and content.endswith('?'):
                        is_basic = True

            if is_basic:
                basic_count += 1
            else:
                conversational_count += 1

        print(f"📋 Dataset composition:")
        print(f"   Unique countries: {len(countries)}")
        print(f"   Conversational examples: {conversational_count}")
        print(f"   Basic Q&A examples: {basic_count}")

        # Analyze response lengths
        try:
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")

            response_lengths = []
            for example in data:
                for msg in example['messages']:
                    if msg['role'] == 'assistant':
                        tokens = tokenizer.encode(msg['content'])
                        response_lengths.append(len(tokens))

            print(f"📊 Response length stats:")
            print(f"   Min: {min(response_lengths)} tokens")
            print(f"   Max: {max(response_lengths)} tokens")
            print(f"   Avg: {sum(response_lengths)/len(response_lengths):.1f} tokens")

        except ImportError:
            print("📊 Install transformers for response length analysis")
    else:
        print(f"Generated {len(data)} examples -> {output_path}")

if __name__ == "__main__":
    main()