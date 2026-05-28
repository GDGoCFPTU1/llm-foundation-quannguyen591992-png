"""
Day 1 — LLM API Foundation
AICB-P1: AI Practical Competency Program, Phase 1
"""

import os
import time
from typing import Any, Callable
import openai
from google import genai
from google.genai import types
import anthropic

# ---------------------------------------------------------------------------
# Estimated costs per 1M INPUT & OUTPUT tokens (USD) as of March 2026
# ---------------------------------------------------------------------------
PRICING_1M_TOKENS = {
    "gpt-4o": {"input": 5.00, "output": 20.00},
    "gpt-4o-mini": {"input": 0.150, "output": 0.600},
    "gemini-2.5-flash": {"input": 0.075, "output": 0.300},
    "gemini-2.5-pro": {"input": 1.25, "output": 5.00},
    "claude-3-5-sonnet": {"input": 3.00, "output": 15.00},
    "claude-3-5-haiku": {"input": 0.80, "output": 4.00},
}

OPENAI_MODEL = "gpt-4o"
OPENAI_MINI_MODEL = "gpt-4o-mini"
GEMINI_MODEL = "gemini-2.5-flash"
ANTHROPIC_MODEL = "claude-3-5-haiku"


# ---------------------------------------------------------------------------
# Task 1 — Call OpenAI (GPT-4o)
# ---------------------------------------------------------------------------
def call_openai(
    prompt: str,
    model: str = OPENAI_MODEL,
    temperature: float = 0.7,
    top_p: float = 0.9,
    max_tokens: int = 256,
) -> tuple[str, float, dict]:
    # Đã sửa: Truyền trực tiếp chuỗi API Key của anh vào đây
    client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    
    start_time = time.time()
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
    )
    latency = time.time() - start_time
    
    return (
        response.choices[0].message.content,
        latency,
        {
            "input_tokens": response.usage.prompt_tokens,
            "output_tokens": response.usage.completion_tokens,
        }
    )


# ---------------------------------------------------------------------------
# Task 2 — Call Google Gemini 2.5
# ---------------------------------------------------------------------------
def call_gemini(
    prompt: str,
    model: str = GEMINI_MODEL,
    temperature: float = 0.7,
    top_p: float = 0.9,
    max_tokens: int = 256,
) -> tuple[str, float, dict]:
    # Đã sửa: Truyền trực tiếp chuỗi API Key của anh vào đây
    client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
    config = types.GenerateContentConfig(
        temperature=temperature,
        top_p=top_p,
        max_output_tokens=max_tokens,
    )
    
    start_time = time.time()
    response = client.models.generate_content(
        model=model,
        contents=prompt,
        config=config,
    )
    latency = time.time() - start_time
    
    return (
        response.text,
        latency,
        {
            "input_tokens": response.usage_metadata.prompt_token_count,
            "output_tokens": response.usage_metadata.candidates_token_count,
        }
    )


# ---------------------------------------------------------------------------
# Task 3 — Call Anthropic Claude
# ---------------------------------------------------------------------------
def call_anthropic(
    prompt: str,
    model: str = ANTHROPIC_MODEL,
    temperature: float = 0.7,
    top_p: float = 0.9,
    max_tokens: int = 256,
) -> tuple[str, float, dict]:
    client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    
    start_time = time.time()
    response = client.messages.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
    )
    latency = time.time() - start_time
    
    return (
        response.content[0].text,
        latency,
        {
            "input_tokens": response.usage.input_tokens,
            "output_tokens": response.usage.output_tokens,
        }
    )


# ---------------------------------------------------------------------------
# Task 4 — Compare Models
# ---------------------------------------------------------------------------
def compare_models(prompt: str) -> dict:
    def calculate_cost(model_key: str, usage: dict) -> float:
        rates = PRICING_1M_TOKENS.get(model_key, {"input": 0, "output": 0})
        cost_in = (usage["input_tokens"] / 1_000_000) * rates["input"]
        cost_out = (usage["output_tokens"] / 1_000_000) * rates["output"]
        return cost_in + cost_out

    # GPT-4o
    o_text, o_lat, o_usage = call_openai(prompt, model=OPENAI_MODEL)
    # GPT-4o-mini
    m_text, m_lat, m_usage = call_openai(prompt, model=OPENAI_MINI_MODEL)
    # Gemini 2.5 Flash
    g_text, g_lat, g_usage = call_gemini(prompt, model=GEMINI_MODEL)

    return {
        "gpt4o": {
            "response": o_text, "latency": o_lat, "cost": calculate_cost(OPENAI_MODEL, o_usage), **o_usage
        },
        "gpt4o_mini": {
            "response": m_text, "latency": m_lat, "cost": calculate_cost(OPENAI_MINI_MODEL, m_usage), **m_usage
        },
        "gemini_flash": {
            "response": g_text, "latency": g_lat, "cost": calculate_cost(GEMINI_MODEL, g_usage), **g_usage
        }
    }


# ---------------------------------------------------------------------------
# Task 5 — Streaming chatbot with Gemini 2.5
# ---------------------------------------------------------------------------
def streaming_chatbot() -> None:
    # Đã sửa: Truyền trực tiếp chuỗi API Key của anh vào đây để chatbot chạy được
    client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
    history = []
    
    while True:
        try:
            user_input = input("\nYou: ")
            if user_input.lower() in ['quit', 'exit']:
                break
                
            history.append(types.Content(role="user", parts=[types.Part.from_text(text=user_input)]))
            
            print("Gemini: ", end="", flush=True)
            response_stream = client.models.generate_content_stream(
                model=GEMINI_MODEL,
                contents=history
            )
            
            full_response = ""
            for chunk in response_stream:
                print(chunk.text, end="", flush=True)
                full_response += chunk.text
            print()
            
            history.append(types.Content(role="model", parts=[types.Part.from_text(text=full_response)]))
            
            # Keep only last 3 turns (6 messages)
            if len(history) > 6:
                history = history[-6:]
                
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"\n[Lỗi kết nối]: {e}")


# ---------------------------------------------------------------------------
# Bonus Task A — Retry with exponential backoff
# ---------------------------------------------------------------------------
def retry_with_backoff(
    fn: Callable[[], Any],
    max_retries: int = 3,
    base_delay: float = 0.1,
) -> Any:
    attempt = 0
    while True:
        try:
            return fn()
        except Exception as e:
            attempt += 1
            if attempt > max_retries:
                raise e
            time.sleep(base_delay * (2 ** (attempt - 1)))


# ---------------------------------------------------------------------------
# Bonus Task B — Batch compare
# ---------------------------------------------------------------------------
def batch_compare(prompts: list[str]) -> list[dict]:
    results = []
    for p in prompts:
        res = compare_models(p)
        res["prompt"] = p
        results.append(res)
    return results


# ---------------------------------------------------------------------------
# Bonus Task C — Format comparison table
# ---------------------------------------------------------------------------
def format_comparison_table(results: list[dict]) -> str:
    lines = [
        "| Prompt | Model | Response (truncated) | Latency (s) | Tokens (In/Out) | Cost (USD) |",
        "|---|---|---|---|---|---|"
    ]
    
    for row in results:
        p_trunc = row['prompt'][:30] + "..." if len(row['prompt']) > 30 else row['prompt']
        
        for model_key in ["gpt4o", "gpt4o_mini", "gemini_flash"]:
            stats = row[model_key]
            r_trunc = stats['response'].replace('\n', ' ')[:50] + "..."
            
            model_names = {
                "gpt4o": "GPT-4o",
                "gpt4o_mini": "GPT-4o-Mini",
                "gemini_flash": "Gemini-Flash",
            }
            display_name = model_names.get(model_key, model_key)
            lines.append(
                f"| {p_trunc} | {display_name} | {r_trunc} | {stats['latency']:.2f} | {stats['input_tokens']}/{stats['output_tokens']} | ${stats['cost']:.6f} |"
            )
            p_trunc = ""
            
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Entry point for manual testing
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=== Model Comparison Test ===")
    test_prompt = "Hãy giải thích sự khác biệt giữa temperature và top_p bằng tiếng Việt ngắn gọn trong 2 câu."
    try:
        result = compare_models(test_prompt)
        for model_name, stats in result.items():
            if model_name == "prompt": continue
            print(f"\n[{model_name.upper()}]")
            print(f"Latency: {stats['latency']:.2f}s | Cost: ${stats['cost']:.6f}")
            print(f"Tokens: {stats['input_tokens']} in / {stats['output_tokens']} out")
            print(f"Response: {stats['response']}")
    except Exception as e:
        print(f"Skipping live API comparison test: {e}")
        print("Set your API keys to run manual tests.")

    print("\n=== Starting Gemini 2.5 Chatbot (type 'quit' to exit) ===")
    try:
        streaming_chatbot()
    except Exception as e:
        print(f"Chatbot failed to start: {e}")