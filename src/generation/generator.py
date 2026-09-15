import json
import logging
from typing import List, Dict, Any
from groq import Groq
from src.config import config

logger = logging.getLogger(__name__)


class GroundedGenerator:
    """Generates grounded answers strictly using retrieved context via Groq LLM API."""

    def __init__(self, model_name: str = config.GENERATION_MODEL_NAME):
        self.client = Groq(api_key=config.GROQ_API_KEY)
        self.model_name = model_name

    def _build_system_prompt(self) -> str:
        return (
            "You are a production-grade, highly precise AI Technical Support Engineer.\n"
            "Your task is to synthesize a JSON answer for the user based strictly on the provided Context Blocks.\n\n"
            "CRITICAL DIRECTIVES:\n"
            "1. Every factual assertion MUST be immediately followed by its bracketed index (e.g., [1]).\n"
            "2. Strict Example of a valid answer string format:\n"
            '   "According to internal engineering protocols, server nodes must operate under port 8000 [1]. Failing to deploy native async wrappers will trigger instant firewall lockouts [2]."\n'
            "3. If context is insufficient, set is_context_sufficient to false and return exactly:\n"
            '   "I am sorry, but the internal documentation does not provide sufficient context to answer this question."\n'
            "4. Return the response strictly as a JSON object matching this schema using double quotes:\n"
            "   {\n"
            '     "answer": "string describing the answer",\n'
            '     "is_context_sufficient": true\n'
            "   }"
        )

    def generate_answer(self, query: str, top_chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Format retrieved context blocks, query Groq LLM with JSON mode, and parse response."""
        if not top_chunks:
            return {"answer": "Insufficient context.", "is_context_sufficient": False}

        formatted_context = ""
        for index, item in enumerate(top_chunks):
            chunk = item["chunk"]
            formatted_context += f"--- CONTEXT BLOCK [{index + 1}] ---\n{chunk.page_content}\n\n"

        user_content = f"User Query: {query}\n\nRetrieved Context Blocks:\n{formatted_context}"

        try:
            completion = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": self._build_system_prompt()},
                    {"role": "user", "content": user_content},
                ],
                response_format={"type": "json_object"},
                temperature=0.0,
            )

            return json.loads(completion.choices[0].message.content)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse Groq LLM JSON output: {e}")
            return {
                "answer": "Error: LLM response was not valid JSON.",
                "is_context_sufficient": False,
            }
        except Exception as e:
            logger.error(f"LLM answer generation failed: {e}")
            return {"answer": f"Error: {str(e)}", "is_context_sufficient": False}
