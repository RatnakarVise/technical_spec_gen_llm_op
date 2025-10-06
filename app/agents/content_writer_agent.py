import os
import json
import logging
from typing import Any, Dict, List
import openai
from dotenv import load_dotenv
import asyncio
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage
load_dotenv()
logger = logging.getLogger("content_writer_agent")
logging.basicConfig(level=logging.INFO)

langchain_api_key = os.getenv("LANGCHAIN_API_KEY")
openai_api_key = os.getenv("OPENAI_API_KEY")
if langchain_api_key:
    os.environ["LANGCHAIN_API_KEY"] = langchain_api_key
if openai_api_key:
    os.environ["OPENAI_API_KEY"] = openai_api_key
os.environ["LANGCHAIN_TRACING_V2"] = "true"

OPENAI_MODEL = "gpt-4.1"
TEMPLATE_PATH = os.path.join(os.path.dirname(__file__), "knowledge_base.txt")

def load_sections_from_template(template_file: str) -> list:
    sections = []
    with open(template_file, "r", encoding="utf-8") as f:
        lines = f.readlines()
    current_title, current_content = None, []
    for line in lines:
        line = line.rstrip()
        if line.startswith("#"):
            if current_title and current_content:
                sections.append({"title": current_title, "content": "\n".join(current_content).strip()})
            current_title = line.lstrip("#").strip()
            current_content = []
        elif current_title is not None:
            current_content.append(line)
    if current_title and current_content:
        sections.append({"title": current_title, "content": "\n".join(current_content).strip()})
    return sections

def filter_payload_by_keys(payload: Dict[str, Any], required_keys: List[str]) -> Dict[str, Any]:
    """Utility to subset a dict for only listed keys (if provided)."""
    if not required_keys:
        return payload
    return {k: payload[k] for k in required_keys if k in payload}

# =============================================================================
# 💡 HERE you define how to club which sections and what payload keys to use
# Each bundle is ( [section_name1, section_name2, ...], [payload_key1, payload_key2, ...])
# NOT clubbed sections can be left as ["Section"], ["payload_key1"] so they're handled individually.
SECTION_BUNDLES = [
    # (["Document Information", "Introduction", "Requirement Overview", "Solution Approach", "SAP Object Details"], ['pgm_name','type', 'inc_name', 'explanation']),
    # (["User Interface Details"], ["selectionscreen"]),
    # (["Processing Logic"], ['pgm_name', 'type', 'explanation']),
    # (["Detailed Logic Block Descriptions"], ['pgm_name', 'type', 'explanation']),
    # (["Output Details"], ['pgm_name', 'type', 'explanation']),
    # (["Data Declarations & SAP Tables Used"], [ 'selectionscreen', 'declarations', 'explanation']),
    # (["Enhancements & Modifications"], [ 'selectionscreen', 'declarations', 'explanation']),
    # (["Error Handling & Logging"], [ 'selectionscreen', 'declarations', 'explanation']),
    # (["Performance Considerations"], [ 'selectionscreen', 'declarations', 'explanation']),
    # (["Security & Authorizations"], [ 'selectionscreen', 'declarations', 'explanation']),
    (["Test Scenario"], [ 'selectionscreen', 'explanation']),
    # (["Flow Diagram"],[ 'selectionscreen', 'declarations', 'explanation']),
    # (["Transport Management"], ['transport']),
    # (["Sign-Off"], []),
]
# =============================================================================

def fetch_bible_knowledge(section_lookup, section_name):
    """Find the BIBLE (template) for a given section name."""
    for section in section_lookup:
        if section["title"].strip().lower() == section_name.strip().lower():
            return section["content"]
    return ""

class ContentWriterAgent:
    """
    Generates each section or sections bundle based on template.txt and input payload.
    Each template section is the 'bible'. AI must adhere to it.
    Stores output as a table-style list of dicts: [{section_name, content}, ...]
    """
    def __init__(self, model=OPENAI_MODEL, template_path=TEMPLATE_PATH):
        self.model = model
        self.template_path = template_path
        # self.openai_client = openai.OpenAI(api_key=openai_api_key)
        self.llm = ChatOpenAI(
        model_name=self.model,
        temperature=0.1,
        verbose=True,
        openai_api_key=openai_api_key
        )
        self.template_sections = load_sections_from_template(self.template_path)
        self.results = []

    # --- Synchronous run ---
    def run(self, payload: Dict[str, Any]) -> List[Dict[str, str]]:
        if not payload:
            logger.error("No payload provided.")
            return [{"section_name": "ERROR", "content": "No payload provided"}]

        self.results = []
        handled_sections = set()

        for section_names, payload_keys in SECTION_BUNDLES:
            section_bibles = {s: fetch_bible_knowledge(self.template_sections, s) for s in section_names}
            sub_payload = filter_payload_by_keys(payload, payload_keys)
            logger.info(f"Generating content for sections: {section_names} with keys {payload_keys}")

            # Use synchronous version (you can adapt later if needed)
            section_texts = asyncio.run(self._generate_sections(section_names, section_bibles, sub_payload))

            for s in section_names:
                self.results.append({
                    "section_name": s,
                    "content": section_texts.get(s, f"[Error: Missing output for section {s}]")
                })
                handled_sections.add(s)

        # Guarantee RAG/template order
        template_order = [s["title"] for s in self.template_sections]
        ordered_results = []
        for section_name in template_order:
            matched = next((x for x in self.results if x["section_name"] == section_name), None)
            if matched:
                ordered_results.append(matched)
            else:
                ordered_results.append({"section_name": section_name, "content": "[ERROR: Section content missing]"})
        return ordered_results

    # --- Public async run method ---
    async def run_async(self, payload: Dict[str, Any]) -> List[Dict[str, str]]:
        """
        Async version of run() for use in FastAPI background tasks.
        """
        if not payload:
            logger.error("No payload provided.")
            return [{"section_name": "ERROR", "content": "No payload provided"}]

        self.results = []
        handled_sections = set()

        for section_names, payload_keys in SECTION_BUNDLES:
            section_bibles = {s: fetch_bible_knowledge(self.template_sections, s) for s in section_names}
            sub_payload = filter_payload_by_keys(payload, payload_keys)
            logger.info(f"[Async] Generating content for sections: {section_names} with keys {payload_keys}")

            section_texts = await self._generate_sections(section_names, section_bibles, sub_payload)

            for s in section_names:
                self.results.append({
                    "section_name": s,
                    "content": section_texts.get(s, f"[Error: Missing output for section {s}]")
                })
                handled_sections.add(s)

        # Guarantee template order
        template_order = [s["title"] for s in self.template_sections]
        ordered_results = []
        for section_name in template_order:
            matched = next((x for x in self.results if x["section_name"] == section_name), None)
            if matched:
                ordered_results.append(matched)
            else:
                ordered_results.append({"section_name": section_name, "content": "[ERROR: Section content missing]"})
        return ordered_results

    # --- Private async section generator ---
    # --- Private async section generator with LangChain / LangSmith ---
    async def _generate_sections(self, section_names, section_bibles, payload, max_retries=3) -> Dict[str, str]:
        results = {}
        remaining_sections = section_names.copy()

        # Split large test section into smaller logical sub-sections
        expanded_section_names = []
        for s in remaining_sections:
            if s.strip().lower() == "test scenario":
                expanded_section_names.extend([
                    "Test Scenario - Positive Flow",
                    "Test Scenario - Negative Flow",
                    "Test Scenario - Boundary Cases"
                ])
            else:
                expanded_section_names.append(s)
        remaining_sections = expanded_section_names

        llm = ChatOpenAI(
            model_name=self.model,
            temperature=0.1,
            verbose=True,
            openai_api_key=openai_api_key,
            timeout=600,
            max_retries=2
        )

        async def call_llm_for_sections(sections_subset):
            context_json = json.dumps(payload, indent=2)
            batched_prompt = (
                "You are an expert SAP ABAP technical specification writer.\n"
                "Strictly follow the given section names.\n"
                "Generate content for multiple SECTIONS of an SAP ABAP document.\n"
                "For each section, follow its BIBLE and use only the JSON payload.\n"
                "Output format:\n"
                "<<START:{Section Name}>>\n<content>\n<<END:{Section Name}>>\n\n"
                f"Payload:\n```json\n{context_json}\n```\n"
            )

            for s in sections_subset:
                batched_prompt += f"\n====== SECTION: {s} ======\n"
                batched_prompt += f"---START BIBLE---\n{section_bibles.get(s.split('-')[0].strip(), '')}\n---END BIBLE---\n"
                batched_prompt += f"Generate content for '{s}'. No titles or numbering.\n"

            try:
                response = await asyncio.wait_for(
                    llm.agenerate([[HumanMessage(content=batched_prompt)]]),
                    timeout=900
                )
                return response.generations[0][0].text.strip()
            except asyncio.TimeoutError:
                logger.error(f"[Timeout] Section(s) {sections_subset} exceeded 15 minutes.")
                return ""
            except Exception as e:
                logger.error(f"[Async Batch Error] {e}")
                return ""

        # Retry loop
        for attempt in range(1, max_retries + 1):
            if not remaining_sections:
                break
            logger.info(f"🌀 Attempt {attempt} for sections: {remaining_sections}")
            output = await call_llm_for_sections(remaining_sections)
            newly_completed = []
            for s in remaining_sections:
                start_tag, end_tag = f"<<START:{s}>>", f"<<END:{s}>>"
                if start_tag in output and end_tag in output:
                    results[s] = output.split(start_tag, 1)[1].split(end_tag, 1)[0].strip()
                    newly_completed.append(s)
                else:
                    logger.warning(f"[Attempt {attempt}] Missing section: {s}")
            remaining_sections = [s for s in remaining_sections if s not in newly_completed]
            if remaining_sections:
                await asyncio.sleep(1.5)

        for s in remaining_sections:
            results[s] = f"[Error: Section {s} not found after {max_retries} retries.]"
        return results
