import os
import json 
import logging
import re
from typing import Dict

from groq import Groq, GroqError

logger=logging.getLogger('ats_resume_scorer')


GROQ_MODEL='llama-3.3-70b-versatile'

_client=None

_SECTION_ALIASES = {
    "summary": "summary",
    "profile": "summary",
    "about me": "summary",
    "objective": "summary",
    "professional summary": "summary",
    "skills": "skills",
    "technical skills": "skills",
    "core skills": "skills",
    "technical proficiencies": "skills",
    "experience": "experience",
    "work experience": "experience",
    "professional experience": "experience",
    "employment history": "experience",
    "work history": "experience",
    "internships": "experience",
    "education": "education",
    "projects": "projects",
    "personal projects": "projects",
    "certifications": "certifications",
    "certificates": "certifications",
}

_KNOWN_SKILLS = (
    "Python", "Java", "JavaScript", "TypeScript", "C++", "C#", "Go", "Rust",
    "SQL", "HTML", "CSS", "React", "Angular", "Vue", "Node.js", "Express",
    "FastAPI", "Flask", "Django", "Spring Boot", "REST API", "GraphQL",
    "PostgreSQL", "MySQL", "MongoDB", "Redis", "AWS", "Azure", "GCP",
    "Docker", "Kubernetes", "Git", "Linux", "Machine Learning",
    "Deep Learning", "Natural Language Processing", "Computer Vision",
    "TensorFlow", "PyTorch", "Scikit-learn", "Pandas", "NumPy", "Spark",
    "Power BI", "Tableau", "Excel", "Data Analysis", "Statistics",
    "Agile", "Scrum", "CI/CD", "Jenkins", "Terraform",
)

_STOP_WORDS = {
    "and", "are", "for", "from", "has", "have", "into", "our", "the",
    "their", "this", "that", "with", "will", "you", "your", "years",
    "year", "work", "role", "team", "ability", "strong", "including",
    "experience", "required", "preferred", "responsibilities", "skills",
}


def _extract_sections(text: str) -> Dict[str, str]:
    sections: Dict[str, list[str]] = {}
    current_section = "preamble"
    sections[current_section] = []

    for line in text.splitlines():
        heading = re.sub(r"[^a-z ]", "", line.lower()).strip()
        heading = re.sub(r"\s+", " ", heading)
        section = _SECTION_ALIASES.get(heading)
        if section:
            current_section = section
            sections.setdefault(section, [])
        else:
            sections.setdefault(current_section, []).append(line.strip())

    return {
        name: "\n".join(lines).strip()
        for name, lines in sections.items()
    }


def _extract_skills(text: str) -> list[str]:
    skills = []
    for skill in _KNOWN_SKILLS:
        if re.search(rf"(?<!\w){re.escape(skill)}(?!\w)", text, re.IGNORECASE):
            skills.append(skill)
    return skills


def _extract_keywords(text: str, skills: list[str]) -> list[str]:
    words = re.findall(r"[A-Za-z][A-Za-z0-9+#./-]{2,}", text)
    keywords = list(dict.fromkeys(skills))
    for word in words:
        normalized = word.lower()
        if normalized not in _STOP_WORDS and word not in keywords:
            keywords.append(word)
        if len(keywords) == 60:
            break
    return keywords


def _fallback_resume_result(raw_text: str) -> Dict:
    sections = _extract_sections(raw_text)
    skills_text = sections.get("skills", "")
    skills = _extract_skills(skills_text or raw_text)
    action_verbs = [
        verb for verb in (
            "developed", "built", "designed", "implemented", "led", "created",
            "improved", "managed", "automated", "analyzed", "deployed",
            "optimized", "delivered", "reduced", "increased",
        )
        if re.search(rf"\b{verb}\b", raw_text, re.IGNORECASE)
    ]
    experience = sections.get("experience", "")
    education = sections.get("education", "")
    projects = sections.get("projects", "")
    preamble = sections.get("preamble", "")
    email = re.search(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", raw_text, re.IGNORECASE)
    phone = re.search(r"(?:\+?\d[\d\s().-]{7,}\d)", raw_text)
    linkedin = re.search(r"https?://(?:www\.)?linkedin\.com/[^\s)]+", raw_text, re.IGNORECASE)
    github = re.search(r"https?://(?:www\.)?github\.com/[^\s)]+", raw_text, re.IGNORECASE)
    first_line = next((line.strip() for line in raw_text.splitlines() if line.strip()), "")
    summary = sections.get("summary", "")
    if not summary and preamble:
        summary = " ".join(preamble.splitlines()[:3])

    return _validate_resume_result({
        "name": first_line[:120],
        "email": email.group(0) if email else None,
        "phone": phone.group(0).strip() if phone else None,
        "linkedin": linkedin.group(0) if linkedin else None,
        "github": github.group(0) if github else None,
        "professional_summary": summary,
        "skills": skills,
        "experience": [{"description": experience}] if experience else [],
        "education": [{"institution": education}] if education else [],
        "certifications": [
            item.strip(" \t-*•")
            for item in sections.get("certifications", "").splitlines()
            if item.strip(" \t-*•")
        ],
        "projects": [{"title": "Projects", "description": projects}] if projects else [],
        "action_verbs": action_verbs,
        "keywords": _extract_keywords(raw_text, skills),
    })


def _fallback_jd_result(raw_text: str) -> Dict:
    skills = _extract_skills(raw_text)
    return _validate_jd_result({
        "required_skills": skills,
        "preferred_skills": [],
        "keywords": _extract_keywords(raw_text, skills),
    })


def _call_with_fallback(parser_name: str, fallback, raw_text: str, call):
    try:
        return call()
    except (GroqError, ValueError) as exc:
        logger.warning(
            "Groq %s unavailable; using local text extraction instead: %s",
            parser_name,
            exc,
        )
        return fallback(raw_text)

def _get_client()->Groq:
    global _client
    if _client is None:
        api_key=os.getenv('GROQ_API_KEY')

        if not api_key:
            raise ValueError("GROQ_API_KEY environment variable not set")
        _client=Groq(api_key=api_key)
    return _client

RESUME_SYSTEM_PROMPT = (
    "You are a resume parser. Extract information from the resume "
    "and return ONLY a valid JSON object. No explanation, no markdown."
)

RESUME_USER_PROMPT = """Extract the following from this resume and return as JSON:
{{
  "name": "full name",
  "email": "email address",
  "phone": "phone number",
  "linkedin": "LinkedIn URL if present, otherwise null",
  "github": "GitHub URL if present, otherwise null",
  "professional_summary": "the full text of the Summary, Profile, About Me, Objective, or Professional Summary section at the top of the resume. Copy the ENTIRE paragraph exactly as written. If no such section exists, return an empty string.",
  "skills": ["list", "of", "skills"],
  "experience": [
    {{
      "job_title": "",
      "company": "",
      "start_date": "",
      "end_date": "",
      "duration_months": 0,
      "description": ""
    }}
  ],
  "education": [
    {{
      "degree": "",
      "institution": "",
      "year": ""
    }}
  ],
  "certifications": ["list of certifications"],
  "projects": [
    {{
      "title": "project name",
      "description": "what the project does and how it was built",
      "technologies": ["tech", "used"]
    }}
  ],
  "action_verbs": ["strong action verbs used in bullet points, e.g. developed, implemented, designed"],
  "keywords": ["important keywords and phrases from the resume for ATS matching"]
}}

Important instructions:
- For duration_months, calculate the number of months between start_date and end_date. If end_date is "Present" or "Current", calculate from start_date to now.
- For skills, extract ALL technical and soft skills mentioned anywhere in the resume.
- For action_verbs, find verbs that start bullet points or describe achievements.
- For keywords, extract noun phrases and technical terms relevant to ATS matching.
- Return ONLY valid JSON. No markdown code fences, no explanation.

Resume Text:
{raw_text}"""

def _call_groq(client:Groq, system_prompt:str, user_prompt:str)->str:

    response=client.chat.completions.create(
        model=GROQ_MODEL, 
        messages=[
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': user_prompt}
        ],
        temperature=0.0,
        max_tokens=4096
    )

    return response.choices[0].message.content.strip()

def _try_parse_json(text: str) -> dict | None:

    # Strip markdown code fences if present
    cleaned = text.strip()
    if cleaned.startswith("```"):

        # Remove opening fence (```json or ```)
        first_newline = cleaned.index("\n") if "\n" in cleaned else len(cleaned)
        cleaned = cleaned[first_newline + 1:]
        # Remove closing fence
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3]
        cleaned = cleaned.strip()

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        return None
    
def parse_resume(raw_text: str)->Dict:

    def parse_with_groq() -> Dict:
        client = _get_client()
        prompt = RESUME_USER_PROMPT.format(raw_text=raw_text)
        raw_response = _call_groq(client, RESUME_SYSTEM_PROMPT, prompt)
        result = _try_parse_json(raw_response)
        if result is None:
            logger.warning("Groq resume parse returned invalid JSON; retrying...")
            strict_prompt = (
                "Your previous response was not valid JSON. "
                "Return ONLY the raw JSON object, no markdown, no explanation, no code fences.\n\n"
                + prompt
            )
            raw_response = _call_groq(client, RESUME_SYSTEM_PROMPT, strict_prompt)
            result = _try_parse_json(raw_response)
        if result is None:
            raise ValueError("Groq returned an unparseable resume response after retry")
        return _validate_resume_result(result)

    return _call_with_fallback(
        "resume parser",
        _fallback_resume_result,
        raw_text,
        parse_with_groq,
    )
    
JD_SYSTEM_PROMPT = (
    "You are a job description parser. Extract information and "
    "return ONLY a valid JSON object. No explanation, no markdown."
)

JD_USER_PROMPT = """Extract the following from this job description and return as JSON:
{{
  "job_title": "",
  "required_skills": ["list of must-have skills"],
  "preferred_skills": ["list of nice-to-have skills"],
  "experience_required": "",
  "education_required": "",
  "key_responsibilities": ["list of responsibilities"],
  "keywords": ["important keywords and phrases for ATS matching"]
}}

Important instructions:
- required_skills: skills explicitly stated as required or must-have.
- preferred_skills: skills stated as preferred, nice-to-have, or bonus.
- keywords: extract ALL important terms an ATS system would match against,
  including skills, technologies, certifications, and domain terms.
- Return ONLY valid JSON. No markdown code fences, no explanation.

Job Description Text:
{raw_text}"""

def parse_job_description(raw_text: str) -> Dict:
    def parse_with_groq() -> Dict:
        client = _get_client()
        prompt = JD_USER_PROMPT.format(raw_text=raw_text)

        raw_response = _call_groq(client, JD_SYSTEM_PROMPT, prompt)
        result = _try_parse_json(raw_response)
        if result is None:
            logger.warning("Groq JD parse returned invalid JSON; retrying...")
            strict_prompt = (
                "Your previous response was not valid JSON. "
                "Return ONLY the raw JSON object, no markdown, no explanation, no code fences.\n\n"
                + prompt
            )
            raw_response = _call_groq(client, JD_SYSTEM_PROMPT, strict_prompt)
            result = _try_parse_json(raw_response)
        if result is None:
            raise ValueError("Groq returned an unparseable job-description response after retry")
        return _validate_jd_result(result)

    return _call_with_fallback(
        "job-description parser",
        _fallback_jd_result,
        raw_text,
        parse_with_groq,
    )

#it will make sure, that the parse json has all the valid fields we expect
def _validate_jd_result(result: dict) -> dict:
    
    defaults = {
        "job_title": "",
        "required_skills": [],
        "preferred_skills": [],
        "experience_required": "",
        "education_required": "",
        "key_responsibilities": [],
        "keywords": [],
    }

    for key, default in defaults.items():
        if key not in result or result[key] is None:
            result[key] = default
        if isinstance(default, list) and not isinstance(result[key], list):
            result[key] = default

    return result


#to make sure the parse json has all the valid json fields
def _validate_resume_result(result: dict) -> dict:

    defaults = {
        "name": "",
        "email": None,
        "phone": None,
        "linkedin": None,
        "github": None,
        "professional_summary": "",
        "skills": [],
        "experience": [],
        "education": [],
        "certifications": [],
        "projects": [],
        "action_verbs": [],
        "keywords": [],
    }
    for key, default in defaults.items():
        if key not in result or result[key] is None:
            result[key] = default
            
        # Ensure list fields are actually lists
        if isinstance(default, list) and not isinstance(result[key], list):
            result[key] = default

    #Validate experience entries
    for exp in result.get("experience", []):
        if not isinstance(exp, dict):
            continue
        exp.setdefault("job_title", "")
        exp.setdefault("company", "")
        exp.setdefault("start_date", "")
        exp.setdefault("end_date", "")
        exp.setdefault("duration_months", 0)
        exp.setdefault("description", "")
        #Ensure duration_months is an int
        try:
            exp["duration_months"] = int(exp["duration_months"])
        except (ValueError, TypeError):
            exp["duration_months"] = 0

    #Validate project entries
    for proj in result.get("projects", []):
        if not isinstance(proj, dict):
            continue
        proj.setdefault("title", "")
        proj.setdefault("description", "")
        proj.setdefault("technologies", [])

    return result