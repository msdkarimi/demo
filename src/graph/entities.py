
from __future__ import annotations

import re

from taxonomy.classify import keyword_pattern

# canonical name -> (kind, aliases). Aliases are lowercase literals. Ambiguous
# words (Go, R, Swift, Unity, Windows, Ruby) are only matched in unambiguous forms.
GAZETTEER: dict[str, tuple[str, list[str]]] = {
    # programming languages
    "Python": ("language", ["python"]),
    "JavaScript": ("language", ["javascript", "js"]),
    "TypeScript": ("language", ["typescript"]),
    "Java": ("language", ["java"]),
    "C++": ("language", ["c++", "cpp"]),
    "C#": ("language", ["c#", "csharp"]),
    "SQL": ("language", ["sql", "t-sql", "pl/sql"]),
    "PHP": ("language", ["php"]),
    "Rust": ("language", ["rust programming", "rust code", "in rust", "rust's"]),
    "Kotlin": ("language", ["kotlin"]),
    "Swift": ("language", ["swiftui", "swift code", "in swift"]),
    "Ruby": ("language", ["ruby on rails"]),
    "MATLAB": ("language", ["matlab"]),
    "VBA": ("language", ["vba", "excel macro", "vb.net"]),
    "Bash": ("language", ["bash", "shell script"]),
    "PowerShell": ("language", ["powershell"]),
    "HTML/CSS": ("language", ["html", "css"]),
    "Solidity": ("language", ["solidity"]),
    "Lua": ("language", ["lua"]),
        # frameworks and libraries
    "React": ("framework", ["reactjs", "react.js", "react native", "react component", "react app", "import react", "usestate"]),
    "Angular": ("framework", ["angularjs", "@angular", "angular component", "angular app"]),
    "Vue": ("framework", ["vue", "vue.js", "vuejs"]),
    "Next.js": ("framework", ["next.js", "nextjs"]),
    "Node.js": ("framework", ["node.js", "nodejs", "express.js"]),
    "Django": ("framework", ["django"]),
    "Flask": ("framework", ["flask app", "from flask", "flask api", "flask route"]),
    "FastAPI": ("framework", ["fastapi"]),
    "Spring": ("framework", ["spring boot"]),
    ".NET": ("framework", [".net", "asp.net"]),
    "Laravel": ("framework", ["laravel"]),
    "Flutter": ("framework", ["flutter"]),
    "PyTorch": ("framework", ["pytorch", "import torch", "torch.nn"]),
    "TensorFlow": ("framework", ["tensorflow", "keras"]),
    "pandas": ("framework", ["pandas"]),
    "NumPy": ("framework", ["numpy"]),
    "scikit-learn": ("framework", ["scikit-learn", "sklearn"]),
    "OpenCV": ("framework", ["opencv", "cv2"]),
    "LangChain": ("framework", ["langchain"]),
    "Unity": ("framework", ["unity3d", "unity engine", "unity game", "monobehaviour"]),
    "Unreal Engine": ("framework", ["unreal engine", "ue4", "ue5"]),
    "Three.js": ("framework", ["three.js", "threejs"]),
    "Tailwind": ("framework", ["tailwind", "tailwindcss"]),
    "Bootstrap": ("framework", ["bootstrap 5", "bootstrap css", "bootstrap 4"]),
    "Selenium": ("framework", ["selenium"]),
    "Playwright": ("framework", ["playwright"]),
    # tools and platforms
    "Excel": ("tool", ["ms excel", "microsoft excel", "excel sheet", "excel file", "excel formula", "in excel", "excel spreadsheet", "excel workbook", "xlsx"]),
    "Google Sheets": ("tool", ["google sheets", "google sheet", "apps script"]),
    "Power BI": ("tool", ["power bi", "powerbi"]),
    "Tableau": ("tool", ["tableau"]),
    "Docker": ("tool", ["docker", "dockerfile"]),
    "Kubernetes": ("tool", ["kubernetes", "k8s"]),
    "Git": ("tool", ["git", "github", "gitlab"]),
    "Linux": ("platform", ["linux", "ubuntu", "debian"]),
    "Windows": ("platform", ["windows 10", "windows 11", "windows os", "microsoft windows"]),
    "Android": ("platform", ["android"]),
    "iOS": ("platform", ["ios", "iphone"]),
    "AWS": ("platform", ["aws", "amazon web services", "s3 bucket", "lambda function"]),
    "Azure": ("platform", ["azure"]),
    "Google Cloud": ("platform", ["google cloud", "gcp"]),
    "Firebase": ("platform", ["firebase"]),
    "MySQL": ("tool", ["mysql"]),
    "PostgreSQL": ("tool", ["postgresql", "postgres"]),
    "MongoDB": ("tool", ["mongodb", "mongoose"]),
    "Redis": ("tool", ["redis"]),
    "WordPress": ("platform", ["wordpress"]),
    "Shopify": ("platform", ["shopify"]),
    "SAP": ("platform", ["sap"]),
    "Salesforce": ("platform", ["salesforce"]),
    "Jira": ("tool", ["jira"]),
    "Figma": ("tool", ["figma"]),
    "Photoshop": ("tool", ["photoshop"]),
    "Blender": ("tool", ["blender 3d", "in blender", "blender python"]),
    "AutoCAD": ("tool", ["autocad"]),
    "SolidWorks": ("tool", ["solidworks"]),
    "Arduino": ("tool", ["arduino"]),
    "Telegram": ("platform", ["telegram"]),
    "Discord": ("platform", ["discord"]),
    # AI products
    "ChatGPT": ("ai_product", ["chatgpt", "chat gpt"]),
    "GPT-4": ("ai_product", ["gpt-4", "gpt4", "gpt 4"]),
    "OpenAI": ("ai_product", ["openai api", "openai"]),
    "Midjourney": ("ai_product", ["midjourney", "/imagine prompt"]),
    "Stable Diffusion": ("ai_product", ["stable diffusion"]),
    "DALL-E": ("ai_product", ["dall-e", "dalle"]),
    "Hugging Face": ("ai_product", ["hugging face", "huggingface"]),
    # social and commerce platforms
    "Instagram": ("platform", ["instagram"]),
    "TikTok": ("platform", ["tiktok"]),
    "YouTube": ("platform", ["youtube"]),
    "LinkedIn": ("platform", ["linkedin"]),
    "Facebook": ("platform", ["facebook"]),
    "Twitter/X": ("platform", ["twitter", "tweet"]),
    "Etsy": ("platform", ["etsy"]),
    "Amazon": ("platform", ["amazon"]),
    "Fiverr": ("platform", ["fiverr", "fivver"]),
    # franchises (fiction / fandom)
    "Doki Doki Literature Club": ("franchise", ["ddlc", "doki doki", "sayori", "monika", "natsuki"]),
    "Naruto": ("franchise", ["naruto"]),
    "Freedom Planet": ("franchise", ["freedom planet"]),
    "Scooby-Doo": ("franchise", ["scooby-doo", "scooby doo", "scooby"]),
    "Marvel": ("franchise", ["marvel comics", "marvel cinematic", "avengers", "spider-man", "spiderman"]),
    "Harry Potter": ("franchise", ["harry potter", "hogwarts"]),
    "Pokémon": ("franchise", ["pokemon", "pokémon"]),
    "Sonic": ("franchise", ["sonic the hedgehog", "sonic all characters", "sonic characters"]),
    "Minecraft": ("franchise", ["minecraft"]),
    "Genshin Impact": ("franchise", ["genshin impact", "genshin"]),
    "Dragon Ball": ("franchise", ["dragon ball", "goku"]),
    "One Piece": ("franchise", ["one piece"]),
    "Star Wars": ("franchise", ["star wars"]),
    "Hades": ("franchise", ["zagreus"]),
    "The Walking Dead": ("franchise", ["walking dead"]),
    "Azur Lane": ("franchise", ["azur lane"]),
    "World of Warcraft": ("franchise", ["world of warcraft", "warcraft"]),
    "Roblox": ("franchise", ["roblox"]),
    "SCP Foundation": ("franchise", ["scp foundation", "scp-"]),
    # standards and exams
    "IELTS": ("standard", ["ielts"]),
    "GDPR": ("standard", ["gdpr"]),
}

_PATTERNS = {name: re.compile("|".join(map(keyword_pattern, aliases)), re.I)
             for name, (_, aliases) in GAZETTEER.items()}
_ISO = re.compile(r"\bISO[ /-]?(\d{4,5})\b")


def extract(text: str) -> list[tuple[str, str]]:
    """(name, kind) for every gazetteer entity and ISO standard in the text."""
    text = text or ""
    found = [(name, GAZETTEER[name][0]) for name, rx in _PATTERNS.items() if rx.search(text)]
    found += [(f"ISO {m}", "standard") for m in sorted(set(_ISO.findall(text)))]
    return found
