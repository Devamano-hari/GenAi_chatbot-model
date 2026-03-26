import os
import requests
import json
import validators
import re
import textwrap
from bs4 import BeautifulSoup
from dotenv import load_dotenv

import traceback

load_dotenv()

# APIMart Integration for Source Code Mode fallback
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.apimart.ai/v1")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
SEARCH_ENGINE_ID = os.getenv("SEARCH_ENGINE_ID")

# Try loading BART summarizer natively
try:
    from transformers import pipeline
    print("Loading Summarizer (facebook/bart-large-cnn)...")
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
    SUMMARIZER_LOADED = True
except Exception as e:
    print(f"CRITICAL ERROR loading Summarizer: {e}")
    SUMMARIZER_LOADED = False

def clean_text(text):
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'\[[^\]]*\]', '', text)
    text = re.sub(r'\(\d+\)', '', text)
    text = re.sub(r'http\S+', '', text)
    return text.strip()

def is_valid_paragraph(text):
    if len(text) < 40:
        return False
    junk_words = ["home","menu","about","contact","privacy","terms",
                  "cookies","subscribe","login","sign up","register"]
    return not any(word in text.lower() for word in junk_words)

def fetch_content_p(url, mode="text"):
    try:
        response = requests.get(url, timeout=10, headers={'User-Agent': 'Mozilla/5.0'})
        
        # Strictly skip if it's a 403 Forbidden or 404
        if response.status_code != 200:
            return ""

        soup = BeautifulSoup(response.text, 'html.parser')

        for script in soup(["script", "style", "nav", "footer", "header", "aside"]):
            script.decompose()

        if mode == "code":
            blocks = soup.find_all(['pre', 'code'])
            cleaned = []
            for b in blocks:
                code_text = b.get_text().strip()
                if len(code_text) > 20: 
                    cleaned.append(code_text)
            final_text = "\n\n".join(cleaned)[:8000]
        else:
            paragraphs = soup.find_all("p")
            cleaned = []
            for para in paragraphs:
                text = clean_text(para.get_text())
                if is_valid_paragraph(text):
                    cleaned.append(text)
            final_text = " ".join(cleaned)[:10000]
        
        # Access Denied Heuristics (Sometimes returns 200 OK but shows an error overlay)
        error_keywords = ["access denied", "you do not have permission", "enable javascript", "please enable cookies", "security check", "cloudflare", "attention required", "proxy wall", "captcha", "verify you are human"]
        if any(keyword in final_text.lower() for keyword in error_keywords):
            return ""
            
        return final_text
    except Exception as e:
        print(f"Scrape error for {url}: {e}")
        return ""

def remove_repetition(text):
    sentences = text.split('.')
    seen = set()
    unique = []

    for s in sentences:
        s = s.strip()
        if s and s not in seen:
            unique.append(s)
            seen.add(s)

    return '. '.join(unique) + '.'

def summarize_text(text):
    if not SUMMARIZER_LOADED:
        return text[:1000] # Fallback if BART failed to load due to OS crash
    
    chunks = textwrap.wrap(text, 800)
    summaries = []

    for chunk in chunks[:3]:
        out = summarizer(chunk, max_length=120, min_length=50, do_sample=False)
        summaries.append(out[0]['summary_text'])

    return remove_repetition(" ".join(summaries))

def search_google(query, mode="text", language=""):
    if not GOOGLE_API_KEY or not SEARCH_ENGINE_ID or "your-" in GOOGLE_API_KEY:
        return []
        
    if mode == "code":
        query = f"{query} {language} code examples tutorial".strip()
    
    url = "https://www.googleapis.com/customsearch/v1"
    links = []
    
    # Fetch top 20 results (2 pages) to ensure we get 5 pristine code sources
    for start in [1, 11]:
        params = {
            "q": query,
            "key": GOOGLE_API_KEY,
            "cx": SEARCH_ENGINE_ID,
            "num": 10,
            "start": start
        }
        try:
            response = requests.get(url, params=params, timeout=10)
            data = response.json()
            items = data.get('items', [])
            links.extend([item.get('link') for item in items if item.get('link')])
        except Exception as e:
            print(f"Google Search Connection Error: {e}")
            break
            
    return links

def generate_gpt_content(prompt, context_str, mode):
    model = os.getenv("API_MODEL", "gpt-4o-mini")
    url = f"{OPENAI_BASE_URL}/chat/completions"
    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json"
    }
    
    sys_code = "Provide highly technical Python code and explanation. Ensure your entire output response is strictly 150 to 200 words." if mode == "code" else "You are a senior technical writer generating an extremely practical, educational summary. Using the provided search context, write a highly rephrased and cleaned technical explanation specifically addressing the prompt. If the user asked for specific types, uses, applications, or advantages, focus entirely on delivering those practical facts. If it is a generic concept, explain its core meaning and mechanics.\nCRITICAL STRICT REQUIREMENT: Your final output MUST be exactly between 150 to 200 words. Count your words. Do not include dates, authors, or direct source names."

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": sys_code},
            {"role": "user", "content": f"Task: {prompt}\n\nSearch Context:\n{context_str}"}
        ],
        "max_tokens": 400,
        "temperature": 0.7,
        "stream": False
    }

    try:
        response = requests.post(url, headers=headers, json=payload)
        data = response.json()
        if "choices" in data:
            return data["choices"][0]["message"]["content"].strip()
        return f"Error reading response format: {data}"
    except Exception as e:
        return f"GPT generation error: {e}"

def extract_technical_nugget(topic, text_chunk, mode="text"):
    model = os.getenv("API_MODEL", "gpt-4o-mini")
    url = f"{OPENAI_BASE_URL}/chat/completions"
    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json"
    }
    
    if mode == "code":
        sys_prompt = f"You are a strict technical filter for a programming AI. Review the raw code snippets scraped from a documentation website. Extract the single most relevant and functional block of source code that strictly solves or explains '{topic}'.\n\nCRITICAL INSTRUCTION: Return ONLY the raw code block. If the text does not contain any functional code snippets related to the syntax, or is just a frontend menu, you MUST reply with exactly one word: REJECT"
    else:
        sys_prompt = f"You are a strict technical filter for an academic research AI. Read the scraped website text. If it contains highly technical, educational, or practical information specifically about '{topic}', extract exactly that information into a single paragraph (Maximum 100 words).\n\nCRITICAL INSTRUCTION: IF the text is primarily course promotions (e.g. Coursera ads), sales pitches, access denied messages, cookie banners, or completely lacks deep technical value regarding the topic, you MUST reply with exactly one word: REJECT"

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": text_chunk[:3000]}
        ],
        "max_tokens": 150,
        "temperature": 0.2,
        "stream": False
    }

    try:
        response = requests.post(url, headers=headers, json=payload)
        data = response.json()
        if "choices" in data:
            return data["choices"][0]["message"]["content"].strip()
        return "REJECT"
    except Exception as e:
        return "REJECT"

def evaluate_domain_tier(url, mode="text"):
    url_lower = url.lower()
    
    if mode == "code":
        tier_1 = [r'(github|stackoverflow)\.com', r'(docs\.python|mozilla)\.org', r'microsoft\.com']
        tier_2 = [r'(geeksforgeeks)\.org', r'(w3schools|hackerrank|leetcode)\.com']
        
        for p in tier_1:
            if re.search(p, url_lower): return 1, 1.0
        for p in tier_2:
            if re.search(p, url_lower): return 2, 0.8
        return 3, 0.5
        
    tier_1_patterns = [
        r'(wikipedia|arxiv)\.org',                 # Knowledge bases (Top priority)
        r'\.(gov|edu|mil|ac)(\.[a-z]{2,3})?',   # Government / Academic institutions
        r'(gov|ac)\.in',                           # India gov/ac
        r'pubmed\.ncbi\.nlm\.nih\.gov',            # Medical research
        r'(nature|ieee|acm|springer)\.(com|org)',  # Academic journals
        r'(britannica|who|un)\.(com|org)'          # Trusted organizations
    ]
    tier_2_patterns = [
        r'(bbc|reuters|apnews|nytimes|wsj)',        # Journalism
        r'(github|gitlab)\.com',                    # Code repositories
        r'(khanacademy|jstor|merriam-webster)',      # Educational / reference
        r'(co|org)\.in'                             # India reputable sites
    ]
    tier_3_patterns = [
        r'\.io',                                    # Tech/startups
        r'(forbes|bloomberg|guardian|cnn)',          # Commercial media
        r'(wired|techcrunch|zdnet)\.com'            # Tech/business media
    ]
    
    for p in tier_1_patterns:
        if re.search(p, url_lower): return 1, 1.0
    for p in tier_2_patterns:
        if re.search(p, url_lower): return 2, 0.8
    for p in tier_3_patterns:
        if re.search(p, url_lower): return 3, 0.5
        
    return 3, 0.5  # All unrecognized domains default to Tier 3

def rank_links(links):
    if not links:
        return []
    
    scored_links = []
    for link in links:
        tier, weight = evaluate_domain_tier(link)
        score = 0
        
        if tier == 1: score += 100
        elif tier == 2: score += 75
        elif tier == 3: score += 50
        else: score += 20
        
        # Prefer HTTPS
        if link.lower().startswith('https'):
            score += 5
            
        # Penalize spam/low-value nodes heavily
        if any(x in link.lower() for x in ['promo', 'discount', 'ad', 'social', 'reddit.com', 'quora.com', 'twitter.com', 'facebook.com', 'instagram.com', 'pinterest']):
            score -= 100
            
        scored_links.append((score, link))
        
    scored_links.sort(key=lambda x: x[0], reverse=True)
    return [link for score, link in scored_links]
def generate_content(prompt, topic, mode="text", language=""):
    # Google API pulls up to 20
    links = search_google(topic, mode, language=language)
    
    if not links:
        links = ["https://example.com/no-search-results"]
        
    ranked_links = rank_links(links)
    
    # Scrape sequentially until exactly 5 solid links are acquired
    successful_links = []
    technical_kb_parts = []
    used_tiers = []
    
    for link in ranked_links:
        if len(successful_links) >= 5:
            break
            
        scraped = fetch_content_p(link, mode)
        if len(scraped) > 50: # Ensure it has meaningful heft, lowered for pure code snippets
            nugget = extract_technical_nugget(topic, scraped, mode)
            
            # If the LLM determines this link is just ads/promotions/garbage, we skip the link entirely
            if nugget.upper() != "REJECT" and not nugget.upper().startswith("REJECT"):
                technical_kb_parts.append(nugget)
                successful_links.append(link)
                
                # Register tier for final accuracy matrix calc
                tier, weight = evaluate_domain_tier(link, mode)
                used_tiers.append(tier)
            
    # Calculate the EXACT Combined Fractional Weighted Average of ALL sources
    if used_tiers:
        weight_map = {1: 1.0, 2: 0.8, 3: 0.5, 4: 0.5}
        total_weight = sum(weight_map.get(t, 0.5) for t in used_tiers)
        domain_weight = total_weight / len(used_tiers)
    else:
        domain_weight = 0.5
        
    tier_counts = {"tier1": 0, "tier2": 0, "tier3": 0}
    for t in used_tiers:
        if t == 1: tier_counts["tier1"] += 1
        elif t == 2: tier_counts["tier2"] += 1
        else: tier_counts["tier3"] += 1
            
    # Compile the final external knowledge base
    kb_text = "\n\n".join([f"[{i+1}] {t}" for i, t in enumerate(technical_kb_parts)])
    if not kb_text.strip():
        kb_text = "No technical content could be scraped from the domain sources."

    # Final generated content (rephrased, cleaned, strictly 150-200 words) using GPT
    final_generated = generate_gpt_content(prompt, kb_text, mode)

    return {
        "reference_source": successful_links[0] if successful_links else "No source",
        "reference_content": kb_text,
        "external_sources": successful_links,
        "knowledge_base": kb_text,
        "generated_content": final_generated,
        "domain_weight": domain_weight,
        "tier_counts": tier_counts
    }
