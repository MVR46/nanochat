# Persona Replication from Survey Data - Complete Guide

A methodology for training nanochat to replicate a specific person's responses based on survey data, with testing and evaluation strategies.

---

## Table of Contents

1. [Overview](#overview)
2. [Best Data Sources for Persona Replication](#best-data-sources-for-persona-replication)
3. [Data Collection Strategy](#data-collection-strategy)
4. [Data Preparation Pipeline](#data-preparation-pipeline)
5. [Training Strategy](#training-strategy)
6. [Testing Methodology](#testing-methodology)
7. [Evaluation Metrics](#evaluation-metrics)
8. [Implementation Guide](#implementation-guide)
9. [Tips for High Fidelity](#tips-for-high-fidelity)
10. [Complete Example](#complete-example)

---

## Overview

**Goal**: Train nanochat to replicate how a specific person (the "source persona") would respond to questions, validated against their actual survey responses.

**Key Challenge**: Getting enough high-quality, diverse data to capture the persona's:
- Opinion patterns
- Communication style
- Reasoning approach
- Value system
- Knowledge domains
- Linguistic preferences

---

## Best Data Sources for Persona Replication

### 1. **Comprehensive Personality Surveys** (BEST)

#### Myers-Briggs Type Indicator (MBTI) Extended
- **Questions**: 100-300 items
- **Coverage**: Decision-making, social behavior, information processing
- **Format**: Strongly disagree → Strongly agree (7-point scale)
- **Advantage**: Rich behavioral data

#### Big Five Personality Test (NEO-PI-R)
- **Questions**: 240 items (or 60-item short form)
- **Coverage**: Openness, Conscientiousness, Extraversion, Agreeableness, Neuroticism
- **Format**: 5-point Likert scale
- **Advantage**: Well-validated, comprehensive

#### 16 Personalities Test
- **Questions**: ~60-100 items
- **Coverage**: Personality traits, work style, relationships
- **Format**: Agree/Disagree spectrum
- **Advantage**: Modern, accessible, free

**Recommendation**: Use multiple personality assessments to get 200-500+ data points.

---

### 2. **Values & Beliefs Surveys**

#### Political Compass / ISideWith
- **Questions**: 50-100 items
- **Coverage**: Political, social, economic views
- **Format**: Agree/Disagree with importance weights
- **Advantage**: Clear stance on controversial topics

#### Moral Foundations Questionnaire
- **Questions**: 30 items
- **Coverage**: Care, Fairness, Loyalty, Authority, Sanctity
- **Format**: Relevance ratings (0-5)
- **Advantage**: Deep ethical framework

#### World Values Survey (Subset)
- **Questions**: Select 50-100 relevant items
- **Coverage**: Cultural values, life satisfaction, trust
- **Format**: Various (rankings, scales)
- **Advantage**: Cross-cultural perspectives

---

### 3. **Cognitive Style Assessments**

#### Critical Thinking Tests
- **Questions**: 30-50 scenario-based questions
- **Coverage**: Logic, reasoning, problem-solving approaches
- **Format**: Multiple choice with written justifications
- **Advantage**: Captures thinking patterns

#### Learning Style Inventory (VARK, Kolb)
- **Questions**: 20-40 items
- **Coverage**: How person processes information
- **Format**: Preference rankings
- **Advantage**: Communication style insights

---

### 4. **Domain-Specific Knowledge Surveys**

#### Interest & Expertise Questionnaires
- **Questions**: 100+ items across domains
- **Coverage**: Technology, arts, science, sports, politics, etc.
- **Format**: Knowledge + opinion + interest levels
- **Advantage**: Captures knowledge domains and opinions

**Example domains**:
- Technology trends (AI, crypto, social media)
- Scientific topics (climate, health, space)
- Cultural topics (music, movies, books)
- Social issues (education, healthcare, justice)

---

### 5. **Conversational Data** (Supplement)

#### Text Message Archives
- Personal chat histories (with consent)
- Email correspondence
- Social media posts/comments

#### Interview Transcripts
- Q&A sessions (30+ questions)
- Think-aloud protocols
- Recorded conversations

**Advantage**: Natural language data shows actual communication style.

---

### 6. **Custom Comprehensive Survey** (RECOMMENDED APPROACH)

Build a custom survey combining all the above:

**Section A: Personality (100 questions)**
- Big Five items
- Decision-making scenarios
- Social preference questions

**Section B: Values & Beliefs (100 questions)**
- Political/social views
- Moral dilemmas
- Life priorities

**Section C: Cognitive Style (50 questions)**
- Problem-solving scenarios
- Communication preferences
- Learning approaches

**Section D: Knowledge & Opinions (150 questions)**
- Domain knowledge tests
- Hot topics opinions
- Predictions and forecasts

**Section E: Open-Ended (50 questions)**
- "Explain your reasoning..."
- "How would you handle..."
- "What do you think about..."

**Total**: 450+ questions capturing comprehensive persona

---

## Data Collection Strategy

### Phase 1: Baseline Survey (Essential)

**What to collect**:
```
For each question, record:
1. The question text
2. The response (rating/choice/text)
3. Confidence level (optional)
4. Reasoning (if open-ended)
5. Response time (optional - shows certainty)
```

**Survey design tips**:
- Mix question types (Likert, multiple choice, open-ended)
- Include follow-up "Why?" questions
- Add scenario-based questions
- Use consistent rating scales
- Allow "neutral" or "don't know" options

### Phase 2: Augmentation (Recommended)

**Expand the data by**:

#### A. Synthetic Question Generation
Generate related questions based on survey responses:

```python
# If person strongly agrees with "I prefer working alone"
# Generate related questions:
- "How do you feel about open office layouts?"
- "Do you prefer async communication or meetings?"
- "Would you rather work remotely or in-office?"
```

#### B. Conversation Simulation
Create multi-turn conversations from survey items:

```
User: "What's your opinion on remote work?"
Persona: [Based on work-style survey responses]
User: "Why do you feel that way?"
Persona: [Based on reasoning patterns]
User: "What about team collaboration?"
Persona: [Based on social preference data]
```

#### C. Cross-Domain Inference
Generate responses to new questions by combining survey insights:

```python
# From survey: High openness, liberal politics, tech-savvy
# Infer response to: "What do you think about AI regulation?"
# Generate: "I believe in AI progress but with ethical guardrails..."
```

---

## Data Preparation Pipeline

### Step 1: Survey Data → Conversation Format

Convert survey responses to nanochat's conversation format:

```python
# survey_to_conversations.py

import json

def convert_survey_to_conversations(survey_data, persona_name="Alex"):
    """
    Convert survey responses to conversation format.
    
    Args:
        survey_data: List of dicts with 'question', 'answer', 'reasoning'
        persona_name: Name of the person being modeled
    
    Returns:
        List of conversation dicts in nanochat format
    """
    conversations = []
    
    for item in survey_data:
        question = item['question']
        answer = item['answer']
        reasoning = item.get('reasoning', '')
        
        # Create conversation
        conversation = {
            "messages": [
                {
                    "role": "user",
                    "content": question
                },
                {
                    "role": "assistant",
                    "content": format_response(answer, reasoning, persona_name)
                }
            ]
        }
        conversations.append(conversation)
    
    return conversations


def format_response(answer, reasoning, persona_name):
    """Format the response in the persona's style."""
    
    # For Likert scale responses
    if isinstance(answer, int):
        scale_words = {
            1: "I strongly disagree",
            2: "I somewhat disagree", 
            3: "I'm neutral",
            4: "I somewhat agree",
            5: "I strongly agree"
        }
        response = scale_words.get(answer, f"My rating is {answer}/5")
        
        if reasoning:
            response += f". {reasoning}"
    
    # For multiple choice
    elif isinstance(answer, str):
        response = answer
        if reasoning:
            response += f" {reasoning}"
    
    return response


# Example usage
survey_data = [
    {
        "question": "Do you prefer working alone or in teams?",
        "answer": 1,  # Strongly prefer alone
        "reasoning": "I find I'm most productive when I can focus deeply without interruptions."
    },
    {
        "question": "How important is work-life balance to you?",
        "answer": 5,  # Very important
        "reasoning": "I prioritize time with family and personal hobbies. Work is important but not everything."
    },
    # ... more survey items
]

conversations = convert_survey_to_conversations(survey_data, persona_name="Alex")

# Save as JSONL
with open('persona_survey_conversations.jsonl', 'w') as f:
    for conv in conversations:
        f.write(json.dumps(conv) + '\n')
```

### Step 2: Augment with Multi-Turn Dialogues

```python
# augment_conversations.py

def create_multi_turn_dialogue(question, answer, reasoning, follow_ups):
    """
    Create multi-turn conversation from single survey item.
    
    Args:
        question: Original survey question
        answer: Person's answer
        reasoning: Person's reasoning
        follow_ups: List of follow-up question-answer pairs
    """
    messages = [
        {"role": "user", "content": question},
        {"role": "assistant", "content": f"{answer}. {reasoning}"}
    ]
    
    for follow_up in follow_ups:
        messages.append({"role": "user", "content": follow_up['question']})
        messages.append({"role": "assistant", "content": follow_up['answer']})
    
    return {"messages": messages}


# Example
dialogue = create_multi_turn_dialogue(
    question="What's your stance on remote work?",
    answer="I strongly support remote work",
    reasoning="I'm most productive in my own environment",
    follow_ups=[
        {
            "question": "What about team collaboration?",
            "answer": "I prefer async communication via Slack or email. Video calls when necessary but not daily standups."
        },
        {
            "question": "Do you think remote work affects company culture?",
            "answer": "It can, but with intentional effort through virtual events and clear communication, culture can thrive remotely."
        }
    ]
)
```

### Step 3: Add Persona Identity & Style

```python
# add_persona_metadata.py

def add_persona_identity(conversations, persona_profile):
    """
    Add identity conversations explaining who the persona is.
    
    Args:
        conversations: Existing conversation list
        persona_profile: Dict with persona details
    """
    identity_convos = [
        {
            "messages": [
                {"role": "user", "content": "Who are you?"},
                {"role": "assistant", "content": f"I'm {persona_profile['name']}, {persona_profile['description']}."}
            ]
        },
        {
            "messages": [
                {"role": "user", "content": "Tell me about yourself."},
                {"role": "assistant", "content": persona_profile['bio']}
            ]
        },
        {
            "messages": [
                {"role": "user", "content": "What are your core values?"},
                {"role": "assistant", "content": persona_profile['values']}
            ]
        },
    ]
    
    return identity_convos + conversations


# Example profile
alex_profile = {
    "name": "Alex",
    "description": "a 32-year-old software engineer from Seattle",
    "bio": "I'm a software engineer who values work-life balance and deep focus. I'm introverted but enjoy meaningful conversations. I care deeply about technology ethics and environmental sustainability.",
    "values": "Autonomy, intellectual honesty, sustainability, and continuous learning are what matter most to me."
}
```

---

## Training Strategy

### Layer 1: Base Model (Pretrained)

Use nanochat's standard base model - no changes needed.

```bash
# Standard base training
torchrun --standalone --nproc_per_node=8 -m scripts.base_train -- --depth=20
```

### Layer 2: Midtraining with Persona Data

Include persona survey data in midtraining mixture:

```python
# Modify scripts/mid_train.py (lines 98-106)

from tasks.customjson import CustomJSON

base_dir = get_base_dir()
persona_survey_path = os.path.join(base_dir, "persona_survey_conversations.jsonl")

train_dataset = TaskMixture([
    SmolTalk(split="train", stop=50000),  # General conversation ability (reduced)
    MMLU(subset="auxiliary_train", split="train"),  # Keep reasoning
    GSM8K(subset="main", split="train"),  # Keep math
    
    # Heavy emphasis on persona data
    CustomJSON(filepath=persona_survey_path),
    CustomJSON(filepath=persona_survey_path),
    CustomJSON(filepath=persona_survey_path),
    CustomJSON(filepath=persona_survey_path),  # 4x repetition for strong imprinting
    
    SimpleSpelling(size=50000, split="train"),
    SpellingBee(size=20000, split="train"),
])
```

### Layer 3: SFT with Pure Persona Focus

Make SFT even more persona-focused:

```python
# Modify scripts/chat_sft.py (lines 84-92)

train_ds = TaskMixture([
    # Minimal general capability maintenance
    SmolTalk(split="train", stop=2000),  # Very limited general chat
    
    # Maximum persona emphasis (80% of training)
    CustomJSON(filepath=persona_survey_path),
    CustomJSON(filepath=persona_survey_path),
    CustomJSON(filepath=persona_survey_path),
    CustomJSON(filepath=persona_survey_path),
    CustomJSON(filepath=persona_survey_path),
    CustomJSON(filepath=persona_survey_path),
    CustomJSON(filepath=persona_survey_path),
    CustomJSON(filepath=persona_survey_path),  # 8x repetition
])
```

### Layer 4: Optional Reinforcement Learning

For critical persona consistency on specific topics:

```python
# Create RL reward based on persona match
# (Advanced - requires modifying chat_rl.py)

def persona_match_reward(response, expected_persona_style):
    """
    Reward function that scores how well response matches persona.
    """
    # Check for:
    # - Key phrases persona would use
    # - Opinion alignment
    # - Communication style match
    # - Reasoning pattern similarity
    
    score = 0.0
    # ... scoring logic
    return score
```

---

## Testing Methodology

### Test Set Creation

**Split your survey data**:
```python
# 80% for training, 20% for testing
training_questions = survey_data[:int(len(survey_data) * 0.8)]
test_questions = survey_data[int(len(survey_data) * 0.8):]
```

**Ensure test set includes**:
- Different question phrasings
- Cross-domain questions
- Edge cases
- Questions the persona had strong opinions on

### Automated Testing Script

```python
# test_persona_replication.py

import json
from nanochat.checkpoint_manager import load_model
from nanochat.engine import Engine
import torch

def test_persona_replication(model_source, test_data_path):
    """
    Test how well the model replicates persona responses.
    
    Args:
        model_source: "sft" or "mid" 
        test_data_path: Path to test questions JSONL
    
    Returns:
        Dict with accuracy metrics
    """
    # Load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, tokenizer, _ = load_model(model_source, device, phase="eval")
    engine = Engine(model, tokenizer)
    
    # Load test data
    with open(test_data_path, 'r') as f:
        test_data = [json.loads(line) for line in f]
    
    results = []
    
    for item in test_data:
        question = item['messages'][0]['content']
        expected_answer = item['messages'][1]['content']
        
        # Generate model response
        model_response = generate_response(engine, tokenizer, question)
        
        # Compare responses
        similarity = compute_similarity(model_response, expected_answer)
        
        results.append({
            'question': question,
            'expected': expected_answer,
            'generated': model_response,
            'similarity': similarity
        })
        
        print(f"\nQ: {question}")
        print(f"Expected: {expected_answer[:100]}...")
        print(f"Generated: {model_response[:100]}...")
        print(f"Similarity: {similarity:.2f}")
    
    # Aggregate metrics
    avg_similarity = sum(r['similarity'] for r in results) / len(results)
    
    return {
        'average_similarity': avg_similarity,
        'total_questions': len(results),
        'detailed_results': results
    }


def generate_response(engine, tokenizer, question):
    """Generate response from model."""
    bos = tokenizer.get_bos_token_id()
    user_start = tokenizer.encode_special("<|user_start|>")
    user_end = tokenizer.encode_special("<|user_end|>")
    assistant_start = tokenizer.encode_special("<|assistant_start|>")
    assistant_end = tokenizer.encode_special("<|assistant_end|>")
    
    tokens = [bos, user_start]
    tokens.extend(tokenizer.encode(question))
    tokens.extend([user_end, assistant_start])
    
    response_tokens = []
    for token_column, _ in engine.generate(
        tokens,
        num_samples=1,
        max_tokens=256,
        temperature=0.3,  # Lower temperature for more consistent responses
        top_k=30
    ):
        token = token_column[0]
        if token == assistant_end or token == bos:
            break
        response_tokens.append(token)
    
    return tokenizer.decode(response_tokens)


def compute_similarity(response1, response2):
    """
    Compute similarity between two responses.
    Multiple approaches combined.
    """
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    import re
    
    # 1. Semantic similarity (TF-IDF cosine)
    vectorizer = TfidfVectorizer()
    try:
        tfidf = vectorizer.fit_transform([response1, response2])
        semantic_sim = cosine_similarity(tfidf[0:1], tfidf[1:2])[0][0]
    except:
        semantic_sim = 0.0
    
    # 2. Sentiment/stance alignment
    stance_sim = compute_stance_alignment(response1, response2)
    
    # 3. Key phrase matching
    phrase_sim = compute_phrase_overlap(response1, response2)
    
    # Combined score
    final_score = (
        0.4 * semantic_sim +
        0.4 * stance_sim +
        0.2 * phrase_sim
    )
    
    return final_score


def compute_stance_alignment(text1, text2):
    """Check if both texts have similar stance (agree/disagree/neutral)."""
    
    positive_words = {'agree', 'yes', 'support', 'favor', 'like', 'love', 'enjoy', 'strongly', 'definitely'}
    negative_words = {'disagree', 'no', 'oppose', 'against', 'dislike', 'hate', 'avoid', 'never'}
    
    def get_stance(text):
        text_lower = text.lower()
        pos_count = sum(1 for word in positive_words if word in text_lower)
        neg_count = sum(1 for word in negative_words if word in text_lower)
        
        if pos_count > neg_count:
            return 1  # Positive
        elif neg_count > pos_count:
            return -1  # Negative
        else:
            return 0  # Neutral
    
    stance1 = get_stance(text1)
    stance2 = get_stance(text2)
    
    # Same stance = 1.0, opposite = 0.0, one neutral = 0.5
    if stance1 == stance2:
        return 1.0
    elif stance1 == 0 or stance2 == 0:
        return 0.5
    else:
        return 0.0


def compute_phrase_overlap(text1, text2):
    """Compute overlap of meaningful phrases."""
    import re
    
    # Extract meaningful phrases (3+ char words)
    words1 = set(re.findall(r'\b\w{3,}\b', text1.lower()))
    words2 = set(re.findall(r'\b\w{3,}\b', text2.lower()))
    
    if not words1 or not words2:
        return 0.0
    
    overlap = len(words1 & words2)
    union = len(words1 | words2)
    
    return overlap / union if union > 0 else 0.0


# Run test
if __name__ == "__main__":
    results = test_persona_replication(
        model_source="sft",
        test_data_path="persona_test_set.jsonl"
    )
    
    print(f"\n{'='*60}")
    print(f"PERSONA REPLICATION TEST RESULTS")
    print(f"{'='*60}")
    print(f"Average Similarity: {results['average_similarity']:.2%}")
    print(f"Total Questions Tested: {results['total_questions']}")
    
    # Save detailed results
    with open('persona_test_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nDetailed results saved to: persona_test_results.json")
```

### Human Evaluation Protocol

For qualitative assessment:

```python
# human_eval_survey.py

def generate_human_eval_survey(test_results, output_html="eval_survey.html"):
    """
    Create an HTML survey for human evaluation.
    Presents question + model response + original response side-by-side.
    """
    
    html = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Persona Replication Evaluation</title>
        <style>
            body { font-family: Arial; max-width: 1200px; margin: 0 auto; padding: 20px; }
            .question-block { border: 1px solid #ccc; padding: 20px; margin: 20px 0; }
            .question { font-weight: bold; font-size: 18px; margin-bottom: 15px; }
            .responses { display: flex; gap: 20px; margin: 15px 0; }
            .response-box { flex: 1; padding: 15px; border-radius: 5px; }
            .original { background: #e3f2fd; }
            .generated { background: #fff3e0; }
            .rating { margin-top: 15px; }
            button { margin: 20px 0; padding: 10px 20px; }
        </style>
    </head>
    <body>
        <h1>Persona Replication Evaluation</h1>
        <p>Rate how well the model response matches the original person's likely response.</p>
        <form id="evalForm">
    """
    
    for i, result in enumerate(test_results['detailed_results']):
        html += f"""
        <div class="question-block">
            <div class="question">Question {i+1}: {result['question']}</div>
            <div class="responses">
                <div class="response-box original">
                    <strong>Original Response:</strong><br>
                    {result['expected']}
                </div>
                <div class="response-box generated">
                    <strong>Model Response:</strong><br>
                    {result['generated']}
                </div>
            </div>
            <div class="rating">
                <strong>How well does the model match the original persona?</strong><br>
                <label><input type="radio" name="q{i}" value="1"> 1 - Very different</label>
                <label><input type="radio" name="q{i}" value="2"> 2 - Somewhat different</label>
                <label><input type="radio" name="q{i}" value="3"> 3 - Neutral</label>
                <label><input type="radio" name="q{i}" value="4"> 4 - Somewhat similar</label>
                <label><input type="radio" name="q{i}" value="5"> 5 - Very similar</label>
            </div>
        </div>
        """
    
    html += """
        <button type="submit">Submit Evaluation</button>
        </form>
        <script>
            document.getElementById('evalForm').onsubmit = function(e) {
                e.preventDefault();
                const formData = new FormData(e.target);
                console.log(Object.fromEntries(formData));
                alert('Evaluation saved! Check console for results.');
            };
        </script>
    </body>
    </html>
    """
    
    with open(output_html, 'w') as f:
        f.write(html)
    
    print(f"Human evaluation survey saved to: {output_html}")
    print(f"Open in browser to complete evaluation.")
```

---

## Evaluation Metrics

### 1. **Quantitative Metrics**

#### Response Similarity Score
```python
similarity_score = (
    0.4 * semantic_similarity +  # TF-IDF/embedding distance
    0.4 * stance_alignment +     # Agree/disagree/neutral match
    0.2 * phrase_overlap         # Key phrase matching
)

# Target: > 0.75 for good persona replication
```

#### Answer Accuracy (for survey questions)
```python
# For Likert scale questions
answer_accuracy = percentage_within_1_point

# Example: Original = 4/5, Model = 5/5 → Counts as correct
# Example: Original = 2/5, Model = 5/5 → Counts as incorrect
```

#### Consistency Score
```python
# Ask same question multiple times with rephrasing
consistency = percentage_of_consistent_responses

# Example:
# "Do you like remote work?" → "Strongly agree"
# "What's your opinion on working from home?" → "I'm against it" ❌ Inconsistent
```

### 2. **Qualitative Metrics**

#### Style Match
- [ ] Uses similar vocabulary
- [ ] Similar sentence complexity
- [ ] Similar level of formality
- [ ] Similar emotional expression

#### Opinion Alignment
- [ ] Matches on controversial topics
- [ ] Matches on personal preferences
- [ ] Matches on priorities/values
- [ ] Matches on reasoning patterns

#### Knowledge Domain Match
- [ ] Shows knowledge in same areas
- [ ] Admits uncertainty in same areas
- [ ] Uses similar examples/references

### 3. **Composite Persona Fidelity Score**

```python
def calculate_fidelity_score(test_results, human_ratings):
    """
    Combine multiple metrics into overall fidelity score.
    
    Returns score from 0.0 (poor) to 1.0 (perfect replication)
    """
    
    # Automated metrics (60%)
    avg_similarity = test_results['average_similarity']  # 30%
    consistency_score = compute_consistency(test_results)  # 15%
    stance_accuracy = compute_stance_accuracy(test_results)  # 15%
    
    # Human ratings (40%)
    human_score = sum(human_ratings) / (len(human_ratings) * 5)  # 40%
    
    fidelity = (
        0.30 * avg_similarity +
        0.15 * consistency_score +
        0.15 * stance_accuracy +
        0.40 * human_score
    )
    
    return fidelity


# Interpretation:
# 0.90-1.00: Excellent replication
# 0.75-0.90: Good replication
# 0.60-0.75: Fair replication
# 0.00-0.60: Poor replication
```

---

## Implementation Guide

### Complete Workflow

```bash
# 1. Collect survey data from source persona
python collect_survey_data.py --persona alex --output surveys/alex_raw.json

# 2. Convert to conversation format
python survey_to_conversations.py \
    --input surveys/alex_raw.json \
    --output ~/.cache/nanochat/persona_survey_conversations.jsonl

# 3. Augment with multi-turn dialogues and variations
python augment_conversations.py \
    --input ~/.cache/nanochat/persona_survey_conversations.jsonl \
    --output ~/.cache/nanochat/persona_augmented.jsonl \
    --multiplier 3

# 4. Split into train/test
python split_data.py \
    --input ~/.cache/nanochat/persona_augmented.jsonl \
    --train-ratio 0.8

# 5. Train base model (standard)
torchrun --standalone --nproc_per_node=8 -m scripts.base_train -- --depth=20

# 6. Midtraining with persona emphasis
torchrun --standalone --nproc_per_node=8 -m scripts.mid_train

# 7. SFT with heavy persona focus
torchrun --standalone --nproc_per_node=8 -m scripts.chat_sft -- --num_epochs=3

# 8. Test persona replication
python test_persona_replication.py \
    --model sft \
    --test-data ~/.cache/nanochat/persona_test.jsonl \
    --output results/replication_results.json

# 9. Human evaluation
python generate_human_eval_survey.py \
    --results results/replication_results.json \
    --output eval_survey.html

# 10. Calculate final fidelity score
python calculate_fidelity.py \
    --auto-results results/replication_results.json \
    --human-ratings eval_responses.json
```

---

## Tips for High Fidelity

### 1. **Data Quality Over Quantity**

✅ **Better**: 300 well-designed, diverse questions with reasoning
❌ **Worse**: 1000 superficial yes/no questions

### 2. **Capture Reasoning, Not Just Answers**

Always ask "Why?" after each survey question:
```
Q: How do you feel about public speaking?
A: Uncomfortable (2/5)
WHY: I get anxious about being judged and prefer writing to express ideas clearly.
```

### 3. **Include Edge Cases**

Test persona on:
- Controversial topics where they have strong views
- Topics they know nothing about
- Hypothetical scenarios
- Contradictory situations

### 4. **Diversity in Question Phrasing**

Ask the same thing multiple ways:
```
"Do you enjoy social events?"
"How do you feel about going to parties?"
"Would you rather stay home or go out on weekends?"
```

### 5. **Temporal Consistency**

Re-administer key questions 1-2 weeks later to ensure persona stability.

### 6. **Multi-Domain Coverage**

Don't just cover one area. Spread across:
- Work/career
- Social life
- Politics
- Technology
- Arts/culture
- Personal habits
- Future aspirations

### 7. **Training Recipe for Maximum Fidelity**

```python
# Optimal training mixture for persona replication:

# Midtraining (lines 98-106 in mid_train.py)
train_dataset = TaskMixture([
    SmolTalk(split="train", stop=100000),  # 15% - Basic conversation
    MMLU(subset="auxiliary_train", split="train"),  # 15% - Reasoning
    GSM8K(subset="main", split="train"),  # 1% - Math
    
    # 69% persona data (repeat 5x)
    CustomJSON(filepath=persona_survey_path),
    CustomJSON(filepath=persona_survey_path),
    CustomJSON(filepath=persona_survey_path),
    CustomJSON(filepath=persona_survey_path),
    CustomJSON(filepath=persona_survey_path),
])

# SFT (lines 84-92 in chat_sft.py)
train_ds = TaskMixture([
    SmolTalk(split="train", stop=3000),  # 10% - Maintain fluency
    
    # 90% persona data (repeat 9x)
    CustomJSON(filepath=persona_survey_path),
    CustomJSON(filepath=persona_survey_path),
    CustomJSON(filepath=persona_survey_path),
    CustomJSON(filepath=persona_survey_path),
    CustomJSON(filepath=persona_survey_path),
    CustomJSON(filepath=persona_survey_path),
    CustomJSON(filepath=persona_survey_path),
    CustomJSON(filepath=persona_survey_path),
    CustomJSON(filepath=persona_survey_path),
])

# Train with 3 epochs in SFT
--num_epochs=3
```

### 8. **Use Lower Temperature for Testing**

```python
# More deterministic = more consistent persona
engine.generate(
    tokens,
    temperature=0.2,  # Lower than default 0.7
    top_k=20          # Lower than default 50
)
```

### 9. **Validate Across Time**

Test the model:
- Immediately after training
- 1 week later (same test)
- With rephrased questions
- In multi-turn conversations

### 10. **Red Team Your Persona**

Try to find inconsistencies:
```
"Earlier you said you prefer remote work, but now you're saying you like office environments. Which is it?"
```

Model should maintain consistency or acknowledge nuance.

---

## Complete Example

### Scenario: Replicating "Alex" - An Introverted Software Engineer

#### Step 1: Survey Design

**File**: `alex_survey.json`

```json
{
  "persona": {
    "name": "Alex Rivera",
    "age": 32,
    "occupation": "Senior Software Engineer",
    "location": "Seattle, WA"
  },
  "responses": [
    {
      "category": "Work Style",
      "question": "I prefer working alone rather than in teams.",
      "answer": 4,
      "scale": "1-5 (1=Strongly Disagree, 5=Strongly Agree)",
      "reasoning": "I'm most productive when I can focus deeply without interruptions. Teams are fine for brainstorming but execution works better solo for me."
    },
    {
      "category": "Work Style",
      "question": "How important is work-life balance?",
      "answer": 5,
      "reasoning": "Extremely important. I have hobbies (rock climbing, reading) that energize me. Work shouldn't consume life."
    },
    {
      "category": "Social",
      "question": "I enjoy attending large social events.",
      "answer": 2,
      "reasoning": "Large groups drain me. I much prefer 1-on-1 coffee chats or small gatherings of close friends."
    },
    {
      "category": "Technology",
      "question": "What's your opinion on AI's impact on software development?",
      "answer": "I'm cautiously optimistic. AI tools like Copilot boost productivity but we need to think about job displacement and ethical implications. It's a tool, not a replacement for human creativity and judgment.",
      "confidence": 4
    },
    {
      "category": "Politics",
      "question": "Government should regulate tech companies more strictly.",
      "answer": 4,
      "reasoning": "Yes, especially around data privacy and monopolistic practices. Self-regulation hasn't worked. But regulations should be thoughtful, not knee-jerk."
    }
    // ... 200+ more questions
  ]
}
```

#### Step 2: Convert & Augment

```bash
python survey_to_conversations.py --input alex_survey.json --output alex_base.jsonl
python augment_conversations.py --input alex_base.jsonl --output alex_full.jsonl
```

Result: 450 → 1,800 conversations after augmentation.

#### Step 3: Train

```bash
# Copy to cache
cp alex_full.jsonl ~/.cache/nanochat/persona_survey_conversations.jsonl

# Run full training pipeline
bash speedrun.sh  # (with modified mid_train.py and chat_sft.py)
```

#### Step 4: Test

Create test questions:

```json
{
  "messages": [
    {"role": "user", "content": "Do you prefer open office layouts or private offices?"},
    {"role": "assistant", "content": "I strongly prefer private offices or quiet spaces. Open offices are my nightmare - constant noise and interruptions make deep focus nearly impossible. I'd take remote work over open office any day."}
  ]
}
```

Run evaluation:

```bash
python test_persona_replication.py --model sft --test-data alex_test.jsonl
```

#### Step 5: Results

```
PERSONA REPLICATION TEST RESULTS
================================================================
Average Similarity: 82.3%
Total Questions Tested: 90

Category Breakdown:
- Work Style: 89% match
- Social Preferences: 85% match  
- Technology Opinions: 78% match
- Political Views: 76% match
- Personal Habits: 83% match

Overall Fidelity Score: 0.81 (Good replication)
```

---

## Conclusion

### Recommended Approach

1. **Collect 300-500 survey questions** across personality, values, knowledge domains
2. **Include reasoning** for every answer (critical!)
3. **Augment to 1,500-2,500 conversations** with multi-turn dialogues
4. **Train with 70-90% persona emphasis** in mid/SFT stages
5. **Test on held-out 20%** with both automated and human evaluation
6. **Target fidelity score** of 0.75+ for good replication

### Best Data Sources Summary

🥇 **Best**: Custom comprehensive survey (300-500 questions) covering:
- Big Five personality items (100)
- Values & beliefs (100)
- Domain knowledge & opinions (150)
- Open-ended reasoning (50)

🥈 **Good**: Combination of existing surveys:
- 16 Personalities + Political Compass + domain surveys
- Supplemented with interview transcripts
- ~200-300 total data points

🥉 **Minimum**: Single comprehensive personality test + custom questions
- MBTI or Big Five (100 questions)
- Custom domain questions (100)
- ~200 total data points

---

**Remember**: Quality of reasoning > quantity of questions. One well-reasoned response is worth 10 yes/no answers for persona replication.

---

*Generated on: October 29, 2025*  
*Repository: [github.com/karpathy/nanochat](https://github.com/karpathy/nanochat)*

