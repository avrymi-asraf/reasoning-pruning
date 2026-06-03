# /// script
# dependencies = []
# ///

import re

def clean_text(text):
    if not text:
        return ""
    return " ".join(text.split())

def get_arithmetic_template(question):
    q = clean_text(question)
    
    # 1. Replace numbers: matches 123, 123.45, 1,234, etc. Also handle currencies/percentages like $50, 50%
    q_norm = re.sub(r'\$?\b\d+(?:[\.,]\d+)?%?\b', '<NUM>', q)
    
    # 2. Replace pronouns
    q_norm = re.sub(r'\b(he|she|him|her|his|hers|himself|herself)\b', '<PRONOUN>', q_norm, flags=re.IGNORECASE)
    
    # 3. Replace names and common capitalized words
    sentences = re.split(r'(?<=[.!?])\s+', q_norm)
    norm_sentences = []
    
    common_cap_words = {
        "I", "We", "You", "They", "He", "She", "It", "The", "A", "An", "In", "On", "At", "By", "For", "With", "About",
        "Against", "Between", "Into", "Through", "During", "Before", "After", "Above", "Below", "To", "From", "Up",
        "Down", "Out", "Off", "Over", "Under", "Again", "Further", "Then", "Once", "Here", "There", "When", "Where",
        "Why", "How", "All", "Any", "Both", "Each", "Few", "More", "Most", "Other", "Some", "Such", "No", "Nor", "Not",
        "Only", "Own", "Same", "So", "Than", "Too", "Very", "S", "T", "Can", "Will", "Just", "Should", "Now",
        "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday",
        "January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December",
        "Yesterday", "Today", "Tomorrow", "Last", "Next", "This", "Every", "First", "Second", "Third", 
        "One", "Two", "Three", "Four", "Five", "Six", "Seven", "Eight", "Nine", "Ten"
    }
    
    for sentence in sentences:
        words = sentence.split()
        if not words:
            continue
        new_words = []
        first_word = words[0]
        first_word_clean = re.sub(r'[^\w]', '', first_word)
        if first_word_clean.istitle() and first_word_clean not in common_cap_words and first_word_clean not in ["What", "Which", "Who", "How", "If", "Is", "Are", "Does", "Do", "Did", "Has", "Have", "Had", "Can", "Could", "Would", "Should", "On", "In", "At", "For", "While", "When", "As"]:
            new_words.append('<NAME>')
        else:
            new_words.append(first_word)
            
        for word in words[1:]:
            word_clean = re.sub(r'[^\w]', '', word)
            if word_clean.istitle() and word_clean not in common_cap_words:
                new_words.append('<NAME>')
            else:
                new_words.append(word)
        norm_sentences.append(" ".join(new_words))
        
    template = " ".join(norm_sentences).lower()
    return template
