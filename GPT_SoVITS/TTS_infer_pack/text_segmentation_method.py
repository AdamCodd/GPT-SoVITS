import re
from typing import Callable

punctuation = set(['!', '?', '…', ',', '.', '-'," "])
METHODS = dict()

def get_method(name:str)->Callable:
    method = METHODS.get(name, None)
    if method is None:
        raise ValueError(f"Method {name} not found")
    return method

def get_method_names()->list:
    return list(METHODS.keys())

def register_method(name):
    def decorator(func):
        METHODS[name] = func
        return func
    return decorator

splits = {"，", "。", "？", "！", ",", ".", "?", "!", "~", ":", "：", "—", "…", }

def split_big_text(text, max_len=510):
    punctuation = "".join(splits)
    # Cutting text
    segments = re.split('([' + punctuation + '])', text)
    # Initialize the results list and current fragment
    result = []
    current_segment = ''
    
    for segment in segments:
        # If the length of the current fragment plus the new fragment exceeds max_len, add the current fragment to the result list and reset the current fragment.
        if len(current_segment + segment) > max_len:
            result.append(current_segment)
            current_segment = segment
        else:
            current_segment += segment
    # Add the last fragment to the list of results
    if current_segment:
        result.append(current_segment)
    
    return result

def split(todo_text):
    todo_text = todo_text.replace("……", "。").replace("——", "，")
    if todo_text[-1] not in splits:
        todo_text += "。"
    i_split_head = i_split_tail = 0
    len_text = len(todo_text)
    todo_texts = []
    while 1:
        if i_split_head >= len_text:
            break  # There must be punctuation at the end, so just skip ahead, the last paragraph was added last time.
        if todo_text[i_split_head] in splits:
            i_split_head += 1
            todo_texts.append(todo_text[i_split_tail:i_split_head])
            i_split_tail = i_split_head
        else:
            i_split_head += 1
    return todo_texts

# Not cutting
@register_method("cut0")
def cut0(inp):
    if not set(inp).issubset(punctuation):
        return inp
    else:
        return "/n"

# Get four sentences at a time
@register_method("cut1")
def cut1(inp):
    inp = inp.strip("\n")
    inps = split(inp)
    split_idx = list(range(0, len(inps), 4))
    split_idx[-1] = None
    if len(split_idx) > 1:
        opts = []
        for idx in range(len(split_idx) - 1):
            opts.append("".join(inps[split_idx[idx]: split_idx[idx + 1]]))
    else:
        opts = [inp]
    opts = [item for item in opts if not set(item).issubset(punctuation)]
    return "\n".join(opts)

# Get 50 words at a time
@register_method("cut2")
def cut2(inp):
    inp = inp.strip("\n")
    inps = split(inp)
    if len(inps) < 2:
        return inp
    opts = []
    summ = 0
    tmp_str = ""
    for i in range(len(inps)):
        summ += len(inps[i])
        tmp_str += inps[i]
        if summ > 50:
            summ = 0
            opts.append(tmp_str)
            tmp_str = ""
    if tmp_str != "":
        opts.append(tmp_str)
    if len(opts) > 1 and len(opts[-1]) < 50: # If the last one is too short, combine it with the first one
        opts[-2] = opts[-2] + opts[-1]
        opts = opts[:-1]
    opts = [item for item in opts if not set(item).issubset(punctuation)]
    return "\n".join(opts)

# Check for the Chinese period, then cut.
@register_method("cut3")
def cut3(inp):
    inp = inp.strip("\n")
    opts = ["%s" % item for item in inp.strip("。").split("。")]
    opts = [item for item in opts if not set(item).issubset(punctuation)]
    return "\n".join(opts)

@register_method("cut4")
def cut4(inp):
    inp = inp.strip("\n")
    opts = ["%s" % item for item in inp.strip(".").split(".")]
    opts = [item for item in opts if not set(item).issubset(punctuation)]
    return "\n".join(opts)

# Cut by punctuation
# contributed by https://github.com/AI-Hobbyist/GPT-SoVITS/blob/main/GPT_SoVITS/inference_webui.py
@register_method("cut5")
def cut5(inp):
    inp = inp.strip("\n")
    punds = {',', '.', ';', '?', '!', '、', '，', '。', '？', '！', ';', '：', '…'}
    mergeitems = []
    items = []

    for i, char in enumerate(inp):
        if char in punds:
            if char == '.' and i > 0 and i < len(inp) - 1 and inp[i - 1].isdigit() and inp[i + 1].isdigit():
                items.append(char)
            else:
                items.append(char)
                mergeitems.append("".join(items))
                items = []
        else:
            items.append(char)

    if items:
        mergeitems.append("".join(items))

    opt = [item for item in mergeitems if not set(item).issubset(punds)]
    return "\n".join(opt)

# Improved segmentation method for EN, a bit slower compared to cut5: O(n) vs O(n*m).
@register_method("cut6")
def cut6(inp):
    if not isinstance(inp, str):
        raise TypeError("Input must be a string")
    
    inp = inp.strip("\n")
    
    # Enhanced abbreviations list
    abbreviations = {
        'mr.', 'mrs.', 'dr.', 'ms.', 'prof.', 'sr.', 'jr.',
        'u.s.a.', 'u.k.', 'i.e.', 'e.g.', 'etc.', 'inc.', 
        'st.', 'ltd.', 'col.', 'gen.', 'govt.', 'dist.',
        'bros.', 'corp.', 'dept.', 'univ.', 'assn.', 'ave.'
    }

    # Compile all regex patterns
    abbr_pattern = re.compile(r'\b(?:[A-Za-z]\.|(?:' + '|'.join(map(re.escape, abbreviations)) + r'))\b')
    ordinal_pattern = re.compile(r'\b\d+(?:st|nd|rd|th)\b', re.IGNORECASE)
    time_pattern = re.compile(r'\b\d{1,2}:\d{2}\b')
    acronym_pattern = re.compile(r'\b[A-Z]{2,}\b')
    list_pattern = re.compile(r'^\s*(?:\d+\.|[•\-*])\s+')

    # Punctuation set
    punds = {',', '.', ';', '?', '!', '…', ':', '；', '：'}
    
    # Pre-process text
    inp = re.sub(r'\.{3,}', '…', inp)  # Normalize ellipsis
    inp = re.sub(r'--+', '—', inp)     # Normalize dashes
    
    items = []
    current_segment = []
    quote_stack = []
    i = 0
    
    def handle_nested_quotes(char):
        nonlocal quote_stack
        if char in '"\'':
            if not quote_stack or quote_stack[-1] != char:
                quote_stack.append(char)
                return True
            else:
                quote_stack.pop()
                return False
        return False

    while i < len(inp):
        try:
            char = inp[i]
            
            # Check for abbreviations
            match = abbr_pattern.match(inp, i)
            if match:
                current_segment.append(match.group())
                i += len(match.group())
                continue
            
            # Check for ordinals
            match = ordinal_pattern.match(inp, i)
            if match:
                current_segment.append(match.group())
                i += len(match.group())
                continue
            
            # Check for time formats
            match = time_pattern.match(inp, i)
            if match:
                current_segment.append(match.group())
                i += len(match.group())
                continue
            
            # Check for acronyms
            match = acronym_pattern.match(inp, i)
            if match:
                current_segment.append(match.group())
                i += len(match.group())
                continue
            
            # Handle decimal numbers
            if (char == '.' and i > 0 and i < len(inp) - 1 
                and inp[i-1].isdigit() and inp[i+1].isdigit()):
                current_segment.append(char)
            
            # Handle quotations and parentheses
            elif char in '"\'([{':
                current_segment.append(char)
                handle_nested_quotes(char)
            
            elif char in '"\')]}':
                current_segment.append(char)
                is_opening = handle_nested_quotes(char)
                
                # Only split if this is a closing quote and no other quotes are open
                if not is_opening and not quote_stack:
                    if i + 1 < len(inp) and inp[i + 1] in punds:
                        current_segment.append(inp[i + 1])
                        items.append(''.join(current_segment))
                        current_segment = []
                        i += 1
            
            # Handle regular punctuation
            elif char in punds and not quote_stack:  # Don't split if inside quotes
                current_segment.append(char)
                items.append(''.join(current_segment))
                current_segment = []
            
            else:
                current_segment.append(char)
            
            i += 1
            
        except re.error:
            # Handle regex matching errors by treating the character as normal text
            current_segment.append(char)
            i += 1
    
    if current_segment:
        items.append(''.join(current_segment))
    
    # Process segments
    processed_items = []
    for item in items:
        # Handle list items
        if list_pattern.match(item):
            processed_items.append(item)
            continue
            
        # Clean and validate segment
        cleaned_item = item.strip()
        if cleaned_item and not set(cleaned_item).issubset(punds):
            processed_items.append(cleaned_item)
    
    # Merge short segments intelligently
    merged = []
    i = 0
    while i < len(processed_items):
        current_item = processed_items[i]
        
        # Check if current segment should be merged
        should_merge = (
            i < len(processed_items) - 1 and
            len(current_item) < 15 and
            not current_item.endswith(('.', '?', '!', '…')) and
            not list_pattern.match(current_item) and
            not list_pattern.match(processed_items[i + 1])
        )
        
        if should_merge:
            merged.append(f"{current_item} {processed_items[i + 1]}")
            i += 2
        else:
            merged.append(current_item)
            i += 1
    
    return "\n".join(merged)

# Improved segmentation method for JP, a bit slower compared to cut5: O(n) vs O(n * max(k,p,m)).
@register_method("cut7")
def cut7(inp):
    if not isinstance(inp, str):
        raise TypeError("Input must be a string")
    
    inp = inp.strip("\n")
    
    JAPANESE_PUNCTUATION = {
        "。", "、", "？", "！", "：", "；", "…", "︙", "⋯", "‥",
        "―", "−", "ー", "─", "『", "』", "「", "」", "（", "）", "【", "】", "・", "·",
        "'", "'", """, """  # Added Western quotes
    }
    
    JAPANESE_ABBREVIATIONS = {
        # Original abbreviations
        '株式会社', '有限会社', '合同会社', '社団法人', '財団法人',
        '国立', '都立', '府立', '県立', '市立', '大学院', '研究所', '研究室',
        'AM', 'PM', '午前', '午後', '特定非営利活動法人', '国際連合',
        '東京都', '国土交通省', '厚生労働省', '金融庁', '財務省', '内閣府', '警察庁',
        # Added titles
        '教授', '准教授', '講師', '助教',
        '代表取締役', '取締役', '執行役員', '部長', '課長', '係長',
        # Added organizations
        '独立行政法人', '一般社団法人', '一般財団法人'
    }
    
    QUOTE_PAIRS = {
        '「': '」', '『': '』', '（': '）', '【': '】',
        '"': '"', ''': ''',  # Added Western quotes
    }
    
    def handle_quotes(text, pos):
        if text[pos] in QUOTE_PAIRS:
            end_pos = pos + 1
            opening_quote = text[pos]
            depth = 1
            while end_pos < len(text) and depth > 0:
                if text[end_pos] == opening_quote:
                    depth += 1
                elif text[end_pos] == QUOTE_PAIRS[opening_quote]:
                    depth -= 1
                end_pos += 1
            return end_pos
        return pos + 1
    
    def handle_number_expression(text, pos):
        patterns = [
            r'\d{1,3}(?:,\d{3})*(?:\.\d+)?[万億兆]?円',  # Numbers with commas
            r'\d+(?:\.\d+)?[万億兆]?円',
            r'\d+(?:\.\d+)?%',  # Percentages
            r'\d+年\d+月\d+日',
            r'\d+時\d+分',
            r'\d+(?:\.\d+)?[kmKM]?[mgML]',
            r'\d+[～〜]\d+[年月日時分秒]',  # Ranges
            r'\d+年|\d+月|\d+日',
            r'\d+時|\d+分|\d+秒'
        ]
        for pattern in patterns:
            match = re.match(pattern, text[pos:])
            if match:
                return pos + match.end()
        return pos + 1

    def should_break_segment(text, pos, context_window=5):
        if pos >= len(text):
            return True
            
        prev_text = text[max(0, pos-context_window):pos]
        next_text = text[pos:min(len(text), pos+context_window)]
        
        if re.match(r'\d', next_text):
            return False
        
        particles = {'は', 'が', 'を', 'に', 'へ', 'で', 'と', 'より', 'から', 'まで', 'など'}  # Added particles
        if prev_text.strip()[-1:] in particles:
            return False
        
        return True
    
    # Rest of the code remains the same as it's working well
    inp = re.sub(r'\.{3,}', '…', inp)
    inp = re.sub(r'--+', '─', inp)
    
    abbr_pattern = re.compile(r'\b(?:' + '|'.join(map(re.escape, JAPANESE_ABBREVIATIONS)) + r')\b')
    kanji_kana_pattern = re.compile(r'[\u4e00-\u9faf\u3040-\u309f\u30a0-\u30ff]+')
    list_pattern = re.compile(r'^\s*(?:\d+\.|[•\-*])\s+')
    
    items = []
    current_segment = []
    i = 0
    
    while i < len(inp):
        match = abbr_pattern.match(inp, i)
        if match:
            current_segment.append(match.group())
            i += len(match.group())
            continue
        
        if inp[i].isdigit():
            new_pos = handle_number_expression(inp, i)
            current_segment.append(inp[i:new_pos])
            i = new_pos
            continue
        
        if inp[i] in QUOTE_PAIRS:
            new_pos = handle_quotes(inp, i)
            current_segment.append(inp[i:new_pos])
            i = new_pos
            continue
        
        match = kanji_kana_pattern.match(inp, i)
        if match:
            current_segment.append(match.group())
            i += len(match.group())
            continue
        
        if inp[i] in JAPANESE_PUNCTUATION and should_break_segment(inp, i):
            current_segment.append(inp[i])
            if current_segment:
                items.append(''.join(current_segment))
            current_segment = []
            i += 1
        else:
            current_segment.append(inp[i])
            i += 1
    
    if current_segment:
        items.append(''.join(current_segment))
    
    processed_items = []
    for item in items:
        if list_pattern.match(item):
            processed_items.append(item)
            continue
        cleaned_item = item.strip()
        if cleaned_item and not set(cleaned_item).issubset(JAPANESE_PUNCTUATION):
            processed_items.append(cleaned_item)
    
    merged = []
    i = 0
    while i < len(processed_items):
        current_item = processed_items[i]
        
        should_merge = (
            i < len(processed_items) - 1 and
            len(current_item) < 15 and
            not current_item.endswith(tuple(JAPANESE_PUNCTUATION)) and
            not list_pattern.match(current_item) and
            not list_pattern.match(processed_items[i + 1])
        )
        
        if should_merge:
            merged.append(f"{current_item}{processed_items[i + 1]}")
            i += 2
        else:
            merged.append(current_item)
            i += 1
    
    return "\n".join(merged)

if __name__ == '__main__':
    # English test cases for cut0-cut6
    en_test_cases = [
        # Basic cases
        "This is a test. Another sentence! And one more?",
        "Hello, world. How are you today?",
        
        # Abbreviation cases
        "Hello Mr. Smith! How are you today? I saw Dr. Johnson at 3.14 Main St.",
        "U.S.A. is a country. The U.K. is another.",
        "Please visit our Corp. headquarters on Ave. A.",
        
        # Numbers and special formats
        "This is the 1st example. It costs $19.99 today.",
        "The time is 9:30. The meeting starts at 10:15.",
        "The temp. was 72.5 degrees F.",
        
        # Quotes and nested punctuation
        'He said "Don\'t go there!" and left.',
        'She asked "What about the U.S. policy?" in the meeting.',
        
        # Lists and bullet points
        "1. First item\n2. Second item\n• Bullet point",
        
        # Ellipsis and dashes
        "Chapter 1. Introduction... This is a test.",
        "Word--another word---and another--done.",
        
        # Mixed cases
        "Mr. Smith (Ph.D., M.D.) works at 123 Main St., Suite 456.",
        "The meeting is at 3:30 P.M. in Rm. 101."
    ]

    # Japanese test cases for cut7
    jp_test_cases = [
        # Basic cases
        "こんにちは。今日はいい天気ですね。",
        "私は日本人です。東京に住んでいます！",
        
        # Abbreviation cases
        "株式会社タナカは、東京都に本社があります。",
        "国立大学の研究所で研究をしています。",
        "午前9時から午後5時まで営業しています。",
        
        # Numbers and date/time formats
        "2024年3月15日に開催される予定です。",
        "価格は1,234,567円です。",
        "営業時間は10時30分から18時00分までです。",
        "約25.5%の確率で発生します。",
        "1～3年後に完成予定です。",
        
        # Quotes and nested punctuation
        "田中さんは「明日は『晴れ』でしょう」と言いました。",
        "【重要】「会議は（午後3時から）開始」です。",
        
        # Lists and bullet points
        "1. 企画書作成\n2. 予算検討\n• 実施計画",
        
        # Particles and connections
        "私は学校へ行きます。友達と会います。",
        "本を読んでから、宿題をします。",
        
        # Titles and positions
        "山田教授は研究室で実験をしています。",
        "代表取締役の鈴木氏が会議に参加します。",
        
        # Mixed cases with measurements
        "体重は65.5kgで、身長は172.5cmです。",
        "約500mLの水を3回に分けて飲みます。",
        
        # Organization names and addresses
        "独立行政法人国際協力機構は支援を行っています。",
        "東京都千代田区丸の内1-1-1",
        
        # Long compound sentences
        "昨日、私は友達と映画を見に行って、その後でラーメンを食べて、最後に本屋に寄って帰りました。",
        
        # Multiple punctuation types
        "はい！そうですね。でも、どうしましょうか？",
        
        # Technical terms and units
        "CPU速度は2.4GHzで、メモリは8GBです。",
        "室温を20℃から25℃に設定してください。"
    ]
    
    # Test English cases with all methods except cut7
    methods = get_method_names()
    methods.remove('cut7')  # Remove cut7 from English tests
    
    print("\nTesting English text segmentation:")
    for method_name in methods:
        print(f"\nTesting {method_name}:")
        method = get_method(method_name)
        for test in en_test_cases:
            print("\nInput:", test)
            print("Output:", method(test))
            print("-" * 50)

    # Test Japanese cases with cut7 only
    print("\nTesting Japanese text segmentation with cut7:")
    cut7_method = get_method('cut7')
    for test in jp_test_cases:
        print("\nInput:", test)
        print("Output:", cut7_method(test))
        print("-" * 50)
