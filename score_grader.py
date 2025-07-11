def parse_phonemes_phrase(phonetic_string):
    """
    将音标字符串解析为单独的音素，支持更广泛的多字符音素（特别是针对美式发音）
    参数:
        phonetic_string: 如 'nəʊrθ' 或 '/nɔːθ əˈmɛrɪkən/' 的字符串
    返回:
        音素列表: ['n', 'əʊ', 'r', 'θ']
    """
    # 标准化输入字符串，移除常见符号
    if "ː" in phonetic_string:
        phonetic_string = phonetic_string.replace('ː', ':')
    if "əʊ" in phonetic_string: # 兼容əʊ
        pass
    phonetic_string = phonetic_string.replace('(', '').replace(')', '').replace('/', '').replace('ˈ', '').replace('ˌ', '').replace('.', ' ').strip()
    
    # 定义多字符音素（顺序很重要，长的在前）
    # 增加了 əʊ, ɔ:, ər, ɪr, ɛr, ʊr, ɔr, ɑr 等常见组合
    multi_char_phonemes = [
        # R-colored vowels
        'ər', 'ɪr', 'ɛr', 'ʊr', 'ɔr', 'ɑr',
        # Diphthongs (双元音)
        'aɪ', 'aʊ', 'ɔɪ', 'eɪ', 'oʊ', 'əʊ', 
        # Other diphthongs
        'ɪə', 'eə', 'ʊə',
        # Long vowels (长元音)
        'i:', 'u:', 'ɑ:', 'ɔ:', 'ɜ:',
        # Affricates (塞擦音)
        'tʃ', 'dʒ',
        # Other common multi-char symbols
        'θ', 'ð', 'ʃ', 'ʒ', 'ŋ'
    ]
    
    phonemes = []
    i = 0
    
    while i < len(phonetic_string):
        # 跳过空格
        if phonetic_string[i] == ' ':
            i += 1
            continue
            
        # 优先匹配多字符音素
        matched = False
        for multi_phoneme in multi_char_phonemes:
            if phonetic_string[i:].startswith(multi_phoneme):
                phonemes.append(multi_phoneme)
                i += len(multi_phoneme)
                matched = True
                break
        
        # 没有多字符匹配则加单字符
        if not matched:
            phonemes.append(phonetic_string[i])
            i += 1
            
    return phonemes

def parse_phonemes(phonetic_string):
    """
    将音标字符串解析为单独的音素，支持多字符音素
    参数:
        phonetic_string: 如 'tʃɔɪs' 或 'laɪk' 的字符串
    返回:
        音素列表: ['tʃ', 'ɔɪ', 's'] 或 ['l', 'aɪ', 'k']
    """
     # 新增：处理全角冒号和括号
    if "ː" in phonetic_string:
        print("Warning: 全角冒号可能导致解析错误，请使用半角冒号")
        phonetic_string = phonetic_string.replace('ː', ':')
    if '(' in phonetic_string or ')' in phonetic_string:
        phonetic_string = phonetic_string.replace('(', '').replace(')', '')
    if '/' in phonetic_string:
        phonetic_string = phonetic_string.replace('/', '')   
    
    
    # 定义多字符音素（顺序很重要，长的在前）
    multi_char_phonemes = [
        # 塞擦音
        'tʃ', 'dʒ',
        # 双元音
        'aɪ', 'aʊ', 'ɔɪ', 'eɪ', 'oʊ', 'ɪə', 'eə', 'ʊə',
        # 长元音（如果用到长度标记）
        'iː', 'uː', 'ɑː', 'ɔː', 'ɜː',
        # 其他常见组合
        'θ', 'ð', 'ʃ', 'ʒ', 'ŋ'
    ]
    
    phonemes = []
    i = 0
     
    while i < len(phonetic_string):
        # 跳过重音和音节分界符
        if phonetic_string[i] in ['ˈ', 'ˌ', '.']:
            i += 1
            continue
            
        # 优先匹配多字符音素
        matched = False
        for multi_phoneme in multi_char_phonemes:
            if phonetic_string[i:].startswith(multi_phoneme):
                phonemes.append(multi_phoneme)
                i += len(multi_phoneme)
                matched = True
                break
        
        # 没有多字符匹配则加单字符
        if not matched:
            phonemes.append(phonetic_string[i])
            i += 1
    
    return phonemes

def get_phonetic_distance(phoneme1, phoneme2):
    """
    根据发音特征计算两个音素的距离
    返回值为0（完全相同）到1（完全不同）
    """
    if phoneme1 == phoneme2:
        return 0.0
    
    # 定义常见音素对的相似度权重
    similarity_groups = {
        # 清浊对（差别小）
        ('p', 'b'): 0.2, ('t', 'd'): 0.2, ('k', 'g'): 0.2,
        ('f', 'v'): 0.2, ('s', 'z'): 0.2, ('ʃ', 'ʒ'): 0.2,
        ('θ', 'ð'): 0.2, ('tʃ', 'dʒ'): 0.2,
        
        # 发音部位（中等差别）
        ('p', 't'): 0.4, ('t', 'k'): 0.4, ('b', 'd'): 0.4, ('d', 'g'): 0.4,
        ('f', 'θ'): 0.4, ('v', 'ð'): 0.4, ('s', 'ʃ'): 0.4, ('z', 'ʒ'): 0.4,
        
        # 常见二语替换（中等差别）
        ('θ', 's'): 0.3, ('ð', 'z'): 0.3, ('θ', 'f'): 0.3, ('ð', 'v'): 0.3,
        ('r', 'l'): 0.3, ('v', 'w'): 0.3, ('b', 'v'): 0.3,
        
        # 元音高低/前后（小到中等差别）
        ('i', 'ɪ'): 0.25, ('e', 'ɛ'): 0.25, ('u', 'ʊ'): 0.25, ('o', 'ɔ'): 0.25,
        ('æ', 'ɛ'): 0.3, ('ɑ', 'ɔ'): 0.3, ('ɪ', 'e'): 0.4, ('ʊ', 'o'): 0.4,
        
        # 双元音变体
        ('aɪ', 'aː'): 0.4, ('aɪ', 'a'): 0.5, ('aɪ', 'ɑɪ'): 0.2,
        ('aʊ', 'aː'): 0.4, ('aʊ', 'a'): 0.5, ('ɔɪ', 'ɔ'): 0.4,
        ('eɪ', 'e'): 0.4, ('oʊ', 'o'): 0.4,
    }
    
    # 检查正反方向
    pair = (phoneme1, phoneme2)
    reverse_pair = (phoneme2, phoneme1)
    
    if pair in similarity_groups:
        return similarity_groups[pair]
    elif reverse_pair in similarity_groups:
        return similarity_groups[reverse_pair]
    else:
        # 判断是否同类（元音/辅音），否则惩罚更高
        vowels = {'i', 'ɪ', 'e', 'ɛ', 'æ', 'ɑ', 'ɔ', 'o', 'ʊ', 'u', 'ə', 'ɜ', 'ʌ', 
                 'aɪ', 'aʊ', 'ɔɪ', 'eɪ', 'oʊ', 'iː', 'uː', 'ɑː', 'ɔː', 'ɜː'}
        consonants = {'p', 'b', 't', 'd', 'k', 'g', 'f', 'v', 'θ', 'ð', 's', 'z', 
                     'ʃ', 'ʒ', 'h', 'm', 'n', 'ŋ', 'l', 'r', 'w', 'j', 'tʃ', 'dʒ'}
        
        if (phoneme1 in vowels and phoneme2 in vowels) or (phoneme1 in consonants and phoneme2 in consonants):
            return 0.7  # 同类但不同音
        else:
            return 1.0  # 不同类（元音vs辅音）

def phoneme_edit_distance(phonemes1, phonemes2):
    """
    用音素距离计算两个音素序列的编辑距离
    """
    m = len(phonemes1)
    n = len(phonemes2)
    
    # 构建DP表，记录代价和操作
    dp = [[(0, "")] * (n + 1) for _ in range(m + 1)]
    
    # 初始化首行首列
    for i in range(m + 1):
        dp[i][0] = (i, "delete")
    for j in range(n + 1):
        dp[0][j] = (j, "insert")
    
    # 填充DP表
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            phoneme1 = phonemes1[i-1]
            phoneme2 = phonemes2[j-1]
            
            substitution_cost = get_phonetic_distance(phoneme1, phoneme2)
            
            costs = {
                "replace": dp[i-1][j-1][0] + substitution_cost,
                "delete": dp[i-1][j][0] + 1.0,
                "insert": dp[i][j-1][0] + 1.0
            }
            
            if substitution_cost == 0:
                min_op = "match"
            else:
                min_op = min(costs, key=costs.get)

            dp[i][j] = (costs.get(min_op, dp[i-1][j-1][0]), min_op)

    # 回溯得到对齐和操作
    alignment = []
    operations = []
    i, j = m, n
    
    while i > 0 or j > 0:
        cost, op = dp[i][j]
        
        if op == "match":
            alignment.append((phonemes1[i-1], phonemes2[j-1], "correct"))
            i -= 1
            j -= 1
        elif op == "replace":
            alignment.append((phonemes1[i-1], phonemes2[j-1], "incorrect"))
            operations.append(f'Replace {phonemes1[i-1]} with {phonemes2[j-1]}')
            i -= 1
            j -= 1
        elif op == "delete":
            alignment.append((phonemes1[i-1], None, "delete"))
            operations.append(f'Delete {phonemes1[i-1]}')
            i -= 1
        elif op == "insert":
            alignment.append((None, phonemes2[j-1], "insert"))
            operations.append(f'Insert {phonemes2[j-1]}')
            j -= 1

    alignment.reverse()
    operations.reverse()
    
    return dp[m][n][0], operations, alignment

def is_completely_different_word(reality_phonemes, standard_phonemes):
    """
    判断两个单词是否完全不同（应得0分）
    用多种标准检测无关单词
    """
    if len(reality_phonemes) == 0 or len(standard_phonemes) == 0:
        return False
        
    # 标准1：有无任何共同音素
    reality_set = set(reality_phonemes)
    standard_set = set(standard_phonemes)
    common_phonemes = reality_set.intersection(standard_set)
    
    # 没有共同音素，基本是完全不同单词
    if len(common_phonemes) == 0:
        print("1. 无共同音素，判为完全不同")
        return True
    
    # 标准2：相似度比例低于20%
    total_unique_phonemes = len(reality_set.union(standard_set))
    similarity_ratio = len(common_phonemes) / total_unique_phonemes
    
    if similarity_ratio < 0.2:  # 低于20%相似
        print("2. 相似度低于20%，判为完全不同")
        return True
    
    # 标准3：长度差异过大
    min_length = min(len(reality_phonemes), len(standard_phonemes))
    max_length = max(len(reality_phonemes), len(standard_phonemes))
    
    # 一个单词长度超过另一个3倍，基本不同
    if max_length / min_length > 3.0:
        print("3. 长度差异过大，判为完全不同")
        return True
    
    # 标准4：位置相似性，是否有音素在相近位置匹配
    position_matches = 0
    max_check_length = min(len(reality_phonemes), len(standard_phonemes))
    
    for i in range(max_check_length):
        if reality_phonemes[i] == standard_phonemes[i]:
            position_matches += 1
        # 允许±1位置容忍
        elif i > 0 and reality_phonemes[i] == standard_phonemes[i-1]:
            position_matches += 0.5
        elif i < max_check_length-1 and reality_phonemes[i] == standard_phonemes[i+1]:
            position_matches += 0.5
    
    # 没有位置匹配且共同音素极少，判为完全不同
    if position_matches == 0 and len(common_phonemes) <= 1:
        print("4. 无位置匹配且共同音素极少，判为完全不同")
        return True
        
    return False

def handle_connected_speech(phonemes, word_boundaries=None):
    """
    处理连读、同化、省略等语流现象
    参数:
        phonemes: 音素列表
        word_boundaries: 单词边界索引列表（可选）
    返回:
        应用连读规则后的音素列表
    """
    if len(phonemes) < 2:
        return phonemes
    
    result = phonemes.copy()
    
    # 常见连读规则
    for i in range(len(result) - 1):
        current = result[i]
        next_phoneme = result[i + 1]
        
        # 规则1: /t/ + 辅音，常被省略或变为喉塞音
        if current == 't' and next_phoneme in ['b', 'p', 'd', 'g', 'k', 'f', 'v', 's', 'z', 'ʃ', 'ʒ', 'm', 'n', 'l', 'r']:
            result[i] = 'ʔ'  # 喉塞音或可视为省略
        
        # 规则2: /d/ + /j/ -> /dʒ/ (如 would you -> wouldja)
        if current == 'd' and next_phoneme == 'j':
            result[i] = 'dʒ'
            result[i + 1] = ''  # 标记删除
        
        # 规则3: /t/ + /j/ -> /tʃ/ (如 get you -> getcha)
        if current == 't' and next_phoneme == 'j':
            result[i] = 'tʃ'
            result[i + 1] = ''  # 标记删除
        
        # 规则4: /s/ + /j/ -> /ʃ/ (如 miss you -> mishu)
        if current == 's' and next_phoneme == 'j':
            result[i] = 'ʃ'
            result[i + 1] = ''  # 标记删除
        
        # 规则5: /z/ + /j/ -> /ʒ/ (如 as you -> azhu)
        if current == 'z' and next_phoneme == 'j':
            result[i] = 'ʒ'
            result[i + 1] = ''  # 标记删除
        
        # 规则6: 连读/r/，单词以/r/结尾且下一个单词以元音开头
        if current == 'r' and next_phoneme in ['a', 'e', 'i', 'o', 'u', 'æ', 'ɛ', 'ɪ', 'ɔ', 'ʊ', 'ə', 'ɜ', 'ʌ']:
            # 保留/r/用于连读
            pass
        
        # 规则7: 插入/j/，/i/或/iː/后接元音
        if current in ['i', 'iː'] and next_phoneme in ['a', 'e', 'o', 'u', 'æ', 'ɛ', 'ɔ', 'ʊ', 'ə', 'ɜ', 'ʌ']:
            # 插入连读/j/
            result.insert(i + 1, 'j')
        
        # 规则8: 插入/w/，/u/、/uː/、/ʊ/、/oʊ/后接元音  
        if current in ['u', 'uː', 'ʊ', 'oʊ'] and next_phoneme in ['a', 'e', 'i', 'o', 'æ', 'ɛ', 'ɪ', 'ɔ', 'ə', 'ɜ', 'ʌ']:
            # 插入连读/w/
            result.insert(i + 1, 'w')
    
    # 移除被标记删除的空字符串
    result = [p for p in result if p != '']
    
    return result

def generate_connected_speech_variants(phonemes_list):
    """
    生成音素序列的所有可能连读变体
    参数:
        phonemes_list: 拼接后得到的音素列表
    返回:
        所有可能连读变体的音素列表
    """
    variants = []
    
    # 原始版本
    variants.append(phonemes_list)
    
    # 应用连读规则后的版本
    connected_version = handle_connected_speech(phonemes_list)
    if connected_version != phonemes_list:
        variants.append(connected_version)
    
    # 常见短语连读特殊变体
    phoneme_string = ''.join(phonemes_list)
    
    # 针对特定短语的连读变体
    if 'nɒtæt' in phoneme_string:  # "not at"
        variants.append(phoneme_string.replace('nɒtæt', 'nɒtət'))
    
    if 'ætliːst' in phoneme_string:  # "at least"  
        variants.append(phoneme_string.replace('ætliːst', 'ətliːst'))
    
    if 'lɛtmi' in phoneme_string:  # "let me"
        variants.append(phoneme_string.replace('lɛtmi', 'lɛmi'))
    
    if 'wʌtdu' in phoneme_string:  # "what do"
        variants.append(phoneme_string.replace('wʌtdu', 'wʌdə'))
    
    return [parse_phonemes(v) if isinstance(v, str) else v for v in variants]

def get_score_level(score):
    """
    根据得分返回星级（0-3）
    """
    if score >= 90:
        return 3  # 优秀
    elif score >= 75:
        return 2  # 良好
    elif score >= 60:
        return 1  # 及格
    else:
        return 0  # 需要努力

def calculate_phrase_score(reality_phone: str, standard_phone: str, is_phrase=False):
    """
    计算短语发音得分，考虑连读现象
    参数:
        reality_phone: 用户发音的音标字符串
        standard_phone: 标准发音的音标字符串  
        is_phrase: 是否为短语（决定是否启用连读分析）
    返回:
        得分、操作列表、星级和高亮音素列表
    """
    if not is_phrase:
        # 单词直接用原始算法
        return calculate_phone_sent_score(reality_phone, standard_phone)
    
    # 短语则考虑连读
    reality_phonemes = parse_phonemes_phrase(reality_phone.strip())
    standard_phonemes = parse_phonemes_phrase(standard_phone.strip())
    
    # 生成标准发音的所有连读变体
    standard_variants = generate_connected_speech_variants(standard_phonemes)
    
    best_score = 0
    best_operations = []
    best_alignment = []
    
    # 尝试与每个变体比对
    for variant in standard_variants:
        # 先判断是否完全不同单词
        if is_completely_different_word(reality_phonemes, variant):
            continue
            
        # 计算与该变体的得分
        distance, operations, alignment = phoneme_edit_distance(reality_phonemes, variant)
        
        max_length = max(len(reality_phonemes), len(variant))
        if max_length == 0:
            score = 100.0
        elif distance >= max_length * 0.9:
            score = 0.0
        else:
            normalized_distance = distance / max_length
            penalty_factor = normalized_distance ** 0.7
            score = 100.0 * (1 - penalty_factor * 0.6)
            score = max(score, 0.0)
        
        # 保留最高得分
        if score > best_score:
            best_score = score
            best_operations = operations
            best_alignment = alignment
    
    # 如果没有合适的变体，回退到原始算法
    if best_score == 0:
        return calculate_phone_sent_score(reality_phone, standard_phone)
    
    final_score = round(best_score, 2)
    score_level = get_score_level(final_score)
    highlighted_phonemes = get_highlighted_phonemes(reality_phonemes, standard_phonemes, best_alignment)

    return final_score, best_operations, score_level, highlighted_phonemes

def get_highlighted_phonemes(reality_phonemes, standard_phonemes, alignment):
    """
    根据对齐结果，生成带高亮信息的标准音素列表
    """
    highlighted_phonemes = []
    for reality_phoneme, standard_phoneme, op in alignment:
        if op == "correct":
            highlighted_phonemes.append({"phoneme": standard_phoneme, "status": "correct"})
        elif op == "incorrect":
            highlighted_phonemes.append({"phoneme": standard_phoneme, "status": "incorrect", "reality": reality_phoneme})
        elif op == "delete":
            # This case needs to be handled carefully. 
            # The phoneme is in the standard but not in reality.
            # We can decide whether to show it and how.
            # Option 1: Add to the list with a "deleted" status.
            # highlighted_phonemes.append({"phoneme": reality_phoneme, "status": "deleted"})
            pass # Option 2: Don't show deleted phonemes in the standard sequence.
        elif op == "insert":
            highlighted_phonemes.append({"phoneme": standard_phoneme, "status": "inserted"})

    return highlighted_phonemes

def calculate_phone_sent_score(reality_phone: str, standard_phone: str):
    """
    用音素级分析和音素距离计算发音得分
    返回:
        得分(0-100), 操作列表, 星级(0-3), 高亮音素列表
    """
    reality_phone = reality_phone.strip()
    standard_phone = standard_phone.strip()
    
    # 解析为音素
    reality_phonemes = parse_phonemes_phrase(reality_phone)
    standard_phonemes = parse_phonemes_phrase(standard_phone)
    
    # 判断是否完全不同单词
    if is_completely_different_word(reality_phonemes, standard_phonemes):
        return 0.0, ['完全不同的单词'], 0, []
    
    # 计算音素编辑距离
    distance, operations, alignment = phoneme_edit_distance(reality_phonemes, standard_phonemes)
    
    # 生成高亮音素
    highlighted_phonemes = get_highlighted_phonemes(reality_phonemes, standard_phonemes, alignment)

    # 改进的打分函数，扣分更宽容
    max_length = max(len(reality_phonemes), len(standard_phonemes))
    if max_length == 0:
        return 100.0, [], 3, []
    
    if distance >= max_length * 0.9:
        return 0.0, operations, 0, highlighted_phonemes
    
    # 更宽容的扣分方式：用0.7次方减缓惩罚
    normalized_distance = distance / max_length
    
    # 惩罚因子
    penalty_factor = normalized_distance ** 0.7  # 比平方根还宽容
    
    # 完美得100分，逐步递减，最多扣60%
    raw_score = 100.0 * (1 - penalty_factor * 0.6)
    
    # 不设最低分，完全错误可为0
    score = max(raw_score, 0.0)
    
    final_score = round(score, 2)
    score_level = get_score_level(final_score)
    
    return final_score, operations, score_level, highlighted_phonemes

# Test function to demonstrate improvements
def test_algorithm():
    """Test cases to show the improvements"""
    
    test_cases = [
        ("laɪk", "laɪk"),      # Perfect match
        ("lɪk", "laɪk"),       # Diphthong -> monophthong
        ("tʃɔɪs", "tʃoɪs"),    # Minor vowel change
        ("sink", "θɪŋk"),      # th -> s substitution
        ("bɪt", "bit"),        # Vowel length
        ("kæt", "bæt"),        # Different consonant
        ("kɒntrɪbju:ʃən","/ˌkɒntrɪˈbjuːʃ(ə)n/"), # (), : /ˌkɒntrɪˈbjuːʃ(ə)n/
        ("dɪskrɪmɪneɪt","/dɪˈskrɪmɪneɪt/"),
        ("prənʌnsieɪʃn","/prəˌnʌnsiˈeɪʃn/"),
        ("wɜːk", "wɔːk"),      # work vs walk - similar but different vowels
        ("ʃi", "hæpi"),        # she vs happy - completely different words
        ("dɪs", "ðɪs"),        # dis vs this - th->d substitution (common L2 error)
        ("stɹeɪndʒ", "stɹeɪn"), # strange vs strain - extra phonemes at end
        ("læf", "læfɪŋ"),      # laugh vs laughing - missing ending
    ]

    # Test phrase cases with connected speech
    phrase_test_cases = [
        ("nəʊrθəmerəkən","/nɔːθ əˈmɛrɪkən/"),
        ("wətsðəmætər","/wɒts ðə ˈmætə/")
    ]
    
    print("=== Algorithm Comparison ===")
    print("\n--- Single Word Tests ---")
    for reality, standard in test_cases:
        new_score, new_ops, new_level, highlighted_phonemes = calculate_phone_sent_score(reality, standard)
        
        print(f"\nInput: '{reality}' vs Standard: '{standard}'")
        print(f"New Score: {new_score}/100, Stars: {new_level}")
        print(f"Operations: {new_ops[:2]}")
        print(f"Highlighted Phonemes: {highlighted_phonemes}")

    
    print("\n--- Phrase Tests (with connected speech) ---")
    for reality, standard in phrase_test_cases:
        phrase_score, phrase_ops, phrase_level, highlighted_phonemes = calculate_phrase_score(reality, standard, is_phrase=True)
        single_score, single_ops, single_level, single_highlighted = calculate_phone_sent_score(reality, standard)
        
        print(f"\nPhrase: '{reality}' vs Standard: '{standard}'")
        print(f"With connected speech: {phrase_score}/100, Stars: {phrase_level}")
        print(f"Operations (phrase): {phrase_ops[:2]}")
        print(f"Highlighted Phonemes (phrase): {highlighted_phonemes}")
        print(f"Without connected speech: {single_score}/100, Stars: {single_level}")
        print(f"Operations (single): {single_ops[:2]}")
        print(f"Highlighted Phonemes (single): {single_highlighted}")
        print(f"Improvement: +{phrase_score - single_score:.1f} points")

if __name__ == "__main__":
    test_algorithm()