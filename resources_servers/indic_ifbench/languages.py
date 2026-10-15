# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Versioned lexical and orthographic policy for the translated IFBench tasks.

These are explicit benchmark inventories, not a morphological/POS analyzer.
They are intentionally local and deterministic. See README.md for limitations.
"""

import unicodedata
from dataclasses import dataclass

import regex


@dataclass(frozen=True)
class Language:
    alphabet: str
    vowels: str
    virama: str
    pronouns: frozenset[str]
    conjunctions: frozenset[str]
    stopwords: frozenset[str]


# Ordered letter inventories follow the translated prompts' wrap from ha to a.
# Dependent vowel signs are mapped to the corresponding independent vowel by
# Unicode name; the final consonant's inherent A is counted unless suppressed.
_ORTHOGRAPHY = {
    "bn": ("অআইঈউঊঋএঐওঔকখগঘঙচছজঝঞটঠডঢণতথদধনপফবভমযরলশষসহ", "অআইঈউঊঋৠঌৡএঐওঔ", "্"),
    "gu": ("અઆઇઈઉઊઋએઐઓઔકખગઘઙચછજઝઞટઠડઢણતથદધનપફબભમયરલવશષસહ", "અઆઇઈઉઊઋૠઌૡએઐઓઔઍઑ", "્"),
    "hi": ("अआइईउऊऋएऐओऔकखगघङचछजझञटठडढणतथदधनपफबभमयरलवशषसह", "अआइईउऊऋॠऌॡएऐओऔऍऑ", "्"),
    "kn": ("ಅಆಇಈಉಊಋಎಏಐಒಓಔಕಖಗಘಙಚಛಜಝಞಟಠಡಢಣತಥದಧನಪಫಬಭಮಯರಲವಶಷಸಹ", "ಅಆಇಈಉಊಋೠಌೡಎಏಐಒಓಔ", "್"),
    "mr": ("अआइईउऊऋएऐओऔकखगघङचछजझञटठडढणतथदधनपफबभमयरलळवशषसह", "अआइईउऊऋॠऌॡएऐओऔऍऑ", "्"),
    "ml": ("അആഇഈഉഊഋഎഏഐഒഓഔകഖഗഘങചഛജഝഞടഠഡഢണതഥദധനപഫബഭമയരലവശഷസഹ", "അആഇഈഉഊഋൠഌൡഎഏഐഒഓഔ", "്"),
    "ne": ("अआइईउऊऋएऐओऔकखगघङचछजझञटठडढणतथदधनपफबभमयरलवशषसह", "अआइईउऊऋॠऌॡएऐओऔ", "्"),
    "or": ("ଅଆଇଈଉଊଋଏଐଓଔକଖଗଘଙଚଛଜଝଞଟଠଡଢଣତଥଦଧନପଫବଭମଯରଲଵଶଷସହ", "ଅଆଇଈଉଊଋୠଌୡଏଐଓଔ", "୍"),
    "pa": ("ਅਆਇਈਉਊਏਐਓਔਕਖਗਘਙਚਛਜਝਞਟਠਡਢਣਤਥਦਧਨਪਫਬਭਮਯਰਲਵੜਸਹ", "ਅਆਇਈਉਊਏਐਓਔ", "੍"),
    "ta": ("அஆஇஈஉஊஎஏஐஒஓஔகஙசஞடணதநபமயரலவழளறனஜஷஸஹ", "அஆஇஈஉஊஎஏஐஒஓஔ", "்"),
    "te": ("అఆఇఈఉఊఋఎఏఐఒఓఔకఖగఘఙచఛజఝఞటఠడఢణతథదధనపఫబభమయరలవశషసహ", "అఆఇఈఉఊఋౠఌౡఎఏఐఒఓఔ", "్"),
    "ur": ("", "", ""),  # The source excludes alphabet/vowel/conjunct/syllable tasks for Urdu.
}

# Closed inventories: inflected forms count only when explicitly listed. No
# English stopword/POS fallback is used for any language.
_LEXICONS = {
    "bn": (
        "আমি আমাকে আমার আমরা আমাদের তুমি তোমাকে তোমার তোমরা তোমাদের আপনি আপনাকে আপনার আপনারা সে তাকে তার তারা তাদের তিনি তাঁকে তাঁর তাঁরা এটি এটা এই ওটা যা কে নিজে নিজের",
        "এবং ও আর কিন্তু অথবা কিংবা তবু তাই অথচ নয়তো",
        "এক একটি এই ওই সেই যে যা কি কী না নয় হয় আছে ছিল হবে থেকে জন্য সঙ্গে মধ্যে উপর নিচে খুব আরও শুধু সব কোনো যদি তবে কারণ",
    ),
    "gu": (
        "હું મને મારું મારો મારી અમે અમને અમારું અમારો અમારી તું તને તારું તારો તારી તમે તમને તમારું તમારો તમારી તે તેને તેનું તેનો તેની તેઓ તેમને તેમનું તેમનો તેમની આ એ પોતે પોતાનું કોણ શું જે",
        "અને પણ અથવા કે પરંતુ છતાં એટલે તેમજ નહિંતર",
        "એક આ એ તે જે છે હતો હતી હતા હશે નથી ના નું ની નો ને થી માટે પર માં સાથે પણ તો જો જ્યારે કારણ બધું કોઈ વધુ જ",
    ),
    "hi": (
        "मैं मुझे मुझको मेरा मेरी मेरे हम हमें हमको हमारा हमारी हमारे तू तुझे तेरा तेरी तेरे तुम तुम्हें तुम्हारा तुम्हारी तुम्हारे आप आपको आपका आपकी आपके वह वो उसे उसको उसका उसकी उसके वे उन्हें उनका उनकी उनके यह ये इसे इसका इसकी इसके कौन किस कोई कुछ खुद स्वयं अपना अपनी अपने",
        "और तथा एवं लेकिन पर परंतु किन्तु किंतु या अथवा बल्कि इसलिए",
        "एक है हैं था थी थे होगा होगी होंगे हो होने होना का की के को से में पर तक लिए साथ भी ही तो नहीं न यदि जब तब क्योंकि बहुत सभी सब कोई कुछ जो कि",
    ),
    "kn": (
        "ನಾನು ನನ್ನ ನನಗೆ ನಾವು ನಮ್ಮ ನಮಗೆ ನೀನು ನಿನ್ನ ನಿನಗೆ ನೀವು ನಿಮ್ಮ ನಿಮಗೆ ಅವನು ಅವನ ಅವನಿಗೆ ಅವಳು ಅವಳ ಅವಳಿಗೆ ಅವರು ಅವರ ಅವರಿಗೆ ಅದು ಅದರ ಅದಕ್ಕೆ ಇವನು ಇವಳು ಇವರು ಇದು ಇದರ ಇದಕ್ಕೆ ಅವು ಇವು ತನ್ನ ತಾನು ಯಾರು ಏನು ಯಾವ",
        "ಮತ್ತು ಹಾಗೂ ಆದರೆ ಅಥವಾ ಇಲ್ಲವೇ ಆದರೂ ಆದ್ದರಿಂದ ಹಾಗಾಗಿ",
        "ಒಂದು ಈ ಆ ಇದು ಅದು ಇದೆ ಇವೆ ಇತ್ತು ಆಗಿದೆ ಇಲ್ಲ ಅಲ್ಲ ಎಂದು ಎಂಬ ನಲ್ಲಿ ಮೇಲೆ ಕೆಳಗೆ ಗೆ ಇಂದ ಅನ್ನು ಜೊತೆಗೆ ಕೂಡ ಮಾತ್ರ ಎಲ್ಲಾ ಕೆಲವು ಯಾವಾಗ ಏಕೆಂದರೆ",
    ),
    "mr": (
        "मी मला माझा माझी माझे आम्ही आम्हाला आमचा आमची आमचे तू तुला तुझा तुझी तुझे तुम्ही तुम्हाला तुमचा तुमची तुमचे तो त्याला त्याचा त्याची त्याचे ती तिला तिचा तिची तिचे ते त्यांना त्यांचा त्यांची त्यांचे हा ही हे याला याचा याची याचे आपण आपला आपली आपले स्वतः कोण काय",
        "आणि व पण परंतु किंवा अथवा तरी म्हणून तसेच नाहीतर",
        "एक आहे आहेत होता होती होते असे असा अशी असेल नाही न चे चा ची ला ने मध्ये वर पासून साठी बरोबर ही देखील तर जर जेव्हा कारण सर्व काही कोणते",
    ),
    "ml": (
        "ഞാൻ എന്നെ എന്റെ ഞങ്ങൾ ഞങ്ങളെ ഞങ്ങളുടെ നാം നമ്മെ നമ്മുടെ നീ നിന്നെ നിന്റെ നിങ്ങൾ നിങ്ങളെ നിങ്ങളുടെ അവൻ അവനെ അവന്റെ അവൾ അവളെ അവളുടെ അവർ അവരെ അവരുടെ അത് അതിനെ അതിന്റെ ഇത് ഇതിനെ ഇതിന്റെ ഇവൻ ഇവൾ ഇവർ അവർ താൻ തന്നെ തന്റെ ആരാണ് എന്ത് ആര്",
        "ഒപ്പം കൂടാതെ എന്നാൽ അല്ലെങ്കിൽ പക്ഷേ എങ്കിലും അതിനാൽ അതുകൊണ്ട്",
        "ഒരു ഈ ആ ഇത് അത് ആണ് ആയിരുന്നു ആകുന്നു ഉണ്ട് ഇല്ല അല്ല എന്ന് എന്ന കൊണ്ട് വേണ്ടി കൂടെ മാത്രം പോലും എല്ലാ ചില എങ്കിൽ കാരണം കൂടുതൽ വളരെ",
    ),
    "ne": (
        "म मलाई मेरो मेरी मेरा हामी हामीलाई हाम्रो हाम्री हाम्रा तँ तिमी तिमीलाई तिम्रो तिम्री तिम्रा तपाईं तपाईँ तपाईंलाई तपाईँलाई तपाईंको तपाईँको ऊ उ उसलाई उसको उसकी उसका उनी उनलाई उनको उनकी उनका उनीहरू उनीहरुलाई उनीहरूलाई उनीहरूको यो यसलाई यसको यी त्यो त्योको को के आफू आफ्नो आफ्नी आफ्ना",
        "र अनि तथा तर वा अथवा तैपनि त्यसैले साथै नत्र",
        "एक यो त्यो यी ती हो हुन् छ छन् थियो थिई थिए हुनेछ होइन छैन को का की लाई ले मा बाट देखि सम्म लागि सँग पनि नै मात्र सबै केही यदि जब किनभने धेरै",
    ),
    "or": (
        "ମୁଁ ମୋତେ ମୋର ଆମେ ଆମକୁ ଆମର ତୁମେ ତୁମକୁ ତୁମର ଆପଣ ଆପଣଙ୍କୁ ଆପଣଙ୍କର ସେ ତାକୁ ତାର ସେମାନେ ସେମାନଙ୍କୁ ସେମାନଙ୍କର ଏହା ଏହାର ଏମାନେ ନିଜେ ନିଜର କିଏ କଣ ଯିଏ",
        "ଏବଂ ଓ ଆଉ କିନ୍ତୁ ଅଥବା କିମ୍ବା ତଥାପି ତେଣୁ ନଚେତ",
        "ଏକ ଏହି ସେହି ଏହା ତାହା ଯାହା ଅଛି ଅଛନ୍ତି ଥିଲା ଥିଲେ ହେବ ନାହିଁ ନୁହେଁ ର କୁ ରେ ଠାରୁ ପାଇଁ ସହିତ ଉପରେ ମଧ୍ୟ କେବଳ ସବୁ କିଛି ଯଦି କାରଣ",
    ),
    "pa": (
        "ਮੈਂ ਮੈਨੂੰ ਮੇਰਾ ਮੇਰੀ ਮੇਰੇ ਅਸੀਂ ਸਾਨੂੰ ਸਾਡਾ ਸਾਡੀ ਸਾਡੇ ਤੂੰ ਤੈਨੂੰ ਤੇਰਾ ਤੇਰੀ ਤੇਰੇ ਤੁਸੀਂ ਤੁਹਾਨੂੰ ਤੁਹਾਡਾ ਤੁਹਾਡੀ ਤੁਹਾਡੇ ਉਹ ਉਸਨੂੰ ਉਸਦਾ ਉਸਦੀ ਉਸਦੇ ਉਹਨਾਂ ਉਨ੍ਹਾਂ ਇਹ ਇਸਨੂੰ ਇਸਦਾ ਇਸਦੀ ਇਸਦੇ ਆਪ ਆਪਣਾ ਆਪਣੀ ਆਪਣੇ ਕੌਣ ਕੀ",
        "ਅਤੇ ਤੇ ਪਰ ਜਾਂ ਲੇਕਿਨ ਪ੍ਰੰਤੂ ਸਗੋਂ ਇਸਲਈ ਨਹੀਂਤਾਂ",
        "ਇੱਕ ਇਹ ਉਹ ਜੋ ਕਿ ਹੈ ਹਨ ਸੀ ਸਨ ਹੋਵੇਗਾ ਨਹੀਂ ਨਾ ਦਾ ਦੀ ਦੇ ਨੂੰ ਨੇ ਵਿੱਚ ਉੱਤੇ ਤੋਂ ਲਈ ਨਾਲ ਵੀ ਹੀ ਤਾਂ ਜੇ ਜਦੋਂ ਕਿਉਂਕਿ ਸਭ ਕੁਝ ਬਹੁਤ ਹੋਰ",
    ),
    "ta": (
        "நான் என்னை என் எனது நாம் நாங்கள் எங்களை எங்கள் நமது நீ உன்னை உன் உனது நீங்கள் உங்களை உங்கள் உங்களது அவன் அவனை அவனது அவள் அவளை அவளது அவர் அவரை அவரது அவர்கள் அவர்களை அவர்களின் அது அதை அதன் இது இதை இதன் இவன் இவள் இவர் இவர்கள் தன் தனது தாம் யார் என்ன",
        "மற்றும் ஆனால் அல்லது எனினும் ஆகவே எனவே மேலும் இல்லையெனில்",
        "ஒரு இந்த அந்த இது அது என்று என்ற உள்ளது உள்ளன இருந்தது இருக்கிறது இல்லை அல்ல இல் இருந்து வரை மீது கீழ் உடன் மட்டும் கூட எல்லாம் சில போது ஏனெனில் மிகவும் இன்னும்",
    ),
    "te": (
        "నేను నన్ను నా మేము మమ్మల్ని మా మనం మనల్ని మన నువ్వు నిన్ను నీ మీరు మిమ్మల్ని మీ అతను అతన్ని అతని ఆమె ఆమెను ఆమెది వారు వారిని వారి అది దాన్ని దాని ఇది దీన్ని దీని ఇతను ఈమె వీరు ఇవి అవి తాను తన తనను ఎవరు ఏమి",
        "మరియు కానీ లేదా గాని అయినా అందువల్ల కాబట్టి అలాగే లేకపోతే",
        "ఒక ఈ ఆ ఇది అది అని అనే ఉంది ఉన్నాయి ఉన్నది లేదు కాదు లో పై నుండి వరకు కోసం తో కూడా మాత్రమే అన్ని కొన్ని అయితే ఎందుకంటే చాలా ఇంకా",
    ),
    "ur": (
        "میں مجھے میرا میری میرے ہم ہمیں ہمارا ہماری ہمارے تو تجھے تیرا تیری تیرے تم تمہیں تمہارا تمہاری تمہارے آپ آپکو آپکا آپکی آپکے وہ اسے اسکو اسکا اسکی اسکے انہیں انکو انکا انکی انکے یہ اسے اس خود اپنا اپنی اپنے کون کیا کوئی کچھ",
        "اور یا مگر لیکن بلکہ تاہم لہذا نیز ورنہ",
        "ایک یہ وہ جو کہ ہے ہیں تھا تھی تھے ہوگا ہوگی ہوں ہو ہونے ہونا کا کی کے کو سے میں پر تک لیے ساتھ بھی ہی تو نہیں نہ اگر جب تب کیونکہ بہت سب کوئی کچھ",
    ),
}

LANGUAGE_RULES = {
    code: Language(
        *_ORTHOGRAPHY[code],
        frozenset(pronouns.split()),
        frozenset(conjunctions.split()),
        frozenset((stopwords + " " + pronouns + " " + conjunctions).split()),
    )
    for code, (pronouns, conjunctions, stopwords) in _LEXICONS.items()
}


def normalize(text: str) -> str:
    """Canonicalize equivalent Unicode encodings without removing vowel marks."""
    return unicodedata.normalize("NFC", text).casefold()


def words(text: str) -> list[str]:
    """Unicode letter/number tokens, preserving marks and joiners inside words."""
    return regex.findall(r"[\p{L}\p{N}][\p{L}\p{M}\p{N}\u200c\u200d]*", normalize(text))


def aksharas(text: str) -> list[str]:
    """Grapheme units with virama-linked consonants kept in one Indic unit."""
    units: list[str] = []
    for cluster in regex.findall(r"\X", normalize(text)):
        previous = units[-1].rstrip("\u200c\u200d") if units else ""
        # Some scripts' viramas are not Unicode grapheme linkers (e.g. Tamil).
        if previous and "VIRAMA" in unicodedata.name(previous[-1], "") and cluster[0].isalpha():
            units[-1] += cluster
        else:
            units.append(cluster)
    return units


def sentences(text: str) -> list[str]:
    """Split on Latin/Indic/Urdu terminators, retaining punctuation and decimals."""
    return [
        match.group().strip()
        for match in regex.finditer(r"(?:[^.!?।॥۔؟]|(?<=\d)\.(?=\d))+(?:[.!?।॥۔؟]+|$)", text)
        if match.group().strip()
    ]


def vowel_nuclei(word: str, language: str) -> list[str]:
    """Written vowel nuclei; a deterministic orthographic syllable approximation."""
    rules = LANGUAGE_RULES[language]
    nuclei = []
    for unit in aksharas(word):
        signs = [
            unicodedata.name(c, "").split("VOWEL SIGN ")[-1] for c in unit if "VOWEL SIGN " in unicodedata.name(c, "")
        ]
        independent = [unicodedata.name(c).split("LETTER ")[-1] for c in unit if c in rules.vowels]
        if signs:
            nuclei.extend(signs)
        elif independent:
            nuclei.extend(independent)
        elif any(c in rules.alphabet and c not in rules.vowels for c in unit):
            if not unit.rstrip("\u200c\u200d").endswith(rules.virama):
                nuclei.append("A")
        elif unit in "aeiou":
            nuclei.append(unit.upper())
    return nuclei
