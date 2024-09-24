lexical_var_non_digit_values = {
    'positif': r"(?i)([¦|]?positifs?|[¦|]pos?i?t?\b|[¦|]?positiv?e?s?|\bpos\b|[^a-zA-Z0-9]+(?:\+|p)[^a-zA-Z0-9]*$|^\+)",
    'negatif': r"(?i)([¦|]?negatifs?|[¦|]neg?a?\b|[¦|]?negati?v?e?s?|\bneg\b|[^a-zA-Z0-9]+(?:\-|n)[^a-zA-Z0-9]*$|^\-|^pas\sd[e'])",
    'augmente': r"(?i)(augmentee?s?|aug)",
    'dimunue': r"(?i)(diminuee?s?|dim)",
    'depasse': r"(?i)(depassee?s?)",
    'normal': r"(?i)(normale?s?|normaux?|correcte?s?|stable?s?|normo|normes?|\bnle\b)",
    'eleve': r"(?i)(elevee?s?)",
    'basse': r"(?i)(basse?s?|\bbas\b)",
    'absent': r"(?i)(absente?s?|absences?|indetectables?)",
    'present': r"(?i)(presente?s?|presences?)",
    'non significatif': r"(?i)(non\s*sign?i?f?i?c?a?t?i?f?|\bns\b)",
    'physiologique': r"(?i)(physiol?o?g?i?q?u?e?s?|physio)",
    'fluctuante': r"(?i)(fluctuante?s?|fluctuen?t?e?s?|fluctuee?s?)",
    'hemolyse': r"(?i)(hemolysee?s?)",
}

